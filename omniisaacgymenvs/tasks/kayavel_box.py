from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.kaya import Kaya
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import quat_to_euler_angles, quat_to_rot_matrix
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.range_sensor")
from omni.isaac.range_sensor import _range_sensor
import omni.kit.commands
from gym import spaces
from pxr import Gf
from pathlib import Path
import numpy as np
import torch
import math
from pxr import Gf, UsdPhysics
import omni.usd
import cv2
from scipy.spatial.transform import Rotation

from omni.isaac.core.objects import FixedCuboid, FixedCylinder, DynamicCuboid
import pdb
import random
from collections import deque

from omniisaacgymenvs.tasks.shared.capsule import Capsule


class KayaVelTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
       
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._kaya_positions = torch.tensor([0.0, 0.0, 0.1])

        self._dt = self._task_cfg["sim"]["dt"] * self._task_cfg["env"]["controlFrequencyInv"]
        self._human_dt = 0.2
        self._no_human_control_freq = self._task_cfg["sim"]["Kaya"]["update_des_velocity_automatic"]

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 10000
        self.target_margin = self._task_cfg["reward"]["target_margin"]

        self._home_name = self._task_cfg["env"]["obstacles"]

        self.ranges_count = 360
        self.ranges_count_min = 360
        self.min_step = self._task_cfg["sim"]["Kaya"]["lidar_min_step"] 

        self.collision_capsule = Capsule(
            self._task_cfg['sim']['Kaya']['collision_model']['width'], 
            self._task_cfg['sim']['Kaya']['collision_model']['length'], 
            self.min_step, 
            self._task_cfg['sim']['Kaya']['collision_model']['d_crit']
            )
        self.collision_ranges = torch.tensor(self.collision_capsule.ranges) # collision termination
        self.crit_ranges = torch.tensor(self.collision_capsule.crit_ranges) # critical distances after which we start punishing

        self.action_smoothness = self._task_cfg["sim"]["Kaya"]["action_smoothness"]

        self.step_counter = 0

        if self.action_smoothness:
            self._num_observations = int(self.ranges_count_min/self.min_step) + 1 + 2 + 3 + 6
        else:
            self._num_observations = int(self.ranges_count_min/self.min_step) + 1 + 2 + 3
        self._num_actions = 3

        RLTask.__init__(self, name, env)

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        self.target_position = torch.tensor([0.0, 0.0, 0.0]).to(self._device)

        self.desired_base_velocities = torch.zeros((self._num_envs, 3)).to(self._device)

        self.base_vel_xy = torch.zeros((self._num_envs, 3)).to(self._device)
        self.ang_vel = torch.zeros((self._num_envs)).to(self._device)
        self.prev_base_vel_xy = self.base_vel_xy.clone()
        self.prev_ang_vel = self.ang_vel.clone()

        self.max_lin_vel = self._task_cfg["sim"]["Kaya"]["max_lin_vel"]
        self.max_ang_vel = self._task_cfg["sim"]["Kaya"]["max_ang_vel"]

        self.action = torch.zeros((self._num_envs, self._num_actions)).to(self._device)
        self.old_action = torch.zeros((self._num_envs, self._num_actions)).to(self._device)
        self.old_old_action = torch.zeros((self._num_envs, self._num_actions)).to(self._device)

        self._max_lidar_range = self._task_cfg["sim"]["Kaya"]["lidar_max_range"]
        self._min_lidar_range = self._task_cfg["sim"]["Kaya"]["lidar_min_range"]

        self.no_Obs = 1.0
        self.obs_after_episodes = self._task_cfg["curriculum"]["obs_after_epochs"]
        self.episode_counter = -2*self.num_envs

        if self.test:
            self._max_episode_length = 1400

        # set this to true for manual task setup
        self.manual_task_definition = False

        self.csv_save = False
        self.csv_filename = "runs/KayaVel_Heading_ProgressReward/nn/KayaVel_Heading_ProgressReward.csv"
        if self.test and self._num_envs == 1 and False:
            self.csv_save = True
            self.csv_data = []
            self._episode = 0

        self.range_reward_buf = torch.zeros_like(self.rew_buf)
        self.heading_reward_buf = torch.zeros_like(self.rew_buf)
        self.velocity_reward_buf = torch.zeros_like(self.rew_buf)
        self.action_reward_buf = torch.zeros_like(self.rew_buf)

        self.times_since_human_control = torch.zeros(self._num_envs).to(self._device)

        self.goal_reached = torch.zeros(self._num_envs, device=self._device)
        self.collisions = torch.zeros(self._num_envs, device=self._device)

        self.extras_success = deque([], maxlen=self.num_envs)
        self.extras_collisions = deque([], maxlen=self.num_envs)
        ## Training in static envs : use_flatcache = True (in task yaml file)
        
        return

    def set_up_scene(self, scene) -> None:
        self._stage = omni.usd.get_context().get_stage()
        self.get_kaya()
        self.add_target()
        self.get_home()
        self.get_box()
        RLTask.set_up_scene(self, scene)

        self._home = ArticulationView(prim_paths_expr="/World/envs/.*/Home/home", name="home_view")
        self._kayas = ArticulationView(
            prim_paths_expr="/World/envs/.*/Kaya", name="jetbot_view", reset_xform_properties=False
        )
        self._boxes = RigidPrimView(prim_paths_expr="/World/envs/.*/box", name="boxes_view")
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/Target/target", name="targets_view")
        self._targets._non_root_link = True
        
        scene.add(self._kayas)
        scene.add(self._targets)

        return
    
    def get_home(self):
        current_working_dir = Path.cwd()
        asset_path = str(current_working_dir.parent) + "/assets/jetbot"

        add_reference_to_stage(
            usd_path=asset_path + "/bigwalls.usd", #"/obstacles_dynamic.usd",
            prim_path= self.default_zero_env_path + "/Home/home"
        )

    def get_box(self):
        FixedCuboid(
            prim_path=self.default_zero_env_path + "/box",
            position = np.array([0, 0, 0.25]),
            scale=[1, 1, 0.5],
            #size=np.array([0.5, 4, 1])
        )
    
    def get_kaya(self):
        kaya = Kaya(
            prim_path=self.default_zero_env_path + "/Kaya", name="Kaya", translation=self._kaya_positions
        )

        self._sim_config.apply_articulation_settings(
            "Kaya", get_prim_at_path(kaya.prim_path), self._sim_config.parse_actor_config("Kaya")
        )
        result, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=self.default_zero_env_path + "/Kaya/base_link/Lidar",
            parent=None,
            min_range=self._min_lidar_range,
            max_range=self._max_lidar_range,     
            draw_points=False,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=360/self.ranges_count, 
            vertical_resolution=4.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )
        lidar.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.015))
        if not result:
            raise Exception("lidar creation failed")
        self._controller = kaya._controller

    def add_target(self):
        target = DynamicCuboid(prim_path=self.default_zero_env_path + "/Target/target",
            name="target",
            position=self.target_position,
            scale=np.array([.1, .1, .1]),
            color=np.array([.125,.82,0.22]))
        
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path),
                                                     self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def get_observations(self) -> dict:
        """Return lidar ranges and polar coordinates as observations to RL agent."""
        self.ranges = torch.zeros((self._num_envs, int(self.ranges_count/self.min_step))).to(self._device)

        for i in range(self._num_envs):
            np_ranges = self.lidarInterface.get_linear_depth_data(self._lidarpaths[i]).squeeze()
            np_ranges = np.clip(np_ranges, 0.0, self._max_lidar_range)
            reshaped_array = np_ranges.reshape(int(self.ranges_count/self.min_step), self.min_step).min(axis=1) #minimum in every #min_step
            self.ranges[i] = torch.tensor(np.nan_to_num(reshaped_array, nan=10.0))
        
        self.positions, self.rotations = self._kayas.get_world_poses()
        self.target_positions, _ = self._targets.get_world_poses()

        self.yaws = []
        for rot in self.rotations:
            self.yaws.append(quat_to_euler_angles(rot)[2])
        self.yaws = torch.tensor(self.yaws).to(self._device)
        self.yaws = torch.where(self.yaws <= 0.0, self.yaws + math.pi, self.yaws - math.pi)
        goal_angles = torch.atan2(self.target_positions[:,1] - self.positions[:,1], self.target_positions[:,0] - self.positions[:,0])

        self.headings = goal_angles - self.yaws
        self.headings = torch.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
        self.headings = torch.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings) 

        self.goal_distances = torch.linalg.norm(self.positions - self.target_positions, dim=1).to(self._device)

        to_target = self.target_positions - self.positions
        to_target[:, 2] = 0.0

        self.prev_base_vel_xy = self.base_vel_xy.clone()
        v = self._kayas.get_velocities()
        for i in range(self._num_envs):
            rotation_matrix = torch.tensor(quat_to_rot_matrix(self.rotations[i]))
            self.base_vel_xy[i] = torch.matmul(rotation_matrix.T, v[i, :3]) * -1.0

        self.prev_ang_vel = self.ang_vel.clone()
        self.ang_vel = v[:, -1]

        # set desired base velocities
        for i in range(self._num_envs):
            if self._no_human_control_freq or self.times_since_human_control[i] >= self._human_dt or self.times_since_human_control[i] == 0.0:
                self.desired_base_velocities[i, :] = self.target_positions[i, :] - self.positions[i, :]
                self.desired_base_velocities[i, 2] = 0.0

                rotation_matrix = torch.tensor(quat_to_rot_matrix(self.rotations[i]))
                rotated_velocities = torch.matmul(rotation_matrix.T, self.desired_base_velocities[i])
                rotated_velocities = rotated_velocities / torch.norm(rotated_velocities)

                self.desired_base_velocities[i] = rotated_velocities * -1.0
                self.times_since_human_control[i] = 0.0

            self.times_since_human_control[i] += self._dt

        self.velocity_headings = torch.atan2(self.desired_base_velocities[:,1], self.desired_base_velocities[:,0]).unsqueeze(1)
        self.velocity_headings = torch.where(self.velocity_headings > math.pi, self.velocity_headings - 2 * math.pi, self.velocity_headings)
        self.velocity_headings = torch.where(self.velocity_headings < -math.pi, self.velocity_headings + 2 * math.pi, self.velocity_headings)
        
        obs_headings = torch.clamp(self.velocity_headings / math.pi, min=-1.0, max=1.0)
        obs_base_velocities = torch.clamp(self.base_vel_xy[:, :2] / self.max_lin_vel, min=-1.0, max=1.0)
        obs_ang_vel = torch.clamp(self.ang_vel.unsqueeze(-1) / self.max_ang_vel, min=-1.0, max=1.0)

        if self.action_smoothness:
            obs = torch.hstack((self.ranges / self._max_lidar_range, obs_headings, self.desired_base_velocities[:, :2], obs_base_velocities, obs_ang_vel, self.action, self.old_action))
        else:
            obs = torch.hstack((self.ranges / self._max_lidar_range, obs_headings, self.desired_base_velocities[:, :2], obs_base_velocities, obs_ang_vel))

        #print('OBSERVATION ------------------------------------ OBSERVATION')
        #print(f'desired_base_velocities: {self.desired_base_velocities}')
        #print(f'unrotated base velocities: {v[:, :3]}')
        #print(f'base_velocities: {self.base_vel_xy}')
        #print(f'obs_base_velocities: {obs_base_velocities}')
        #print(f'headings: {self.headings}')
        #print(f'velocity_headings: {self.velocity_headings}')
        #print(f'action: {self.action}')
        #print('OBSERVATION ------------------------------------ OBSERVATION')

        self.obs_buf[:] = obs

        observations = {
            self._kayas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        """Perform actions to move the robot."""
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        self.old_old_action = self.old_action.clone()
        self.old_action = self.action.clone()
        self.action = actions.clone()

        # adjust actions to reasonable values
        actions[:, :2] = actions[:, :2] * self.max_lin_vel #+ torch.tensor([0.1, 0])
        actions[:, 2] = actions[:, 2] * self.max_ang_vel

        indices = torch.arange(self._kayas.count, dtype=torch.int32, device=self._device)
        joint_velocities = torch.zeros(self.num_envs, 33)
        for i in range(self.num_envs):
            controls = self._controller.forward(command=actions[i].clone())
            joint_velocities[i,:3] = torch.tensor(controls.joint_velocities)

        self._kayas.set_joint_velocity_targets(joint_velocities, indices)

    def reset_idx(self, env_ids):
        """Resetting the environment at the beginning of episode."""
        num_resets = len(env_ids)
        if self.csv_save:
            self.write_csv(self.csv_filename, self.csv_data)
            self._episode += 1

        # update success rate
        for env_id in env_ids:
            env_id = env_id.item()
            self.extras_success.append(self.goal_reached[env_id])
            self.extras_collisions.append(self.collisions[env_id])
        self.extras['success'] = np.mean(self.extras_success)
        self.extras['collisions'] = np.mean(self.extras_collisions)

        self.goal_reached = torch.zeros(self._num_envs, device=self._device)
        self.collisions = torch.zeros(self._num_envs, device=self._device)

        if self.test:
            for i in range(env_ids.shape[0]):
                print(f'range reward of env {env_ids[i]}: {self.range_reward_buf[env_ids[i]]}')
                print(f'heading reward of env {env_ids[i]}: {self.heading_reward_buf[env_ids[i]]}')
                print(f'velocity reward of env {env_ids[i]}: {self.velocity_reward_buf[env_ids[i]]}')
                print(f'action reward of env {env_ids[i]}: {self.action_reward_buf[env_ids[i]]}')
                print('------------------------------------')

        self.range_reward_buf[env_ids] = 0.0
        self.heading_reward_buf[env_ids] = 0.0
        self.velocity_reward_buf[env_ids] = 0.0
        self.action_reward_buf[env_ids] = 0.0

        self.action[env_ids] = torch.zeros((num_resets, self._num_actions)).to(self._device)
        self.old_action[env_ids] = torch.zeros((num_resets, self._num_actions)).to(self._device)
        self.old_old_action[env_ids] = torch.zeros((num_resets, self._num_actions)).to(self._device)

        # apply resets
        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]

        ## Uncomment to randomize initial yaw
        rotations = Rotation.from_quat(root_rot)
        euler_angles = rotations.as_euler('xyz')
        random_yaw_values = np.random.uniform(0, 2 * np.pi, size=(len(root_rot),))
        euler_angles[:, 0] = random_yaw_values
        rotations = Rotation.from_euler('xyz', euler_angles)
        quaternions = rotations.as_quat()
        root_rot1 = torch.tensor(quaternions, dtype=torch.float32)

        # randomize kaya positions and orientations
        rand_root_pos, rand_target_pos = self.random_initial_positions(len(env_ids))
        target_pos = self.initial_target_pos[env_ids]

        if self.manual_task_definition:
            for id in env_ids:
                id = id.item()
                x_r = input(f'Enter x position for kaya {id}: ')
                y_r = input(f'Enter y position for kaya {id}: ')
                x_t = input(f'Enter x position for target {id}: ')
                y_t = input(f'Enter y position for target {id}: ')
                rand_root_pos[id] = torch.tensor([float(x_r), float(y_r), 0.0])
                rand_target_pos[id] = torch.tensor([float(x_t), float(y_t), 0.0])

        self._kayas.set_world_poses(root_pos+rand_root_pos, root_rot1, indices=env_ids)
        self._targets.set_world_poses(target_pos+rand_target_pos, indices=env_ids)

        # set box size
        scales = torch.ones(num_resets, 3)
        scales[:, 0] = torch.rand(num_resets) * 3.0 + 1.0
        scales[:, 1] = torch.rand(num_resets) + 1.0
        scales[:, 2] = 0.5
        self._boxes.set_local_scales(scales, indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def random_initial_positions(self, num):
        kayas = torch.zeros(num, 3)
        targets = torch.zeros(num, 3)

        X_DIS = 1.0
        Y_DIS = 2.5
        kayas[:, 0] = torch.rand(num) * 2 * X_DIS - X_DIS
        kayas[:, 1] = Y_DIS

        targets[:, 0] = torch.rand(num) * 2 * X_DIS - X_DIS
        targets[:, 1] = -Y_DIS

        return kayas, targets

    def post_reset(self):
        """This is run when first starting the simulation before first episode."""
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        kaya_paths = self._kayas.prim_paths
        self._lidarpaths = [path + "/base_link/Lidar" for path in kaya_paths]

        # get some initial poses
        self.initial_root_pos, self.initial_root_rot = self._kayas.get_world_poses()
        self.initial_target_pos, _ = self._targets.get_world_poses()

        num = int(self.no_Obs * self._num_envs)
        if num > 0:
            ics = torch.Tensor([i for i in range(num)]).long()
            home_pos = self.initial_root_pos[ics].clone()
            home_pos = home_pos + torch.tensor([200.0, 200.0, 0.0])
            self._home.set_world_poses(home_pos, indices=ics)
            self._boxes.set_world_poses(home_pos, indices=ics)

        self.dt = 1.0 / 40.0

        # randomize all envs
        indices = torch.arange(self._kayas.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)



    def calculate_metrics(self) -> None:
        """Calculate rewards for the RL agent."""
        rewards = torch.zeros_like(self.rew_buf)

        range_punishment = torch.zeros_like(self.rew_buf)
        heading_punishment = torch.zeros_like(self.rew_buf)
        vel_error_punishment = torch.zeros_like(self.rew_buf)
        action_punishment = torch.zeros_like(self.rew_buf)

        collision_ranges = self.ranges[:] - self.collision_ranges
        _, indices = torch.min(collision_ranges, 1)
        for i in range(self.num_envs):
            self.collisions[i] = torch.where(self.ranges[i, indices[i]] < self.collision_ranges[indices[i]], 1.0, 0.0).to(self._device)

        self.goal_reached = torch.where(self.goal_distances < self.target_margin, 1, 0).to(self._device)

        episode_end = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, 0.0)

        speed = torch.where(self.goal_reached.bool(),self._max_episode_length - self.progress_buf,0)

        r_crit = self._task_cfg['reward']['r_crit']
        r_col = self._task_cfg['reward']['r_col']
        r_c = self._task_cfg['reward']['r_c']

        crit_distances = self.ranges[:] - self.crit_ranges
        _, crit_indices = torch.min(crit_distances, 1)
        for i in range(self.num_envs):
            range_punishment[i] = torch.where(self.ranges[i, crit_indices[i]] < self.crit_ranges[crit_indices[i]], r_c + r_crit*abs(self.crit_ranges[crit_indices[i]]-self.ranges[i, crit_indices[i]]), 0.0)
        range_punishment += r_col * self.collisions
        self.range_reward_buf[:] += range_punishment
        
        heading_punishment = torch.where(abs(self.headings) > self._task_cfg['reward']['h_threashold'], -self._task_cfg['reward']['h_w'] * abs(self.headings)**2, 0.0)
        self.heading_reward_buf[:] += heading_punishment

        r_a = self._task_cfg['reward']['r_a']
        l = self._task_cfg['reward']['lambda']

        vel_error = torch.norm(self.action[:, :1] - self.desired_base_velocities[:, :1], dim=1)
        vel_error_punishment = r_a * torch.exp(l * vel_error**2)
        self.velocity_reward_buf[:] += vel_error_punishment

        # reward for action smoothness
        action_weight = self._task_cfg['reward']['a_s']
        action_error = self.action - self.old_action
        second_order_error = self.action - 2 * self.old_action + self.old_old_action
        action_punishment += action_weight * torch.norm(action_error, dim=1)**2 + action_weight * torch.norm(second_order_error, dim=1)**2
        self.action_reward_buf[:] += action_punishment

        if self.action_smoothness:
            rewards += action_punishment

        rewards += vel_error_punishment
        rewards += range_punishment
        rewards += heading_punishment

        #print('REWARD ------------------------------------ REWARD')
        #print(f'desired_base_velocities: {self.desired_base_velocities}')
        #print(f'base_velocities: {self.base_vel_xy}')
        #print(f'headings: {self.headings}')
        #print(f'x error: {-LIN_VEL_ERROR_WEIGHT * abs(self.base_vel_xy[:,0]/self.max_lin_vel - self.desired_base_velocities[:, 0])**2}')
        #print(f'y error: {-LIN_VEL_ERROR_WEIGHT * abs(self.base_vel_xy[:,1]/self.max_lin_vel - self.desired_base_velocities[:, 1])**2}')
        #print(f'y vel error: {-LIN_VEL_ERROR_WEIGHT * abs(self.desired_base_velocities[:,1])}')
        ##print(f'heading error: {-1 * abs(self.headings)}')
        ##print(f'progress_dis: {100 * progress_dis}')
        #print(f'min_range1: {min_range1}')
        #print(f'min_range2: {min_range2}')
        #print(f'min_range3: {min_range3}')  
        #print(f'rewards: {rewards}')
        #print('REWARD ------------------------------------ REWARD')

        self.rew_buf[:] = rewards

        self.step_counter += 1

        if self.csv_save:
            new_row = [self._episode, self.base_vel_xy[0,0].item()/self.max_lin_vel, self.base_vel_xy[0,1].item()/self.max_lin_vel, self.ang_vel[0].item()/self.max_ang_vel, self.desired_base_velocities[0,0].item(), self.desired_base_velocities[0,1].item(), self.collisions[0].item(), self.goal_reached[0].item(), episode_end[0].item(), min_range1[0].item(), rewards[0].item()]
            self.csv_data.append(new_row)

        if self.test:
            for i in range(self._num_envs):
                if self.goal_reached[i]:
                    print(f'env {i} reached goal')
                if self.collisions[i]:
                    print(f'env {i} collided')
                if episode_end[i]:
                    print(f'env {i} reached max episode length')


    def is_done(self) -> None:
        """Flags the environnments in which the episode should end."""
        
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, self.reset_buf.double())
        resets = torch.where(self.collisions.bool(), 1.0, resets.double())
        #resets = torch.where(self.goal_reached.bool(), 1.0, resets.double())
        self.reset_targets()

        
        self.reset_buf[:] = resets

    def write_csv(self, filename, data) -> None:
        np.savetxt(filename, data, delimiter=",", fmt='%s')

    def reset_targets(self) -> None:
        """Reset the target position."""
        if torch.any(self.goal_reached):
            indices = torch.where(self.goal_reached)
            target_positions = self.random_initial_positions(len(indices[0]))[1]
            robot_positions = self.positions[indices] - self.initial_root_pos[indices]
            target_positions[:, 1] = torch.where(robot_positions[:, 1] > 0.0, target_positions[:, 1], -target_positions[:, 1])
            self._targets.set_world_poses(self.initial_target_pos[indices] + target_positions, indices=indices[0])