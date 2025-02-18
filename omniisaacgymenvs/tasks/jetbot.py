from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.jetbot import Jetbot
from omniisaacgymenvs.robots.articulations.kaya import Kaya
from omni.isaac.core.articulations import ArticulationView
from omniisaacgymenvs.robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import quat_to_euler_angles
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


"""
TODO:
- add variables like episode length and collision range to config
- use @torch.jit.script to speed up functions that get called every step
- clean up code
"""



class JetbotTask(RLTask):
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
        self._jetbot_positions = torch.tensor([1.5, 1.5, 0.1])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 1200
        self.target_margin = 0.35 
        self.collision_range = 0.25 # collision termination

        self.ranges_count = 360
        self.ranges_count_min = 360
        self.min_step = 3

        # number of observation = LIDAR raanges+ distance_heading * 5 + previous (x,y_linear_vel) and z_angular_vel
        self._num_observations = int(self.ranges_count_min/self.min_step) + 2*5 + 3  
        self._num_actions = 2


        self._diff_controller = DifferentialController(name="simple_control",wheel_radius=0.0325, wheel_base=0.1125)
        RLTask.__init__(self, name, env)
        

        self.action_space = spaces.Box(low=np.array([0.1, -0.5]), high=np.array([0.5, 0.5]), dtype=np.float32)

        # init tensors that need to be set to correct device
        self.prev_dyn_obstacle1_pos = torch.zeros(self._num_envs,3).to(self._device)
        self.direction1 = torch.ones(self._num_envs).to(self._device)
        self.prev_dyn_obstacle2_pos = torch.zeros(self._num_envs,3).to(self._device)
        self.direction2 = torch.ones(self._num_envs).to(self._device)

        self.prev_goal_distance = torch.zeros(self._num_envs).to(self._device)
        self.prev_heading = torch.zeros(self._num_envs).to(self._device)
        self.prev_jetbot_positions = torch.zeros(self._num_envs,3).to(self._device)
        self.target_position = torch.tensor([-1.5, -1.5, 0.0]).to(self._device)

        ## Training in static envs : use_flatcache = True (in task yaml file)
        
        return

    def set_up_scene(self, scene) -> None:
        self._stage = omni.usd.get_context().get_stage()
        self.get_jetbot()
        self.add_target()
        self.get_home()
        RLTask.set_up_scene(self, scene)
        self._home = ArticulationView(prim_paths_expr="/World/envs/.*/Home/home", name="home_view")
        self._jetbots = ArticulationView(prim_paths_expr="/World/envs/.*/Kaya", name="jetbot_view")
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/Target/target", name="targets_view")
        self._targets._non_root_link = True
        # scene.add(self._home)
        scene.add(self._jetbots)
        scene.add(self._targets)
        return

    def get_jetbot(self):
        #jetbot = Jetbot(prim_path=self.default_zero_env_path + "/Jetbot/jetbot", name="Jetbot", translation=self._jetbot_positions)
        jetbot = Kaya(prim_path=self.default_zero_env_path + "/Kaya", name="Jetbot", translation=self._jetbot_positions)
        self._sim_config.apply_articulation_settings("Jetbot", get_prim_at_path(jetbot.prim_path), self._sim_config.parse_actor_config("Jetbot"))
        result, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=self.default_zero_env_path + "/Kaya/chassis/Lidar",
            #path=self.default_zero_env_path + "/Jetbot/jetbot/chassis/Lidar/Lidar",
            parent=None,
            min_range=0.15,
            max_range=2,     
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


    def add_target(self):
        target = DynamicCuboid(prim_path=self.default_zero_env_path + "/Target/target",
            name="target",
            position=self.target_position,
            scale=np.array([.1, .1, .1]),
            color=np.array([.125,.82,0.22]))
        
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path),
                                                     self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)
        

    def get_home(self):
        current_working_dir = Path.cwd()
        asset_path = str(current_working_dir.parent) + "/assets/jetbot"

        add_reference_to_stage(
            usd_path=asset_path + "/obstacles.usd", #"/obstacles_dynamic.usd",
            prim_path=self.default_zero_env_path + "/Home/home"
        )

    def get_observations(self) -> dict:
        """Return lidar ranges and polar coordinates as observations to RL agent."""
        self.ranges = torch.zeros((self._num_envs, int(self.ranges_count/self.min_step))).to(self._device)

        for i in range(self._num_envs):
            np_ranges = self.lidarInterface.get_linear_depth_data(self._lidarpaths[i]).squeeze()
            reshaped_array = np_ranges.reshape(int(self.ranges_count/self.min_step), self.min_step).min(axis=1) #minimum in every #min_step
            self.ranges[i] = torch.tensor(reshaped_array)
            # self.ranges[i] = torch.tensor(np_ranges) # for 360 ranges

        
        self.positions, self.rotations = self._jetbots.get_world_poses()
        self.target_positions, _ = self._targets.get_world_poses()

        self.yaws = []
        for rot in self.rotations:
            self.yaws.append(quat_to_euler_angles(rot)[2])
        self.yaws = torch.tensor(self.yaws).to(self._device)
        goal_angles = torch.atan2(self.target_positions[:,1] - self.positions[:,1], self.target_positions[:,0] - self.positions[:,0])

        self.headings = goal_angles - self.yaws

        self.headings = torch.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
        self.headings = torch.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings)        
        self.goal_distances = torch.linalg.norm(self.positions - self.target_positions, dim=1).to(self._device)

        to_target = self.target_positions - self.positions
        to_target[:, 2] = 0.0

        self.prev_potentials[:] = self.potentials.clone()
        self.potentials[:] = -torch.norm(to_target, p=2, dim=-1) / self.dt

        v= self._jetbots.get_velocities()
        base_vel_xy = v[:, :2]
        base_angvel_z = v[:, -1].unsqueeze(1)

        obs = torch.hstack((self.ranges, self.headings.unsqueeze(1), self.headings.unsqueeze(1), self.headings.unsqueeze(1), self.headings.unsqueeze(1), self.headings.unsqueeze(1)
                            , self.goal_distances.unsqueeze(1), self.goal_distances.unsqueeze(1), self.goal_distances.unsqueeze(1), self.goal_distances.unsqueeze(1), self.goal_distances.unsqueeze(1),base_vel_xy,base_angvel_z))

        self.obs_buf[:] = obs

        observations = {
            self._jetbots.name: {
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

        indices = torch.arange(self._jetbots.count, dtype=torch.int32, device=self._device)
        
        controls = torch.zeros((self._num_envs, 33))

        for i in range(self._num_envs):
            controls[i,:2] = self._diff_controller.forward([actions[i][0].item(), actions[i][1].item()])

        self._jetbots.set_joint_velocity_targets(controls, indices=indices)






    def reset_idx(self, env_ids):
        # """Resetting the environment at the beginning of episode."""

        """Resetting the environment at the beginning of episode."""
        num_resets = len(env_ids)

        self.goal_reached = torch.zeros(self._num_envs, device=self._device)
        self.collisions = torch.zeros(self._num_envs, device=self._device)


        ## Uncomment to randomize initial jet_pose based on the occupancy_map
        # current_working_dir = Path.cwd()
        # occupancy_path = str(current_working_dir) + "/imgs/new_cost2024.png"
        # binary_array = (cv2.imread(occupancy_path, cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
        # indices_of_ones = np.argwhere(binary_array == 1)
        # self.selected_index = tuple(indices_of_ones[np.random.choice(len(indices_of_ones))])
        # center = np.array(binary_array.shape) // 2
        # self.final_index = (self.selected_index - center)/2
        # self.rand_pose = torch.zeros(3)
        # self.rand_pose[0],self.rand_pose[1],self.rand_pose[2] = self.final_index[0], self.final_index[1],0

        # apply resets
        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]

        ## Uncomment to randomize initial jet_yaw
        rotations = Rotation.from_quat(root_rot)
        # euler_angles = rotations.as_euler('xyz')
        # random_yaw_values = np.random.uniform(0, 2 * np.pi, size=(len(root_rot),))
        # euler_angles[:, 0] = random_yaw_values
        # rotations = Rotation.from_euler('xyz', euler_angles)

        quaternions = rotations.as_quat()
        root_rot1 = torch.tensor(quaternions, dtype=torch.float32)

        root_vel = torch.zeros((num_resets, 6), device=self._device)
        self._jetbots.set_world_poses(root_pos, root_rot1, indices=env_ids)

        ## Uncomment to randomize initial jet_pose
        # self._jetbots.set_world_poses(root_pos+self.rand_pose, root_rot1, indices=env_ids)
        self._jetbots.set_velocities(root_vel, indices=env_ids)
        target_pos = self.initial_target_pos[env_ids] 


        self._targets.set_world_poses(target_pos, indices=env_ids)
        to_target = target_pos - self.initial_root_pos[env_ids]
  
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self):
        """This is run when first starting the simulation before first episode."""
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        jetbot_paths = self._jetbots.prim_paths
        self._lidarpaths = [path + "/base_link/Lidar" for path in jetbot_paths]


        # get some initial poses
        self.initial_root_pos, self.initial_root_rot = self._jetbots.get_world_poses()
        self.initial_target_pos, _ = self._targets.get_world_poses()

        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # randomize all envs
        indices = torch.arange(self._jetbots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)



    def calculate_metrics(self) -> None:
        """Calculate rewards for the RL agent."""
        rewards = torch.zeros_like(self.rew_buf)

        closest_ranges, indices = torch.min(self.ranges, 1)
        self.collisions = torch.where(closest_ranges < self.collision_range, 1.0, 0.0).to(self._device)

        ## The closer to target, the greater the reward 
        progress_dis = self.prev_goal_distance - self.goal_distances

        self.prev_goal_distance = self.goal_distances
        self.goal_reached = torch.where(self.goal_distances < 0.30, 1, 0).to(self._device)


        ## The closer to obstacles, the greater the penalty 
        min_range1 = torch.where(torch.abs(closest_ranges) < 0.40, -1, 0)
        min_range2 = torch.where(torch.abs(closest_ranges) < 0.35, -2, 0)
        min_range3 = torch.where(torch.abs(closest_ranges) < 0.30, -2, 0)

        self.prev_heading = self.headings

        progress_reward = self.potentials - self.prev_potentials

        episode_end = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, 0.0)

        speed = torch.where(self.goal_reached.bool(),self._max_episode_length - self.progress_buf,0)


        rewards += min_range1
        rewards += min_range2
        rewards += min_range3
        rewards -= 30 * self.collisions
        rewards -= 30 * episode_end
        rewards += speed * 0.02
        rewards += progress_dis*10
        rewards += 0.1 * progress_reward
        rewards += 30 * self.goal_reached

        self.rew_buf[:] = rewards

    def is_done(self) -> None:
        """Flags the environnments in which the episode should end."""
        
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, self.reset_buf.double())
        resets = torch.where(self.collisions.bool(), 1.0, resets.double())
        resets = torch.where(self.goal_reached.bool(), 1.0, resets.double())

        
        self.reset_buf[:] = resets
