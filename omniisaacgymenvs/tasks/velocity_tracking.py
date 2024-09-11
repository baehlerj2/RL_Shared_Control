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


"""
TODO:
- add variables like episode length and collision range to config
- use @torch.jit.script to speed up functions that get called every step
- clean up code
"""


class VelTracker(RLTask):
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

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500
        self.target_margin = 0.35

        self._home_name = self._task_cfg["env"]["obstacles"]
    
        self.collision_range = 0.5 # collision termination

        self.ranges_count = 360
        self.ranges_count_min = 360
        self.min_step = 3

        # number of observation = LIDAR raanges+ distance_heading * 5 + previous (x,y_linear_vel) and z_angular_vel
        self._num_observations = int(self.ranges_count_min/self.min_step) + 2 + 3
        self._num_actions = 3

        RLTask.__init__(self, name, env)

        self.prev_heading = torch.zeros(self._num_envs).to(self._device)
        self.prev_kaya_positions = torch.zeros(self._num_envs,3).to(self._device)

        self.desired_base_velocities = torch.zeros((self._num_envs, 2)).to(self._device)

        self.base_vel_xy = torch.zeros((self._num_envs, 3)).to(self._device)
        self.ang_vel = torch.zeros((self._num_envs)).to(self._device)
        self.prev_base_vel_xy = self.base_vel_xy.clone()
        self.prev_ang_vel = self.ang_vel.clone()

        self.max_lin_vel = 0.5
        self.max_ang_vel = 0.25

        self._max_lidar_range = 2.5

        self.heading_reward_buf = torch.zeros_like(self.rew_buf)
        self.vel_reward_buf = torch.zeros_like(self.rew_buf)

        ## Training in static envs : use_flatcache = True (in task yaml file)
        
        return

    def set_up_scene(self, scene) -> None:
        self._stage = omni.usd.get_context().get_stage()
        self.get_kaya()
        #if self._home_name != 'None':
        #    self.get_home()
        RLTask.set_up_scene(self, scene)

        #if self._home_name != 'None':
        #    self._home = ArticulationView(prim_paths_expr="/World/envs/.*/Home/home", name="home_view")
        self._kayas = ArticulationView(
            prim_paths_expr="/World/envs/.*/Kaya", name="jetbot_view", reset_xform_properties=False
        )
        
        scene.add(self._kayas)

        return
    
    def get_home(self):
        current_working_dir = Path.cwd()
        asset_path = str(current_working_dir.parent) + "/assets/jetbot"

        add_reference_to_stage(
            usd_path=asset_path + "/ridgeback.usd", #"/obstacles_dynamic.usd",
            prim_path= self.default_zero_env_path + "/Home/home"
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
            min_range=0.4,
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

    def get_observations(self) -> dict:
        """Return lidar ranges and polar coordinates as observations to RL agent."""
        self.ranges = torch.zeros((self._num_envs, int(self.ranges_count/self.min_step))).to(self._device)

        for i in range(self._num_envs):
            np_ranges = self.lidarInterface.get_linear_depth_data(self._lidarpaths[i]).squeeze()
            np_ranges = np.clip(np_ranges, 0.0, self._max_lidar_range)
            reshaped_array = np_ranges.reshape(int(self.ranges_count/self.min_step), self.min_step).min(axis=1) #minimum in every #min_step
            self.ranges[i] = torch.tensor(np.nan_to_num(reshaped_array, nan=10.0))
        
        self.positions, self.rotations = self._kayas.get_world_poses()

        self.prev_base_vel_xy = self.base_vel_xy.clone()
        v = self._kayas.get_velocities()

        for i in range(self._num_envs):
            rotation_matrix = torch.tensor(quat_to_rot_matrix(self.rotations[i]))
            self.base_vel_xy[i] = torch.matmul(rotation_matrix.T, v[i, :3]) * -1.0

        base_vel_xy = v[:, :2].clip(-self.max_lin_vel, self.max_lin_vel) / self.max_lin_vel
        base_angvel_z = v[:, -1].unsqueeze(1).clip(-self.max_ang_vel, self.max_ang_vel) / self.max_ang_vel

        self.prev_ang_vel = self.ang_vel.clone()
        self.ang_vel = v[:, -1]
        
        obs = torch.hstack((self.ranges/self._max_lidar_range, self.desired_base_velocities, self.base_vel_xy[:, :2]/self.max_lin_vel, self.ang_vel.unsqueeze(-1)/self.max_ang_vel))

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

        actions[:, :2] = actions[:, :2] * self.max_lin_vel
        actions[:, 2] = actions[:, 2] * self.max_ang_vel

        indices = torch.arange(self._kayas.count, dtype=torch.int32, device=self._device)
        joint_velocities = torch.zeros(self.num_envs, 33)
        for i in range(self.num_envs):
            controls = self._controller.forward(command=actions[i].clone())
            joint_velocities[i,:3] = torch.tensor(controls.joint_velocities)

        self._kayas.set_joint_velocity_targets(joint_velocities, indices)


    def reset_idx(self, env_ids):
        # """Resetting the environment at the beginning of episode."""

        """Resetting the environment at the beginning of episode."""
        num_resets = len(env_ids)

        self.goal_reached = torch.zeros(self._num_envs, device=self._device)
        self.collisions = torch.zeros(self._num_envs, device=self._device)

        self.vel_reward_buf[env_ids] = 0.0

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

        # randomize kaya positions
        rand_root_pos, rand_target_pos = self.random_initial_positions(len(env_ids))
        self._kayas.set_world_poses(root_pos+rand_root_pos, root_rot1, indices=env_ids)
        #self._kayas.set_world_poses(root_pos, root_rot1, indices=env_ids)


        ## Uncomment to randomize initial jet_pose
        # self._kayas.set_world_poses(root_pos+self.rand_pose, root_rot1, indices=env_ids)
        #self._kayas.set_velocities(root_vel, indices=env_ids)


        self.desired_base_velocities[env_ids] = torch.rand(len(env_ids), 2) * 2 - 1

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def random_initial_positions(self, num):
        kayas = torch.zeros(num, 3)
        targets = torch.zeros(num, 3)

        DIS = 2.5
        POINTS = [torch.tensor([0.0, 0.0, 0.0]), 
                  torch.tensor([0.0, DIS, 0.0]),
                  torch.tensor([0.0, -DIS, 0.0]), 
                  torch.tensor([DIS, 0.0, 0.0]),
                  torch.tensor([DIS, DIS, 0.0]),
                  torch.tensor([DIS, -DIS, 0.0]),
                  torch.tensor([-DIS, 0.0, 0.0]),
                  torch.tensor([-DIS, -DIS, 0.0]),
                  torch.tensor([-DIS, DIS, 0.0]),
                 ]
        indexes = [i for i in range(len(POINTS))]
        for j in range(num):
            random.shuffle(indexes)
            kayas[j, :] = POINTS[indexes[0]]
            targets[j, :] = POINTS[indexes[1]]

        return kayas, targets

    def post_reset(self):
        """This is run when first starting the simulation before first episode."""
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        kaya_paths = self._kayas.prim_paths
        self._lidarpaths = [path + "/base_link/Lidar" for path in kaya_paths]


        # get some initial poses
        self.initial_root_pos, self.initial_root_rot = self._kayas.get_world_poses()

        self.dt = 1.0 / 60.0

        # randomize all envs
        indices = torch.arange(self._kayas.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    def calculate_metrics(self) -> None:
        """Calculate rewards for the RL agent."""
        rewards = torch.zeros_like(self.rew_buf)

        episode_end = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, 0.0)

        #print(self._des_lin_vel)
        #print(v)
        #print("-------------")
        #print(self._des_ang_vel)
        #print(self._kayas.get_velocities()[:, 5])
        #print("-------------")
        LIN_VEL_ERROR_WEIGHT = 1.0

        rewards -= LIN_VEL_ERROR_WEIGHT * abs(self.base_vel_xy[:, 0]/self.max_lin_vel - self.desired_base_velocities[:, 0])**2
        rewards -= LIN_VEL_ERROR_WEIGHT * abs(self.base_vel_xy[:, 1]/self.max_lin_vel - self.desired_base_velocities[:, 1])**2

        self.vel_reward_buf += rewards
        #rewards -= 10 * abs(self.headings)

        self.rew_buf[:] = rewards

    def is_done(self) -> None:
        """Flags the environnments in which the episode should end."""
        
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, self.reset_buf.double())
        resets = torch.where(self.collisions.bool(), 1.0, resets.double())
        resets = torch.where(self.goal_reached.bool(), 1.0, resets.double())

        
        self.reset_buf[:] = resets