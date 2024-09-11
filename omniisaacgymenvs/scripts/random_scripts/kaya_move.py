# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.controllers.holonomic_controller import HolonomicController
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from omni.isaac.core.objects import FixedCuboid, FixedCylinder
import omni.replicator.core as rep
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.range_sensor")
from omni.isaac.range_sensor import _range_sensor
import omni.kit.commands

from pxr import Gf
import numpy as np

from omni.isaac.core.utils.stage import add_reference_to_stage
from pathlib import Path

import pdb

my_world = World(stage_units_in_meters=1.0)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
kaya_asset_path = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
my_kaya = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Kaya",
        name="my_kaya",
        wheel_dof_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
        create_robot=True,
        usd_path=kaya_asset_path,
        position=np.array([-0.3, 1.3, 0.02]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
)
#my_kaya.set_collision_enabled(False)
my_world.scene.add_default_ground_plane()

kaya_setup = HolonomicRobotUsdSetup(
    robot_prim_path=my_kaya.prim_path, com_prim_path="/World/Kaya/base_link/control_offset"
)
(
    wheel_radius,
    wheel_positions,
    wheel_orientations,
    mecanum_angles,
    wheel_axis,
    up_axis,
) = kaya_setup.get_holonomic_controller_params()
my_controller = HolonomicController(
    name="holonomic_controller",
    wheel_radius=wheel_radius,
    wheel_positions=wheel_positions,
    wheel_orientations=wheel_orientations,
    mecanum_angles=mecanum_angles,
    wheel_axis=wheel_axis,
    up_axis=up_axis,
)

POS = 1.7
LENGTH = POS*2
HEIGHT = 1.0
DEPTH = .1

#current_working_dir = Path.cwd()
#asset_path = str(current_working_dir.parent) + "/assets/jetbot"
#
#add_reference_to_stage(
#    usd_path=asset_path + "/ridgeback.usd", #"/obstacles_dynamic.usd",
#    prim_path= "World/Home/home"
#)

#wall0 = my_world.scene.add(
#    FixedCuboid(
#        prim_path="/World/wall0",
#        position = np.array([-POS, 0, 0]),
#        scale=[DEPTH, LENGTH, HEIGHT],
#        name="wall0"
#        #size=np.array([0.5, 4, 1])
#    )
#)
#wall1 = my_world.scene.add(
#    FixedCuboid(
#        prim_path="/World/wall1",
#        position = np.array([POS, 0, 0]),
#        scale=[DEPTH, LENGTH, HEIGHT],
#        name="wall1"
#        #size=np.array([0.5, 4, 1])
#    )
#)
#
#wall2 = my_world.scene.add(
#    FixedCuboid(
#        prim_path="/World/wall2",
#        position = np.array([0, POS+DEPTH/2, 0]),
#        scale=[LENGTH+DEPTH, DEPTH, HEIGHT],
#        name="wall2"
#        #size=np.array([0.5, 4, 1])
#    )
#)
#
#wall3 = my_world.scene.add(
#    FixedCuboid(
#        prim_path="/World/wall3",
#        position = np.array([0, -POS-DEPTH/2, 0]),
#        scale=[LENGTH+DEPTH, DEPTH, HEIGHT],
#        name="wall3"
#        #size=np.array([0.5, 4, 1])
#    )
#)
#
#
#POS = 0.75
#HEIGHT = 0.5
#WIDTH = .2
#my_world.scene.add(
#    FixedCylinder(
#            prim_path="/World/obs4",
#            position=[POS, POS, HEIGHT/2],
#            scale=[WIDTH, WIDTH, HEIGHT],
#            name="zyl1",
#        )
#)
#my_world.scene.add(
#    FixedCylinder(
#        prim_path="/World/obs5",
#        position=[-POS, POS, HEIGHT/2],
#        scale=[WIDTH, WIDTH, HEIGHT],
#        name="zyl2",
#    )
#)
#my_world.scene.add(
#    FixedCylinder(
#        prim_path="/World/obs6",
#        position=[POS, -POS, HEIGHT/2],
#        scale=[WIDTH, WIDTH, HEIGHT],
#        name="zyl3",
#    )
#)
#my_world.scene.add(
#    FixedCylinder(
#        prim_path="/World/obs7",
#        position=[-POS, -POS, HEIGHT/2],
#        scale=[WIDTH, WIDTH, HEIGHT],
#        name="zyl4",
#    )
#)
#my_world.scene.add(
#    FixedCylinder(
#        prim_path="/World/obs8",
#        position=[0, 0, HEIGHT/2],
#        scale=[WIDTH, WIDTH, HEIGHT]
#    )
#)
result, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/World/Kaya/base_link/Lidar",
            parent=None,
            min_range=0.15,
            max_range=10,     
            draw_points=False,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=1,
            vertical_resolution=4.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )
lidar.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.015))
lidarinterface = _range_sensor.acquire_lidar_sensor_interface()
lidarpath = "/World/Kaya/base_link/Lidar"



my_world.reset()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
            #pdb.set_trace()
        #if i >= 0 and i < 500:
        #    my_kaya.apply_wheel_actions(my_controller.forward(command=[0.1, 0.0, 0.0]))
        #elif i >= 500 and i < 1000:
        #    my_kaya.apply_wheel_actions(my_controller.forward(command=[0.0, 0.1, 0.0]))
        #elif i >= 1000 and i < 1200:
        #    my_kaya.apply_wheel_actions(my_controller.forward(command=[0.0, 0.0, 0.01]))
        #elif i == 1200:
        #    i = 0
        #i += 1
        my_kaya.apply_wheel_actions(my_controller.forward(command=[0.1, 0.0, 0.0]))
        #v = my_kaya.get_velocity()
        #print(v)
    


simulation_app.close()
