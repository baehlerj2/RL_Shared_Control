# Reinforcement Learning for Shared Control

This repo is a copy of the [OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs) repository  extended with the work on reinforcement learning for shared control. The repository is designed to be used with isaac sim version 2022.2.1.

## Installation

First follow the documentation of the original repository to install the correct isaac sim version.

To install omniisaacgymenvs, clone this repository:
```console
git clone git@github.com:baehlerj2/RL_Shared_Control.git
```

Set a `PYTHON_PATH` variable in the terminal that links to the python executable, 

```
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
```

Install `omniisaacgymenvs` as a python module for `PYTHON_PATH`. Change directory to the root of this repo and run:

```bash
PYTHON_PATH -m pip install -e .
```

## Training

cd to omniisaacgymenvs
```bash
PYTHON_PATH scripts/rlgames_train.py task=KayaVel pipeline=cpu
```
More information about the task setup is provided in /omniisaacgymenvs

## Exporting Neural Networks to ONNX
for models without lstm:
```bash
PYTHON_PATH scripts/rlgames_onnx_normalized.py task=KayaVel test=True checkpoint=CHECKPOINT_PATH pipeline=cpu
```

for models with lstm:
```bash
PYTHON_PATH scripts/rlgames_onnx_normalized_lstm.py task=KayaVel test=True checkpoint=CHECKPOINT_PATH pipeline=cpu
```

## Ridgeback Gazebo
Install the ridgeback simulation package as described [here](https://www.clearpathrobotics.com/assets/guides/kinetic/ridgeback/simulation.html).

Some adjustments have to be made to the description files:

in ```/opt/ros/noetic/share/ridgeback_description/urdf/accessories/hokuyo_ust-10lx_mount.urdf.xacro```, adjust the resolution of the scan , the min/max angles and the min max range to the following values (lines 44-56):
```urdf
<scan>
    <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-3.1415</min_angle>
        <max_angle>3.1415</max_angle>
    </horizontal>
</scan>
<range>
    <min>0.20</min>
    <max>2.5</max>
    <resolution>0.01</resolution>
</range>
```

For the usage of the POV camera, add this to the main xacro file (/opt/ros/noetic/share/ridgeback_description/urdf/ridgeback.urdf.xacro):

```
<joint name="camera_joint" type="fixed">
    <parent link="chassis_link"/>
    <child link="camera_link"/>
    <origin xyz="0.305 0 1.0" rpy="0 0 0"/>
</joint>
 
<link name="camera_link">
    <visual>
        <geometry>
            <box size="0.010 0.03 0.03"/>
        </geometry>
        <material name="red"/>
    </visual>
</link>
 
<joint name="camera_optical_joint" type="fixed">
    <parent link="camera_link"/>
    <child link="camera_link_optical"/>
    <origin xyz="0 0 0" rpy="${-pi/2} 0 ${-pi/2}"/>
</joint>

<link name="camera_link_optical"></link>

<gazebo reference="camera_link">
    <material>Gazebo/Red</material>

    <sensor name="camera" type="wideanglecamera">
        <pose> 0 0 0 0 0 0 </pose>
        <visualize>true</visualize>
        <update_rate>100</update_rate>
        <camera>
            <horizontal_fov>2.0</horizontal_fov>
            <image>
                <format>R8G8B8</format>
                <width>2000</width>
                <height>1200</height>
            </image>
            <clip>
                <near>0.05</near>
                <far>20.0</far>
            </clip>
                <lens>
                <type>gnomonical</type>
                <scale_to_hfov>true</scale_to_hfov>
                <cutoff_angle>1.5707</cutoff_angle>
                <env_texture_size>512</env_texture_size>
                </lens>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
            <frame_name>camera_link_optical</frame_name>
        </plugin>
    </sensor>
</gazebo>
```


## Simulations in Gazebo
The folder /gazebo contains the launch file and the worlds used for simulation (/gazebo/worlds). The simulation can be started with the launch file by runnning:
```console
roslaunch ridgeback_world.launch
```
The command can also be adjusted with variables to spawn ridgeback in a specific position/orientation and world, e.g.:
```console
roslaunch ridgeback_world.launch world:=door1_25.world x:=1.5 y:=1.82 yaw:=-0.35
```

#### Running Controller Nodes
There are different ros nodes to control the robot in gazebo:

First cd to ```/omniisaacgymenvs```
- to use the robot with a simulated input pointing to a specific target and an RL agent for shared control run:
    ```bash
    python3 ROS/kayavel_sim.py
    ```
    then enter the target position

    you can do the same with RDS instead of an RL agent:
    ```bash
    python3 ROS/rds_sim.py
    ```

- to control the robot with a joystick while using an RL agent run:
    ```bash
    python3 ROS/kayavel.py
    ```

- to control the robot manually with a joystick without shared control run:
    ```bash
    python3 ROS/controller_test.py
    ```


## Running on Real Wheelchair
In order to run an agent for the real wheelchair, one only needs to run the following node:

```console
cd oige22/omniisaacgymenvs
python3 ROS/kayavel.py
```

The node is implemented such that the robot only moves when the joystick is moved, so make sure that there is no joystick noise when it is not moved.

To test the node without moving the robot, simply don't publish to /cmd_vel by commenting line 205:

```python
self.base_cmd_vel_pub.publish(twist)
```

such that it looks like this:
```python
if self.base_control:
    # self.base_cmd_vel_pub.publish(twist)
    a = 0
```

Note that there are several parameters that need to be set correctly:
- make sure that the subscriber topics are set correctly, for the daav wheelchair they are:
    ```python
    self.pose_sub = rospy.Subscriber('/odom', Odometry, self.position_callback)
    self.lidar_sub = rospy.Subscriber('/lidar/input/scan', LaserScan, self.lidar_scan)
    self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
    ```
- make sure that the publisher for velocity command is correct:
    ```python
    self.base_cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=20)
    ```
- set the model path (relative to /omniisaacgymenvs):
    ```python
    self.model_path = "onnx_models/SCLFC_R2.onnx"
    ```
    You can find all other onnx models used in the paper in the folder ```/onnx_models```.
    - if the model uses lstm, set the parameter to True and set the nr of units (normally just 128):
    ```python
    self.lstm = True
    self.lstm_hidden_size = 128
    ```

- define the lidar settings, the samples are 360 for models with LCNN and 36 if without:
    ```python
    self.lidar_samples = 360 # set to 360 for models with LCNN and 36 without
    self.ranges = None 
    self.max_range = 2.5 # don't change
    self.min_range = 0.3 # don't change
    ```

- Set the velocity limits to reasonable values:
    ```python
    self.max_x_vel = 0.6
    self.max_y_vel = 0.6
    self.max_ang_vel = 2 # this is for scaling the ouput
    self.max_allowed_ang_vel = 0.5 # this is for limiting the actual angular velocity
    ```

use this command for bag recording:
```shell
rosbag record /odom /cmd_vel /lidar/input/scan /Odometry /joy /heading /desired_velocity
```