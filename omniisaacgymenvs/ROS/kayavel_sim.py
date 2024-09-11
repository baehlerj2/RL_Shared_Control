import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist, TransformStamped, Vector3Stamped
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import onnxruntime as ort
import numpy as np
from sensor_msgs.msg import LaserScan, Joy
import math
from numpy import inf
import tf2_geometry_msgs
import pdb
import sys
from omniisaacgymenvs.tasks.shared.capsule import Capsule

'''
This script drives the robot to a target position using a trained model and simulated user input.

Run the script and input the target position (from robot perspective) as requested.
'''

class KayaNode:
    def __init__(self, target_pos=np.array([5.0, 0.0, 0.0])):

        self.pose_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.position_callback)
        #self.vel_sub = rospy.Subscriber('/ridgeback_velocity_controller/odom', Odometry, self.velocity_callback)        
        self.lidar_sub = rospy.Subscriber('/front/scan', LaserScan, self.lidar_scan)

        self.base_cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=20)
        self.target_pub = rospy.Publisher("/target", PointStamped, queue_size=20)
        self.heading_pub = rospy.Publisher("/heading", Float64, queue_size=20)
        self.des_vel_pub = rospy.Publisher("/desired_velocity", Vector3Stamped, queue_size=20)
        self.joy_pub = rospy.Publisher("/bluetooth_teleop/joy", Joy, queue_size=20)

        self.model_path = "onnx_models/paper/KayaVel_AllTasks_SDCM_-1_-1_0d9-1d75_frozenLCNN_cont.onnx"
        self.lstm = True
        self.lstm_hidden_size = 128
        self.ort_model = ort.InferenceSession(self.model_path)
        self.base_control = True

        self.lin_vel_command = Vector3Stamped()

        self.target_pos = target_pos
        self.desired_base_vel = np.array([0.0, 0.0, 0.0])
        self.target_margin = 0.35
        self.goal_distances = None
        self.goal_reached = False

        self.lidar_samples = 360
        self.ranges = None
        self.max_range = 2.5

        self.base_position = None
        self.base_yaw = None
        self.base_linear_vel = None
        self.base_angular_vel = None

        self.action = np.zeros(3)
        self.old_action = np.zeros(3)

        self.max_lin_vel = 2/3 #1.0
        self.max_ang_vel = 2 #1.7

        self.max_allowed_ang_vel = self.max_ang_vel # this is for limiting the actual angular velocity

        self.moving_target = False

        self.backwards = False
        self.mirroring = 1.0

        self.activation_mode = False
        self.manual = True
        self.activation_range = 1.0
        self.deactivation_range = 1.3

        self.out_state = np.zeros((1, 1, self.lstm_hidden_size)).astype(np.float32)
        self.hidden_state = np.zeros((1, 1, self.lstm_hidden_size)).astype(np.float32)

        rospy.Timer(rospy.Duration(1/40.0), self.send_control)
        rospy.Timer(rospy.Duration(1/40.0), self.update_base_pose)

        self.collision_capsule = Capsule(
            0.65, 
            0.3, 
            int(360/self.lidar_samples), 
            0.1,
            )
        self.collision_ranges = np.array((self.collision_capsule.ranges)) # collision termination
        
    def position_callback(self, msg):
        self.base_position = msg.pose.pose.position
        self.base_orientation = msg.pose.pose.orientation
        orientation_list = [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.base_yaw = yaw
        self.base_linear_vel = msg.twist.twist.linear
        self.base_angular_vel = msg.twist.twist.angular.z

    def velocity_callback(self, msg):
        self.base_linear_vel = msg.twist.twist.linear
        self.base_angular_vel = msg.twist.twist.angular.z

    def update_base_pose(self,timer_event):
        x1, y1, _ = self.target_pos
        if self.base_position is None:
            return
        x2, y2 = self.base_position.x, self.base_position.y
        self.goal_distances = abs(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        if self.goal_distances <= 0.35:
            self.base_control = False
            print("Goal reached!", "distance:", self.goal_distances)
            x = float(input("Target x: "))
            y = float(input("Target y: "))
            self.target_pos = np.array([x, y, 0.0])
            self.base_control = True

    def lidar_scan(self, msg1):
        self.ranges = msg1.ranges[:]

    def send_control(self, timer_event):
        if self.ranges is None or self.base_angular_vel is None or self.base_position is None:
            return
        
        # update target position
        if self.moving_target and self.base_control:
            x, y, _ = self.target_pos
            x += 1/40 * 0.2
            self.target_pos = np.array([x, y, 0.0])
        # publish target position
        if self.base_control:
            point = PointStamped()
            point.point.x = self.target_pos[0]
            point.point.y = self.target_pos[1]
            point.point.z = self.target_pos[2]
            self.target_pub.publish(point)

        x1, y1, _ = self.target_pos
        x2, y2 = self.base_position.x, self.base_position.y

        self.desired_base_vel[0] = x1 - x2
        self.desired_base_vel[1] = y1 - y2
        unrotated_velocities = self.desired_base_vel[:2] / np.linalg.norm(self.desired_base_vel[:2])

        # Convert desired_base_vel to the rotated frame
        rotated_desired_base_vel = np.array([self.desired_base_vel[0], self.desired_base_vel[1]])
        print(f'desired velocity: {rotated_desired_base_vel}')
        print(f'base yaw: {self.base_yaw}')
        rotation_matrix = np.array([[math.cos(self.base_yaw), math.sin(self.base_yaw)],
                        [-math.sin(self.base_yaw), math.cos(self.base_yaw)]])
        rotated_desired_base_vel = np.dot(rotation_matrix, rotated_desired_base_vel)
        print(f'rotated desired velocity: {rotated_desired_base_vel}')      

        # Update desired_base_vel with rotated values
        self.desired_base_vel[0] = rotated_desired_base_vel[0]
        self.desired_base_vel[1] = rotated_desired_base_vel[1]

        self.desired_base_vel = self.desired_base_vel / np.linalg.norm(self.desired_base_vel)
        
        range = np.array(self.ranges, dtype=np.float32)
        range = np.roll(range, int(len(self.ranges)/2))
        range[range > self.max_range] = self.max_range
        min_step = int(range.shape[0] / self.lidar_samples)
        obs_range = range.reshape((self.lidar_samples, min_step)).min(axis=1)
        
        if np.any(obs_range < self.collision_ranges):
            self.base_control = False

        vel = np.array([self.base_linear_vel.x, self.base_linear_vel.y])
        ang_vel = np.array([self.base_angular_vel])

        des_velocity = self.desired_base_vel[:2]
        heading = np.arctan2(des_velocity[1], des_velocity[0])[..., np.newaxis]
        heading = np.where(heading > math.pi, heading - 2 * math.pi, heading)
        heading = np.where(heading < -math.pi, heading + 2 * math.pi, heading)

        obs_vel = np.clip(vel, -self.max_lin_vel, self.max_lin_vel) / self.max_lin_vel
        obs_ang_vel = np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel) / self.max_ang_vel

        #print('------------------------------------')
        #print(f'robot position: {x2}, {y2}')
        #print(f'target position: {x1}, {y1}')
        #print(f'distance: {self.goal_distances}')
        #print(f'angle: {heading}')
        #print(f'base yaw: {self.base_yaw}')
        #print(f'unrotated desired velocity: {unrotated_velocities}')
        #print(f'desired velocity: {des_velocity}')
        #print(f'observed velocity: {obs_vel}')

        # mirroring for driving backwards
        if des_velocity[0] < 0.0 and self.backwards:
            self.mirroring = -1.0
            obs_range = np.roll(obs_range, int(self.lidar_samples/2))
            if heading < 0.0:
                heading = heading + math.pi
                
            else:
                heading = heading - math.pi
        else:
            self.mirroring = 1.0
        des_velocity = des_velocity * self.mirroring
        obs_vel = obs_vel * self.mirroring
        self.action = self.action * self.mirroring
        self.old_action = self.old_action * self.mirroring

        observation = np.concatenate((obs_range/self.max_range, heading/math.pi, des_velocity, obs_vel, obs_ang_vel, self.action, self.old_action)).astype(np.float32)
        observation = observation.reshape((1,-1))
        observation = np.nan_to_num(observation)
        if self.lstm:
            outputs = self.ort_model.run(None, {"obs": observation, "out_state.1" : self.out_state, "hidden_state.1" : self.hidden_state})
            self.out_state = outputs[3]
            self.hidden_state = outputs[4]
        else:
            outputs = self.ort_model.run(None, {"obs": observation})
        mu = outputs[0].squeeze()
        action = np.clip(mu, -1.0, 1.0)
        base_action = action[:3]

        self.old_action = self.action
        self.action = base_action

        print(f'base action: {base_action}')

        base_action[:2] = base_action[:2] * self.max_lin_vel * self.mirroring
        base_action[2] = np.clip(base_action[2] * self.max_ang_vel, -self.max_allowed_ang_vel, self.max_allowed_ang_vel)
    
        # publish base actions as twist message, base_action[0] is the linear velocity, base_action[1] is the angular velocity
        twist = Twist()
        if np.all(obs_range > self.deactivation_range):
            self.manual = True
        elif np.any(obs_range <= self.activation_range):
            self.manual = False

        if self.activation_mode and self.manual:
            twist.linear.x = des_velocity[0]
            twist.linear.y = des_velocity[1]
            twist.angular.z = 0.0
        else:
            twist.linear.x = base_action[0]
            twist.linear.y = base_action[1]
            twist.angular.z = base_action[2]
        if self.base_control:
            self.base_cmd_vel_pub.publish(twist)
        else:
            twist.linear.x = 0.0
            twist.linear.y = 0.0 
            twist.angular.z = 0.0
            self.base_cmd_vel_pub.publish(twist)

        # publish heading
        heading_msg = Float64()
        heading_msg.data = heading
        self.heading_pub.publish(heading_msg)

        # publish desired velocity and joy for bag file plots
        joy_msg = Joy()
        joy_msg.axes = [des_velocity[1], des_velocity[0], 0.0]
        self.joy_pub.publish(joy_msg)
        self.lin_vel_command.vector.x = des_velocity[0]
        self.lin_vel_command.vector.y = des_velocity[1]
        self.lin_vel_command.vector.z = 0.0
        self.des_vel_pub.publish(self.lin_vel_command)



if __name__ == '__main__':
    x = float(input("Target x: "))
    y = float(input("Target y: "))
    rospy.init_node('rl_node', anonymous=True)
    KayaNode(target_pos=np.array([x, y, 0.0]))
    rospy.spin()