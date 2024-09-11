import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist, TransformStamped, Vector3Stamped, TwistStamped
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import onnxruntime as ort
import numpy as np
from sensor_msgs.msg import LaserScan, Joy
import math
from numpy import inf
import tf2_geometry_msgs
from omniisaacgymenvs.tasks.shared.capsule import Capsule
import tf
import pdb

'''
This script can be used to move the robot with a joystick while the rl-agent is running.
'''

class KayaNode:
    def __init__(self):

        # define subscribers and publishers
        # daav 
        # self.pose_sub = rospy.Subscriber('/odom', Odometry, self.position_callback)
        # self.lidar_sub = rospy.Subscriber('/lidar/input/scan', LaserScan, self.lidar_scan)
        # self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)

        # ridgeback gazebo
        self.pose_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.position_callback)
        self.lidar_sub = rospy.Subscriber('/front/scan', LaserScan, self.lidar_scan)
        self.joy_sub = rospy.Subscriber('/bluetooth_teleop/joy', Joy, self.joy_callback)

        self.base_cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=20)
        self.heading_pub = rospy.Publisher("/heading", Float64, queue_size=20)
        self.des_vel_pub = rospy.Publisher("/desired_velocity", Vector3Stamped, queue_size=20)
        self.base_vel_pub = rospy.Publisher("/base_velocity_anim", TwistStamped, queue_size=20)
        self.des_vel_anim_pub = rospy.Publisher("/desired_velocity_anim", TwistStamped, queue_size=20)

        # define model path
        self.model_path = "onnx_models/paper/KayaVel_All_small_LCNN_LSTM_onlyxvelerror_r_v_y_0_16_h_w_0_5.onnx"
        # self.model_path = 'onnx_models/paper/KayaVel_All_small_LCNN_LSTM_onlyxvelerror_r_v_y_0_16_h_w_0_5.onnx'
        self.lstm = True
        self.lstm_hidden_size = 64
        self.ort_model = ort.InferenceSession(self.model_path)
        self.base_control = True

        self.lin_vel_command = Vector3Stamped()

        # define lidar settings
        self.lidar_samples = 360 # set to 360 for models with LCNN and 36 without
        self.ranges = None 
        self.max_range = 2.5
        self.min_range = 0.3

        # define velocity limits
        # self.max_lin_vel =  0.3 #2/3 #1.0
        self.max_x_vel = 2/3 # 0.4
        self.max_y_vel = 2/3 # 0.2
        self.max_ang_vel = 2 #1.7 # this is for scaling the ouput
        self.max_allowed_ang_vel = self.max_ang_vel # 0.5 # this is for limiting the actual angular velocity

        self.base_position = None
        self.base_yaw = None
        self.base_linear_vel = None
        self.base_angular_vel = None

        self.action = np.zeros(3)
        self.old_action = np.zeros(3)

        self.manual = False

        self.backwards = False
        self.mirroring = 1.0   

        self.activation_mode = False
        # self.manual = True
        self.activation_range = 1.0
        self.deactivation_range = 1.3

        self.out_state = np.zeros((1, 1, self.lstm_hidden_size)).astype(np.float32)
        self.hidden_state = np.zeros((1, 1, self.lstm_hidden_size)).astype(np.float32)  

        rospy.Timer(rospy.Duration(1/40.0), self.send_control)

        self.collision_capsule = Capsule(
            0.65, 
            0.3, 
            int(360/self.lidar_samples), 
            0.2,
            )
        self.collision_ranges = np.array((self.collision_capsule.ranges)) # collision termination
    def joy_callback(self, msg):
        # davv
        # self.lin_vel_command.vector.x = msg.axes[0] * -1
        # self.lin_vel_command.vector.y = msg.axes[1]
        # self.lin_vel_command.vector.z = msg.axes[2]

        # ridgeback gazebo
        self.lin_vel_command.vector.x = msg.axes[1]
        self.lin_vel_command.vector.y = msg.axes[0]
        self.lin_vel_command.vector.z = msg.axes[2]
        self.manual = bool(msg.buttons[0])
        
    def position_callback(self, msg):
        self.base_position = msg.pose.pose.position
        self.base_orientation = msg.pose.pose.orientation
        orientation_list = [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.base_yaw = yaw
        self.base_linear_vel = msg.twist.twist.linear
        self.base_angular_vel = msg.twist.twist.angular.z

    def lidar_scan(self, msg1):
        self.ranges = msg1.ranges[:]

    def send_control(self, timer_event):
        if self.ranges is None or self.base_angular_vel is None or self.base_position is None:
            return
        
        range = np.array(self.ranges, dtype=np.float32)
        range = np.roll(range, int(len(self.ranges)/2))
        range[range > self.max_range] = self.max_range
        range[range < self.min_range] = self.min_range
        min_step = int(range.shape[0] / self.lidar_samples)
        obs_range = range.reshape((self.lidar_samples, min_step)).min(axis=1)

        if np.any(obs_range < self.collision_ranges):
           print("Collision detected")
           self.base_control = False

        vel = np.array([self.base_linear_vel.x, self.base_linear_vel.y])
        ang_vel = np.array([self.base_angular_vel])

        des_velocity = np.array([self.lin_vel_command.vector.x, self.lin_vel_command.vector.y])
        self.des_vel_pub.publish(self.lin_vel_command)

        heading = np.arctan2(des_velocity[1], des_velocity[0])[..., np.newaxis]
        heading = np.where(heading > math.pi, heading - 2 * math.pi, heading)
        heading = np.where(heading < -math.pi, heading + 2 * math.pi, heading)

        #obs_vel = np.clip(vel, -self.max_lin_vel, self.max_lin_vel) / self.max_lin_vel
        obs_vel = np.zeros(2)
        obs_vel[0] = np.clip(vel[0], -self.max_x_vel, self.max_x_vel) / self.max_x_vel
        obs_vel[1] = np.clip(vel[1], -self.max_y_vel, self.max_y_vel) / self.max_y_vel
        obs_ang_vel = np.clip(ang_vel, -1.0, 1.0) / 1.0

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

        #base_action[:2] = base_action[:2] * self.max_lin_vel * self.mirroring
        base_action[0] = base_action[0] * self.max_x_vel
        base_action[1] = base_action[1] * self.max_y_vel
        base_action[2] = np.clip(base_action[2] * self.max_ang_vel, -self.max_allowed_ang_vel, self.max_allowed_ang_vel)

        if self.lin_vel_command.vector.x == 0.0 and self.lin_vel_command.vector.y == 0.0:
            base_action[0] = 0.0
            base_action[1] = 0.0
            base_action[2] = 0.0
    
        # publish base actions as twist message, base_action[0] is the linear velocity, base_action[1] is the angular velocity
        twist = Twist()
        
        if self.activation_mode:
            if np.all(obs_range > self.deactivation_range):
                self.manual = True
            elif np.any(obs_range <= self.activation_range):
                self.manual = False
        
        if not self.manual:
            twist.linear.x = base_action[0]
            twist.linear.y = base_action[1]
            twist.angular.z = base_action[2]
        else:
            twist.linear.x = self.lin_vel_command.vector.x
            twist.linear.y = self.lin_vel_command.vector.y
            twist.angular.z = self.lin_vel_command.vector.z
        if self.base_control:
            self.base_cmd_vel_pub.publish(twist)
            a = 0
        else:
            twist.linear.x = 0
            twist.linear.y = 0
            twist.angular.z = 0
            self.base_cmd_vel_pub.publish(twist)

        # publish heading
        heading_msg = Float64()
        heading_msg.data = heading
        self.heading_pub.publish(heading_msg)

        # publish vellcities for animation
        base_vel_anim = TwistStamped()
        base_vel_anim.twist.linear.x = vel[0]
        base_vel_anim.twist.linear.y = vel[1]
        base_vel_anim.twist.angular.z = ang_vel[0]
        base_vel_anim.header.frame_id = "anim_link"
        self.base_vel_pub.publish(base_vel_anim)
        des_vel_anim = TwistStamped()
        des_vel_anim.twist.linear.x = des_velocity[0]
        des_vel_anim.twist.linear.y = des_velocity[1]
        des_vel_anim.header.frame_id = "anim_link"
        self.des_vel_anim_pub.publish(des_vel_anim)



if __name__ == '__main__':
    rospy.init_node('rl_node', anonymous=True)
    KayaNode()
    rospy.spin()