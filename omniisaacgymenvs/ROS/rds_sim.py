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

'''
This script drives the robot to a target position using a RDS and simulated user input.

Run the script and input the target position (from robot perspective) as requested.
'''

class KayaNode:
    def __init__(self, target_pos=np.array([5.0, 0.0, 0.0])):

        self.pose_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.position_callback)

        self.base_cmd_vel_pub = rospy.Publisher("/remote_cmd_vel", Twist, queue_size=20)
        self.desired_velocity_pub = rospy.Publisher("/desired_velocity", Twist, queue_size=20)
        self.target_pub = rospy.Publisher("/target", PointStamped, queue_size=20)
        self.heading_pub = rospy.Publisher("/heading", Float64, queue_size=20)

        self.lin_vel_command = Vector3Stamped()

        self.target_pos = target_pos
        self.desired_base_vel = np.array([0.0, 0.0, 0.0])
        self.target_margin = 0.25
        self.goal_distances = None
        self.goal_reached = False

        self.base_position = None
        self.base_yaw = None
        self.base_linear_vel = None
        self.base_angular_vel = None

        self.base_control = True

        rospy.Timer(rospy.Duration(1/120.0), self.send_control)
        rospy.Timer(rospy.Duration(1/120.0), self.update_base_pose)
        
    def position_callback(self, msg):
        self.base_position = msg.pose.pose.position
        self.base_orientation = msg.pose.pose.orientation
        orientation_list = [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.base_yaw = yaw
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
        #self.publish_target()

    def send_control(self, timer_event):
        if self.base_angular_vel is None or self.base_position is None:
            return
        x1, y1, _ = self.target_pos
        x2, y2 = self.base_position.x, self.base_position.y

        self.desired_base_vel[0] = x1 - x2
        self.desired_base_vel[1] = y1 - y2
        unrotated_velocities = self.desired_base_vel[:2] / np.linalg.norm(self.desired_base_vel[:2])

        # Convert desired_base_vel to the rotated frame
        rotated_desired_base_vel = np.array([self.desired_base_vel[0], self.desired_base_vel[1]])
        rotation_matrix = np.array([[math.cos(self.base_yaw), math.sin(self.base_yaw)],
                        [-math.sin(self.base_yaw), math.cos(self.base_yaw)]])
        rotated_desired_base_vel = np.dot(rotation_matrix, rotated_desired_base_vel)

        # Update desired_base_vel with rotated values
        self.desired_base_vel[0] = rotated_desired_base_vel[0]
        self.desired_base_vel[1] = rotated_desired_base_vel[1]

        self.desired_base_vel = self.desired_base_vel / np.linalg.norm(self.desired_base_vel)

        # publish base actions as twist message, base_action[0] is the linear velocity, base_action[1] is the angular velocity
        twist = Twist()

        twist.linear.x = self.desired_base_vel[0]
        twist.linear.y = self.desired_base_vel[1]
        twist.angular.z = 0.0

        if self.base_control:
            self.base_cmd_vel_pub.publish(twist)
        else:
            twist.linear.x = 0.0
            twist.linear.y = 0.0 
            twist.angular.z = 0.0
            self.base_cmd_vel_pub.publish(twist)

        # publish heading
        heading_msg = Float64()
        des_velocity = self.desired_base_vel[:2]
        heading = np.arctan2(des_velocity[1], des_velocity[0])[..., np.newaxis]
        heading = np.where(heading > math.pi, heading - 2 * math.pi, heading)
        heading = np.where(heading < -math.pi, heading + 2 * math.pi, heading)
        heading_msg.data = heading
        self.heading_pub.publish(heading_msg)
        self.desired_velocity_pub.publish(twist)



if __name__ == '__main__':
    x = float(input("Target x: "))
    y = float(input("Target y: "))
    rospy.init_node('rl_node', anonymous=True)
    KayaNode(target_pos=np.array([x, y, 0.0]))
    rospy.spin()