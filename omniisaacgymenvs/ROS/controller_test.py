import rospy
from geometry_msgs.msg import Twist, TransformStamped, Vector3Stamped
from sensor_msgs.msg import Joy, LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import pdb
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_geometry_msgs
import math

'''
A simple node to manually drive around a robot in Gazebo using a joystick.
'''

class ControllerNode:
    def __init__(self) -> None:
        self.sub = rospy.Subscriber('/bluetooth_teleop/joy', Joy, self.joy_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=20)
        #self.lidar_sub = rospy.Subscriber('/front/scan', LaserScan, self.lidar_scan)
        self.pose_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.position_callback)
        self.vel_sub = rospy.Subscriber('/ridgeback_velocity_controller/odom', Odometry, self.velocity_callback)


        self.lin_vel_command = Vector3Stamped()
        self.x_vel_command = 0.0
        self.y_vel_command = 0.0
        self.z_vel_command = 0.0

        self.target_pos = np.array([0.0, 0.0, 0.0])

        self.count = 0

        self.desired_base_vel = np.array([0.0, 0.0])

        rospy.Timer(rospy.Duration(1/5.0), self.send_control)

    def joy_callback(self, msg):
        self.lin_vel_command.vector.x = msg.axes[1]
        self.lin_vel_command.vector.y = msg.axes[0]
        self.lin_vel_command.vector.z = 0.0
        self.x_vel_command = msg.axes[1]
        self.y_vel_command = msg.axes[0]
        if abs(msg.axes[2]) > 0.2:
            self.z_vel_command = msg.axes[2] * math.pi
        else:
            self.z_vel_command = 0.0

    def send_control(self, timer_event):
        twist = Twist()
        twist.linear.x = self.x_vel_command
        twist.linear.y = self.y_vel_command
        twist.angular.z = self.z_vel_command

        self.pub.publish(twist)
        print(f'base position: {self.base_position}')

    def lidar_scan(self, msg1):
        self.ranges = np.array(msg1.ranges)
        self.ranges = np.roll(self.ranges, int(len(msg1.ranges)/2))
        if self.count % 1000 == 0:
            print(self.ranges)
        self.count += 1

    def position_callback(self, msg):
        self.base_position = msg.pose.pose.position
        self.base_orientation = msg.pose.pose.orientation
        orientation_list = [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        self.base_yaw = yaw

        t = TransformStamped()
        t.transform.rotation = self.base_orientation
        vel_in_w = tf2_geometry_msgs.do_transform_vector3(self.lin_vel_command, t)
        
        x1, y1, _ = self.target_pos
        x2, y2 = self.base_position.x, self.base_position.y

        self.desired_base_vel[0] = x1 - x2
        self.desired_base_vel[1] = y1 - y2

        # Convert desired_base_vel to the rotated frame
        rotated_desired_base_vel = np.array([self.desired_base_vel[0], self.desired_base_vel[1]])
        rotation_angle = self.base_yaw
        rotation_matrix = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle)],
                                    [math.sin(rotation_angle), math.cos(rotation_angle)]])
        rotated_desired_base_vel = np.dot(rotation_matrix, rotated_desired_base_vel)

        # Update desired_base_vel with rotated values
        self.desired_base_vel[0] = rotated_desired_base_vel[0]
        self.desired_base_vel[1] = rotated_desired_base_vel[1]

    def velocity_callback(self, msg):
        self.base_linear_vel = msg.twist.twist.linear
        self.base_angular_vel = msg.twist.twist.angular.z


if __name__ == '__main__':
    rospy.init_node('my_node', anonymous=True)
    ControllerNode()
    rospy.spin()