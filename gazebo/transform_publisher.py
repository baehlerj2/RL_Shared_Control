#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Vector3Stamped, TwistStamped
import tf2_ros
import tf2_geometry_msgs  # To convert Pose to TransformStamped
import pdb

'''
example of broadcasting transforms when 
'''

class mynode():
    def __init__(self):
        self.original_odometry = None
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.sub1 = rospy.Subscriber('/odom', Odometry, self.callback)
        self.sub2 = rospy.Subscriber('/desired_velocity', Vector3Stamped, self.callback2)
        self.new = TwistStamped()
        self.new.header.frame_id = "base_link"
        rospy.Timer(rospy.Duration(1.8), self.pub_wist)
    
    def callback(self, msg):
        
        # Create a TransformStamped message to be published
        transform = TransformStamped()
        self.original_odometry = msg
        
        transform.header = msg.header
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        # transform.header.stamp = rospy.Time.now()
        
        transform.transform.translation = msg.pose.pose.position
        transform.transform.rotation = msg.pose.pose.orientation

        # Send the transform
        self.broadcaster.sendTransform(transform)

    def callback2(self, msg):
        twist = TwistStamped()
        twist.header.frame_id = "base_link"
        twist.twist.linear = msg.vector
        self.new = twist


    def pub_wist(self, event):
        rospy.Publisher('/des_vel_twiststamped', TwistStamped, queue_size=10).publish(self.new)


if __name__ == "__main__":
    rospy.init_node('odom_to_tf_publisher')

    # Create a tf2 broadcaster

    # Subscribe to the odom topic
    
    mynode()
    rospy.spin()