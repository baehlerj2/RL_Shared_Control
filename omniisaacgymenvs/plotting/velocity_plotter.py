import rosbag

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pdb
import csv

import math

x_vel = []
y_vel = []
ang_vel = []
time = []

ac1 = []
ac2 = []
ac3 = []

# Example usage
csv_file = 'mycsv.csv'

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x_vel.append(float(row[4]))
        y_vel.append(float(row[5]))
        ang_vel.append(float(row[6]))
        ac1.append(float(row[7]))
        ac2.append(float(row[8]))
        ac3.append(float(row[9]))
        time.append(float(row[2]))
        old_time = float(row[2])

bag_name = 'test3.bag'


bag = rosbag.Bag('bag_files/' + bag_name)  # Replace with the path to your rosbag file

bag_x_vel = []
bag_y_vel = []
bag_ang_vel = []
bag_vel_time = []

bag_ac1 = []
bag_ac2 = []
bag_ac3 = []
bag_ac_time = []

for topic, msg, t in bag.read_messages(topics=['/odometry/filtered', '/cmd_vel']):
    # Extract the data you need from the rosbag message
    timestamp = round(t.secs + t.nsecs * 1e-9, 1)
    if topic == '/odometry/filtered':
        bag_y_vel.append(msg.twist.twist.linear.x)
        bag_x_vel.append(msg.twist.twist.linear.y)
        bag_ang_vel.append(msg.twist.twist.angular.z)
        bag_vel_time.append(timestamp)

    elif topic == '/cmd_vel':
        bag_ac1.append(msg.linear.x)
        bag_ac2.append(msg.linear.y)
        bag_ac3.append(msg.angular.z)
        bag_ac_time.append(timestamp)

bag.close()

# Create a figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 10))

# Plot for csv data
axs[0, 0].plot(time, x_vel, label='x_vel')
axs[0, 0].plot(time, y_vel, label='y_vel')
axs[0, 0].plot(time, ac1, label='x action')
axs[0, 0].plot(time, ac2, label='y action')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Value')
axs[0, 0].set_title('Kaya - Isaac - Linear Velocity Data')
axs[0, 0].legend()

# Plot for bag data
axs[0, 1].plot(bag_vel_time, bag_x_vel, label='x_vel')
axs[0, 1].plot(bag_vel_time, bag_y_vel, label='y_vel')
axs[0, 1].plot(bag_ac_time, bag_ac1, label='x action')
axs[0, 1].plot(bag_ac_time, bag_ac2, label='y action')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Value')
axs[0, 1].set_title('Ridgeback - Gazebo - Linear Velocity Data')
axs[0, 1].legend()

# Plot for csv data
axs[1, 0].plot(time, ang_vel, label='ang_vel')
axs[1, 0].plot(time, ac3, label='action')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Value')
axs[1, 0].set_title('Kaya - Isaac - Angular Velocity Data')
axs[1, 0].legend()

# Plot for bag data
axs[1, 1].plot(bag_vel_time, bag_ang_vel, label='ang_vel')
axs[1, 1].plot(bag_ac_time, bag_ac3, label='action')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Value')
axs[1, 1].set_title('Ridgeback - Gazebo - Angular Velocity Data')
axs[1, 1].legend()

# Adjust the layout of subplots
plt.tight_layout()
plt.show()

# Save the figure as a PNG file
#plt.savefig('velocity_plot.png')
# calculate difference between 