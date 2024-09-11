import rosbag
import pdb
import numpy as np

bag_file = 'daav_bags/box/videos/2024-08-08-16-22-14.bag'

# daav topics
joy_topic = '/joy'
cmd_vel_topic = '/cmd_vel'
lidar_topic = '/lidar/input/scan'
position_topic =  '/odom' #/Odometry'
heading_topic = '/heading'
velocity_topic = '/odom'
desired_velocity_topic = '/desired_velocity'
x_vel_scale = 0.4
y_vel_scale = 0.66
ang_vel_scale = 2.0

# # gazebo topics
# joy_topic = '/bluetooth_teleop/joy'
# cmd_vel_topic = '/cmd_vel'
# lidar_topic = '/front/scan'
# position_topic = '/odometry/filtered'
# heading_topic = '/heading'
# velocity_topic = '/odometry/filtered'
# desired_velocity_topic = '/desired_velocity'
# x_vel_scale = 2/3
# y_vel_scale =2/3
# ang_vel_scale = 2


# Open the bag file
bag = rosbag.Bag(bag_file)
user_input = []
cmd_vel = []
network_output = []
lidar_ranges = []
position = []
heading = []
velocity = []
desired_velocity = []

# Iterate over the messages in the bag file
for topic, msg, t in bag.read_messages():
    if topic == joy_topic:
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if len(user_input) == 0 or user_input[-1][2] != timestamp:
            if joy_topic == '/joy':
                x = msg.axes[0]
                y = msg.axes[1]
                z = msg.axes[2]
                user_input.append([x, y, timestamp, z])
            elif joy_topic == '/bluetooth_teleop/joy':
                x = msg.axes[0]
                y = msg.axes[1]
                z = msg.axes[2]
                user_input.append([x, y, timestamp, z])

    if topic == desired_velocity_topic:
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if len(desired_velocity) == 0 or desired_velocity[-1][2] != timestamp:
            x = msg.vector.x
            y = msg.vector.y
            z = msg.vector.z
            desired_velocity.append([x, y, timestamp, z])
    
    if topic == cmd_vel_topic:
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if len(cmd_vel) == 0 or cmd_vel[-1][3] != timestamp:
            x = msg.linear.x
            y = msg.linear.y
            z = msg.angular.z
            cmd_vel.append([x, y, z, timestamp])
            x_out = x / x_vel_scale * 2/3
            y_out = y / y_vel_scale * 2/3
            z_out = z / ang_vel_scale
            network_output.append([x_out, y_out, z_out, timestamp])
    
    if topic == lidar_topic:
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if len(lidar_ranges) == 0 or lidar_ranges[-1][1] != timestamp:
            lidar_ranges.append([msg.ranges, timestamp])

    if topic == position_topic:
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if len(position) == 0 or position[-1][2] != timestamp:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            position.append([x, y, timestamp])

    if topic == heading_topic:
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if len(heading) == 0 or heading[-1][1] != timestamp:
            h = msg.data
            heading.append([h, timestamp])

    if topic == velocity_topic:
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if len(velocity) == 0 or velocity[-1][3] != timestamp:
            x = msg.twist.twist.linear.x
            y = msg.twist.twist.linear.y
            z = msg.twist.twist.angular.z
            velocity.append([x, y, z, timestamp])

# Close the bag file
bag.close()

user_input_msgs = len(user_input)
cmd_vel_msgs = len(cmd_vel)
network_output_msgs = len(network_output)
lidar_msgs = len(lidar_ranges)
position_msgs = len(position)
heading_msgs = len(heading)
velocity_msgs = len(velocity)

print("User input: ", len(user_input))
print("Cmd vel: ", len(cmd_vel))
print("Network output: ", len(network_output))
print("Lidar ranges: ", len(lidar_ranges))
print("Position: ", len(position))
print("Heading: ", len(heading))
print("Velocity: ", len(velocity))


#assert cmd_vel_msgs == network_output_msgs == heading_msgs

# Convert lists to numpy arrays
user_input = np.array(user_input)
desired_velocity = np.array(desired_velocity)
cmd_vel = np.array(cmd_vel)
network_output = np.array(network_output)
lidar_ranges = np.array(lidar_ranges)
position = np.array(position)
heading = np.array(heading)
velocity = np.array(velocity)

min_time = max(user_input[0, 2], cmd_vel[0, 3], network_output[0, 3], lidar_ranges[0, 1], position[0, 2], heading[0, 1], velocity[0, 3])
max_time = min(user_input[-1, 2], cmd_vel[-1, 3], network_output[-1, 3], lidar_ranges[-1, 1], position[-1, 2], heading[-1, 1], velocity[-1, 3])

user_input = user_input[np.where(user_input[:, 2] >= min_time)]
user_input = user_input[np.where(user_input[:, 2] <= max_time)]
lidar_ranges = lidar_ranges[np.where(lidar_ranges[:, 1] >= min_time)]
lidar_ranges = lidar_ranges[np.where(lidar_ranges[:, 1] <= max_time)]
position = position[np.where(position[:, 2] >= min_time)]
position = position[np.where(position[:, 2] <= max_time)]
velocity = velocity[np.where(velocity[:, 3] >= min_time)]
velocity = velocity[np.where(velocity[:, 3] <= max_time)]
cmd_vel = cmd_vel[np.where(cmd_vel[:, 3] >= min_time)]
cmd_vel = cmd_vel[np.where(cmd_vel[:, 3] <= max_time)]
network_output = network_output[np.where(network_output[:, 3] >= min_time)]
network_output = network_output[np.where(network_output[:, 3] <= max_time)]
heading = heading[np.where(heading[:, 1] >= min_time)]
heading = heading[np.where(heading[:, 1] <= max_time)]
if desired_velocity.shape[0] > 0:
    desired_velocity = desired_velocity[np.where(desired_velocity[:, 2] >= min_time)]
    desired_velocity = desired_velocity[np.where(desired_velocity[:, 2] <= max_time)]



# Normalize time for every array
user_input[:, 2] -= min_time
cmd_vel[:, 3] -= min_time
network_output[:, 3] -= min_time
lidar_ranges[:, 1] -= min_time
position[:, 2] -= min_time
heading[:, 1] -= min_time
velocity[:, 3] -= min_time

if desired_velocity.shape[0] > 0:
    desired_velocity[:, 2] -= min_time
    #assert desired_velocity.shape[0] == cmd_vel.shape[0]

#assert user_input.shape[0] == cmd_vel.shape[0] == network_output.shape[0] == lidar_ranges.shape[0] == position.shape[0] == heading.shape[0] == velocity.shape[0]

import matplotlib.pyplot as plt

print('min ax0: ', np.min(user_input[:, 0]))
print('max ax0: ', np.max(user_input[:, 0]))
print('min ax1: ', np.min(user_input[:, 1]))
print('max ax1: ', np.max(user_input[:, 1]))
print('min ax2: ', np.min(user_input[:, 3]))
print('max ax2: ', np.max(user_input[:, 3]))

# Create a figure with 5 subplots
# plt.plot(user_input[:, 2], user_input[:, 0], label='axis 0')
# plt.plot(user_input[:, 2], user_input[:, 1], label='axis 1')
# plt.plot(user_input[:, 2], user_input[:, 3], label='axis 2')
# plt.legend()
# plt.show()

fig, axs = plt.subplots(5, 1, figsize=(10, 15))

# Plot 1: x velocity
axs[0].plot(network_output[:, 3], network_output[:, 0], label='network output')
axs[0].plot(cmd_vel[:, 3], cmd_vel[:, 0], label='cmd vel')
axs[0].plot(velocity[:, 3], velocity[:, 0], label='robot velocity')
axs[0].plot(user_input[:, 2], user_input[:, 0], label='joy')
if desired_velocity.shape[0] > 0:
    axs[0].plot(desired_velocity[:, 2], desired_velocity[:, 0], label='desired velocity')

axs[0].set_xlabel('Timestamp [s]')
axs[0].set_ylabel('Velocity [m/s]')
axs[0].set_title('X Velocity')
axs[0].legend()

# Plot 2: y velocity
axs[1].plot(network_output[:, 3], network_output[:, 1], label='network output')
axs[1].plot(cmd_vel[:, 3], cmd_vel[:, 1], label='cmd vel')
axs[1].plot(velocity[:, 3], velocity[:, 1], label='robot velocity')
axs[1].plot(user_input[:, 2], user_input[:, 1], label='joy')
if desired_velocity.shape[0] > 0:
    axs[1].plot(desired_velocity[:, 2], desired_velocity[:, 1], label='desired velocity')

axs[1].set_xlabel('Timestamp [s]')
axs[1].set_ylabel('Velocity [m/s]')
axs[1].set_title('Y Velocity')
axs[1].legend()

# Plot 3: angular velocity
axs[2].plot(network_output[:, 3], network_output[:, 2], label='network output')
axs[2].plot(cmd_vel[:, 3], cmd_vel[:, 2], label='cmd vel')
axs[2].plot(velocity[:, 3], velocity[:, 2], label='robot velocity')

axs[2].set_xlabel('Timestamp [s]')
axs[2].set_ylabel('Velocity [rad/s]')
axs[2].set_title('Angular Velocity')
axs[2].legend()

# Plot 4: Trajectory
axs[3].plot(position[:, 1], position[:, 0])
axs[3].set_xlabel('X [m]')
axs[3].set_ylabel('Y [m]')
axs[3].set_title('Trajectory')

# Plot 5: Heading
axs[4].plot(heading[:, 1], heading[:, 0])
axs[4].set_xlabel('Timestamp [s]')
axs[4].set_ylabel('Heading [rad]')
axs[4].set_title('Heading')




# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

