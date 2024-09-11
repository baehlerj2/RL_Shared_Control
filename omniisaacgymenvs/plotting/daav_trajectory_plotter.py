import rosbag
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
import os

import numpy as np

import pdb
import csv
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

def plot_trajectory(trajectories, headings, des_velocities, ranges, targets, task, rectangle_width, rectangle_height, door_width, starts, labels, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 13), height_ratios=[3, 1])
    
    title_font_size = 20
    label_font_size = 20
    
    ax1.set_xlabel('X [m]', fontsize=label_font_size)
    ax1.set_ylabel('Y [m]', fontsize=label_font_size)
    ax1.set_title('Gazebo Trajectories', fontsize=title_font_size)
    ax1.tick_params(axis='both', which='major', labelsize=label_font_size)
    ax1.grid(True)
    
    ax2.set_xlabel('X [m]', fontsize=label_font_size)
    ax2.set_ylabel('Heading [°]', fontsize=label_font_size)
    ax2.set_title('Headings over Time', fontsize=title_font_size)
    ax2.tick_params(axis='both', which='major', labelsize=label_font_size)
    ax2.grid(True)
    
    # Set the axes range
    min = -10
    max= 10
    ax1.axis('square')
    ax1.set_xlim(min, max)
    ax1.set_ylim(min, max)
    ax2.set_xlim(min, max)
    ax1.set_xticks(np.arange(min, max+1, 1))
    ax2.set_xticks(np.arange(min, max+1, 1))
    ax1.set_yticks(np.arange(min, max+1, 1))
    
    # Draw a point
    point_x = 0.0  # Change the x-coordinate of the point here
    point_y = 2.5  # Change the y-coordinate of the point here
    # ax1.plot(point_x, point_y, 'go')
    # for start in starts:
    #     ax1.plot(start, -point_y, 'mo')
    
    # Add text to name the points
    # ax1.text(point_x, point_y, 'Target ', ha='left', va='top', fontsize=label_font_size)
    # for start in starts:
    #     ax1.text(start, -point_y, 'Start ', ha='right', va='bottom', fontsize=label_font_size)
    
    # Iterate over the list of trajectories
    for i, (x, y, t) in enumerate(trajectories):
        ax1.plot(x, y, label=labels[i])
        if len(targets[i][0]) > 0:
            ax1.plot(targets[i][0], targets[i][1], 'g--', label='Target')
        
        # Draw arrow from (x, y) to (point_x, point_y)
        interval = 20  # Change the interval value here
        for j in range(0, len(x), interval):
            if len(des_velocities) == 0:
                if len(targets[i][0]) > 0:
                    try:
                        arrow = np.array([targets[i][0][j] - x[j], targets[i][1][j] - y[j]])
                    except:
                        continue
                else:
                    arrow = np.array([point_x - x[j], point_y - y[j]])
            else:
                arrow = np.array([des_velocities[i][0][j], des_velocities[i][1][j]])
            arrow = arrow / np.linalg.norm(arrow) * 0.7
            # plt.arrow(x[j], y[j], arrow[0], arrow[1], color=plt.gca().lines[-1].get_color(), width=0.01, head_width=0.1, length_includes_head=True)
            arrow = ax1.arrow(x[j], y[j], arrow[0], arrow[1], color='black', width=0.01, head_width=0.1, length_includes_head=True)
                
            #        plt.annotate(f'{t[j]}s', color=plt.gca().lines[-1].get_color(), xy=(x[j], y[j]), xytext=(x[j]+0.3, y[j]+0.3),
            #                     arrowprops=dict(facecolor='black', arrowstyle='->', color=plt.gca().lines[-1].get_color()),
            #                     ha='left', va='bottom')
            #        plt.tight_layout()
        for j in range(0, len(x)):
            lidars = ranges[i][0][j]
            for l in lidars:
                if l <= 12.5:
                    # pdb.set_trace()
                    angle = -np.pi + np.pi/180 * lidars.index(l) - yaw[j]
                    ax1.plot(x[j] + l*np.cos(angle), y[j] + l*np.sin(angle), 'ro', markersize=0.1)
    
    # if task == 'box':
    #     # Draw rectangle
    #     center_x = 0.0  # Change the x-coordinate of the center here
    #     center_y = 0.0  # Change the y-coordinate of the center here
    #     rect = patches.Rectangle((center_x - rectangle_width/2, center_y - rectangle_height/2), rectangle_width, rectangle_height, facecolor='red')
    #     ax1.add_patch(rect)

    # if task == 'door':
    #     # Draw a door
    #     total_width = 10.0  # Change the total width of the wall here
    #     width = (total_width - door_width) / 2
    #     height = 0.4
    #     rect1 = patches.Rectangle((door_width/2, -height/2), width, height, facecolor='red')
    #     rect2 = patches.Rectangle((-door_width/2-width, -height/2), width, height, facecolor='red')
    #     ax1.add_patch(rect1)
    #     ax1.add_patch(rect2)
    
    # Iterate over the list of headings
    for i, (h, t) in enumerate(headings):
        ax2.plot(trajectories[i][0], h, label=labels[i])
    
    # Add legend for trajectory labels
    # handles, labels = ax1.get_legend_handles_labels()
    # handles.append(arrow)
    # labels.append('User Input')
    # ax1.legend(handles=handles, labels=labels, handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)}, fontsize=label_font_size, loc='upper right')
    
    fig.savefig('plots/'+name+'_trajectory.png', format='png', bbox_inches='tight')
    # ax2.savefig('plots/'+name+'_headings.png', format='png', bbox_inches='tight')
    #plt.show()

# define bag names and csv names
#bag_names = ['bag_files/block/block_rds/block1_rds_0angle.bag', 'bag_files/block/block_rl/block1_rl_0angle.bag', 'bag_files/block/block_rl_lstm/block1_rl_lstm_0angle.bag', 'bag_files/block/block_rl_lstm_lcnn/block1_rl_lstm_lcnn_0angle.bag']
# bag_names2 = ['bag_files/block/block_rds/block2_rds_0angle.bag', 'bag_files/block/block_rl/block2_rl_0angle.bag', 'bag_files/block/block_rl_lstm/block2_rl_lstm_0angle.bag', 'bag_files/block/block_rl_lstm_lcnn/block2_rl_lstm_lcnn_0angle.bag']
# bag_names3 = ['bag_files/block/block_rds/block3_rds_0angle.bag', 'bag_files/block/block_rl/block3_rl_0angle.bag', 'bag_files/block/block_rl_lstm/block3_rl_lstm_0angle.bag', 'bag_files/block/block_rl_lstm_lcnn/block3_rl_lstm_lcnn_0angle.bag']

# bag_names = ['bag_files/door_rds/door1_25_rds_0angle.bag', 'bag_files/door_rl_lstm_lcnn/door1_25_rl_lstm_lcnn_0angle.bag']
# bag_names = ['bag_files/door_rds/door1_5_rds_0angle.bag', 'bag_files/door_rl_lstm_lcnn/door1_5_rl_lstm_lcnn_0angle.bag']
# bag_names = ['bag_files/door_rds/door1_5_rds_20angle.bag', 'bag_files/door_rl_lstm_lcnn/door1_5_rl_lstm_lcnn_20angle.bag']#, 'bag_files/door_rl_lstm_lcnn/door1_5_rl_lstm_lcnn_-20angle.bag']
#bag_names = ['bag_files/door/door_rl_lstm_lcnn/door1_5_rl_lstm_lcnn_20angle.bag', 'bag_files/door/door_rl_lstm_lcnn/door1_5_rl_lstm_lcnn_-20angle.bag']
# bag_names = ['bag_files/door_rl_lstm_lcnn/door1_5_rl_lstm_lcnn_movetarget.bag']
# bag_names = ['bag_files/block_rl_lstm_lcnn/block3_rl_lstm_lcnn_movetarget.bag']


folder = 'daav_bags/door/reward2_side_door1/'
task = 'door'
rectangle_width = 4.0  # Change the width of the rectangle here
rectangle_height = 1.0  # Change the height of the rectangle here
door_width = 1.5  # Change the width of the door here
start = 0.0

bag_names = []
for file in os.listdir(folder):
    if file.endswith('.bag'):
        bag_names.append(folder+file)

bag_names = ['daav_bags/door/video_reward2.bag']

bag_names.sort()
plot_names = []
for bag_name in bag_names:
    plot_names.append(bag_name.split('/')[-1].split('.')[0])

starts = [start for _ in range(len(bag_names))]
plot_names = [f'bag{i}' for i in range(len(bag_names))]

csv_names = [
    'mycsv.csv',
    ]

csv_names = []

assert len(bag_names) == len(starts)

trajectory_name = 'test'
heading_name = trajectory_name + '_heading'

trajectories = []
des_velocities = []
headings = []
targets = []
lidar_ranges = []
for i in range(len(bag_names)):
    bag_name = bag_names[i]
    start = starts[i]
    bag = rosbag.Bag(bag_name)  # Replace with the path to your rosbag file
    x = []
    y = []
    x_vel = []
    y_vel = []
    yaw = []
    time = []
    old_time = 0
    old_x = 0
    for topic, msg, t in bag.read_messages(topics=['/odometry/filtered', '/odom']):
    #for topic, msg, t in bag.read_messages(topics=['/ridgeback_velocity_controller/odom']):
    # for topic, msg, t in bag.read_messages(topics=['/odom']):
        # Extract the data you need from the rosbag message
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if timestamp != old_time: # and abs(msg.pose.pose.position.x) > 1e-5 and abs(msg.pose.pose.position.y - old_x) > 0.001:
            y.append(msg.pose.pose.position.y)
            x.append(msg.pose.pose.position.x)
            x_vel.append(msg.twist.twist.linear.x)
            y_vel.append(msg.twist.twist.linear.y)
            yaw.append(msg.pose.pose.orientation.z)
            time.append(timestamp)
            old_time = timestamp
            old_x = msg.pose.pose.position.y
    # x.append(0.0)
    # y.append(0.0)
    # for i in range(1, len(x_vel)):
    #     angle = yaw[i]
    #     d_t = time[i] - time[i-1]
    #     x.append(x[i-1] + d_t * (x_vel[i-1] * np.cos(angle) - y_vel[i-1] * np.sin(angle)))
    #     y.append(y[i-1] + d_t * (x_vel[i-1] * np.sin(angle) + y_vel[i-1] * np.cos(angle)))
    des_x = []
    des_y = []
    des_t = []
    old_time = 0
    for topic, msg, t in bag.read_messages(topics=['/desired_velocity']):
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if timestamp != old_time and (abs(msg.vector.x) > 0 or abs(msg.vector.y) > 0):
            des_y.append(msg.vector.y)
            des_x.append(msg.vector.x)
            des_t.append(timestamp)
            old_time = timestamp

    ranges = []
    ranges_time = []
    old_time = 0
    for topic, msg, t in bag.read_messages(topics=['/lidar/input/scan']):
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if timestamp != old_time:
            ranges.append(msg.ranges)
            ranges_time.append(timestamp)
            old_time = timestamp

    h = []
    h_time = []
    old_time = 0
    for topic, msg, t in bag.read_messages(topics=['/heading']):
        # Extract the data you need from the rosbag message
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if timestamp != old_time:
            heading = msg.data * 180 / 3.14159
            h.append(heading)
            h_time.append(timestamp)
            old_time = timestamp

    target_x = []
    target_y = []
    target_time = []
    old_time = 0
    for topic, msg, t in bag.read_messages(topics=['/target']):
        # Extract the data you need from the rosbag message
        timestamp = round(t.secs + t.nsecs * 1e-9, 1)
        if timestamp != old_time:
            target_y.append(msg.point.x)
            target_x.append(msg.point.y-start)
            target_time.append(timestamp)
            old_time = timestamp

    time_list = [time, h_time, des_t, ranges_time]
    all_times = set.intersection(*[set(times) for times in time_list])
    x = [x[i] for i in range(len(time)) if time[i] in all_times]
    y = [y[i] for i in range(len(time)) if time[i] in all_times]
    yaw = [yaw[i] for i in range(len(time)) if time[i] in all_times]
    time = [time[i] for i in range(len(time)) if time[i] in all_times]

    h = [h[i] for i in range(len(h_time)) if h_time[i] in all_times]
    h_time = [h_time[i] for i in range(len(h_time)) if h_time[i] in all_times]

    des_x = [des_x[i] for i in range(len(des_t)) if des_t[i] in all_times]
    des_y = [des_y[i] for i in range(len(des_t)) if des_t[i] in all_times]
    des_t = [des_t[i] for i in range(len(des_t)) if des_t[i] in all_times]

    ranges = [ranges[i] for i in range(len(ranges_time)) if ranges_time[i] in all_times]
    ranges_time = [ranges_time[i] for i in range(len(ranges_time)) if ranges_time[i] in all_times]

    time = [t - time[0] for t in time]
    # x = [-i for i in x]
    # x = [x[i]-x[0]-start for i in range(len(x))]
    # y = [y[i]-y[0] for i in range(len(y))]
    trajectories.append((x, y, time))

    h_time = [t - h_time[0] for t in h_time]
    headings.append((h, h_time))

    target_time = [t - target_time[0] for t in target_time]
    target_x = [-i for i in target_x]
    targets.append((target_x, target_y, target_time))
    des_t = [t - des_t[0] for t in des_t]
    for i in range(len(des_x)):
        vel = np.array([des_x[i], des_y[i]])
        angle = yaw[i]
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])
        vel = np.dot(rotation_matrix, vel)
        des_x[i] = vel[0]
        des_y[i] = vel[1]
    des_velocities.append((des_x, des_y, des_t))

    ranges_time = [t - ranges_time[0] for t in ranges_time]
    lidar_ranges.append((ranges, ranges_time))

    bag.close()

# read csv files
for csv_file in csv_names:
    x = []
    y = []
    time = []
    old_time = -1
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if old_time != float(row[2]) and int(float(row[3])) == 0:
                y.append(float(row[1]))
                x.append(-float(row[0]))
                time.append(float(row[2]))
                old_time = float(row[2])


    time = [t - time[0] for t in time]
    x = [-i for i in x]
    trajectories.append((x, y, time))

starts = set(starts)
starts = list(starts)
plot_trajectory(trajectories, headings, des_velocities, lidar_ranges, targets, task, rectangle_width, rectangle_height, door_width, starts, labels=plot_names, name=trajectory_name)
# plot_headings(headings, labels=plot_names, name=heading_name)