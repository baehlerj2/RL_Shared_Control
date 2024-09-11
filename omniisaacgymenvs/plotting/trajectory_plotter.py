import rosbag
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Bbox
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import numpy as np
import os

import pdb
import csv
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


def full_extent(ax, pad=0.01):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title, ax.yaxis.label, ax.xaxis.label]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)
def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

LABELS = ['FC', 'LFC', 'CLFC', 'CLFC_D', 'CLFC_D_R1', 'CLFC_D_R2', 'RDS']
TITLE = 'Trajectory'
COLORS = ['b', 'tab:purple', 'r', 'g', 'tab:orange', 'tab:brown', 'tab:pink']

def plot_trajectory(trajectories, headings, des_velocities, targets, task, rectangle_width, rectangle_height, door_width, starts, labels, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 11), height_ratios=[3, 1])
    
    title_font_size = 20
    label_font_size = 20
    
    # ax1.set_xlabel('X [m]', fontsize=label_font_size)
    ax1.set_ylabel('Y [m]', fontsize=label_font_size)
    ax1.set_xlabel('X [m]', fontsize=label_font_size)
    ax1.set_title(TITLE, fontsize=title_font_size)
    ax1.tick_params(axis='both', which='major', labelsize=label_font_size)
    ax1.grid(True)
    
    ax2.set_xlabel('X [m]', fontsize=label_font_size)
    ax2.set_ylabel('Heading [°]', fontsize=label_font_size)
    ax2.tick_params(axis='both', which='major', labelsize=label_font_size)
    ax2.grid(True)
    
    # Set the axes range
    if task == 'box' or task == 'door':
        r = 8
        ax1.axis('square')
        ax1.set_xlim(-1, r-1)
        ax1.set_ylim(-r/2, r/2)
        ax2.set_xlim(-1, r-1)
        ax1.set_xticks(np.arange(-1, r, 1))
        ax2.set_xticks(np.arange(-1, r, 1))
        ax1.set_yticks(np.arange(-r/2, r/2+1, 1))
        

    if task == 'rooms':
        ax1.set_xlim(-2, 13)
        ax1.set_ylim(-11.5, 3.5)
        LABELS = [None for i in range(len(trajectories))]

    
    # Draw a point
    if task == 'box' or task == 'door':
        point_x = 5.0  # Change the x-coordinate of the point here
        point_y = 0.0  # Change the y-coordinate of the point here
        ms = 15
        offset = 0.2
        ax1.plot(point_x, point_y, 'go', markersize=ms)
        for start in starts:
            ax1.plot(0.0, start, 'mo', markersize=ms)
        
        # Add text to name the points
        ax1.text(point_x+1.5*offset, point_y, 'Target ', ha='left', va='center', fontsize=label_font_size)
        old_start1 = []
        for start in starts:
            if start not in old_start1:
                ax1.text(0.0, start-offset, 'Start ', ha='right', va='top', fontsize=label_font_size)
                old_start1.append(start)
    
    # Iterate over the list of trajectories
    for i, (x, y, t) in enumerate(trajectories):
        if len(trajectories) == 6 or len(trajectories) == 7 or task == 'rooms':
            ax1.plot(x, y, label=LABELS[i], color=COLORS[i])
        else:  
            ax1.plot(x, y, label=LABELS[i+3], color=COLORS[i+3])
        unique_tx, unique_ty = [], []
        plot_target = False
        for j in range(len(targets[i][0])):
            if targets[i][0][j] not in unique_tx or targets[i][1][j] not in unique_ty:
                unique_tx.append(targets[i][0][j])
                unique_ty.append(targets[i][1][j])
            if len(unique_tx) > 1:
                plot_target = True
                break

        if plot_target:
            ax1.plot(targets[i][0], targets[i][1], 'g--', label='Target')
        
        # Draw arrow from (x, y) to (point_x, point_y)
        interval = 20  # Change the interval value here
        x_old, y_old = -10, -10
        for j in range(0, len(x), interval):
            dis = np.sqrt((x[j] - x_old)**2 + (y[j] - y_old)**2)
            if dis > 0.5:
                x_old, y_old = x[j], y[j]
                if len(des_velocities[i][0]) == 0:
                    if len(targets[i][0]) > 0:
                        try:
                            arrow = np.array([targets[i][0][j] - x[j], targets[i][1][j] - y[j]])
                        except:
                            continue
                    else:
                        arrow = np.array([point_x - x[j], point_y - y[j]])
                else:
                    arrow = np.array([des_velocities[i][0][j], des_velocities[i][1][j]])
                # arrow = np.array([point_x - x[j], point_y - y[j]])
                arrow = arrow / np.linalg.norm(arrow) * 0.5
                # plt.arrow(x[j], y[j], arrow[0], arrow[1], color=plt.gca().lines[-1].get_color(), width=0.01, head_width=0.1, length_includes_head=True)
                arrow = ax1.arrow(x[j], y[j], arrow[0], arrow[1], color='black', width=0.01, head_width=0.1, length_includes_head=True)
                
            #        plt.annotate(f'{t[j]}s', color=plt.gca().lines[-1].get_color(), xy=(x[j], y[j]), xytext=(x[j]+0.3, y[j]+0.3),
            #                     arrowprops=dict(facecolor='black', arrowstyle='->', color=plt.gca().lines[-1].get_color()),
            #                     ha='left', va='bottom')
            #        plt.tight_layout()
    
    if task == 'box':
        # Draw rectangle
        center_x = 2.5  # Change the x-coordinate of the center here
        center_y = 0.0  # Change the y-coordinate of the center here
        rect = patches.Rectangle((center_x - rectangle_width/2, center_y - rectangle_height/2), rectangle_width, rectangle_height, facecolor='red')
        ax1.add_patch(rect)

    if task == 'door':
        # Draw a door
        total_width = 10.0  # Change the total width of the wall here
        width = (total_width - door_width) / 2
        height = 0.5
        rect1 = patches.Rectangle((-height/2+2.5, door_width/2), height, width, facecolor='red')
        rect2 = patches.Rectangle((-height/2+2.5, -door_width/2-width), height, width, facecolor='red')
        ax1.add_patch(rect1)
        ax1.add_patch(rect2)

    if task == 'rooms':
        ms=10
        off = 0.1
        ax1.plot(0.0, 0.0, 'mo', markersize=ms)
        ax1.text(0.0, 0.0+off, 'S', ha='center', va='bottom', fontsize=label_font_size)
        ax1.plot(11.0, 0.0, 'go', markersize=ms)
        ax1.text(11.0, 0.0+off, 'T1', ha='center', va='bottom', fontsize=label_font_size)
        ax1.plot(11.0, -7.1, 'go', markersize=ms)
        ax1.text(11.0, -7.1+off, 'T2', ha='center', va='bottom', fontsize=label_font_size)
        ax1.plot(11.0, -9.15, 'go', markersize=ms)
        ax1.text(11.0, -9.15+off, 'T3', ha='center', va='bottom', fontsize=label_font_size)
        params = [[-1.15, 0.3, -9.9, 11.4],
                  [-1.15, 13.4, 1.5, 0.3],
                  [-1.15, 13.4, -10.2, 0.3],
                  [-1.15,  5, -1.8, 0.3],
                  [5.05,  3, -1.8, 0.3],
                  [9.25,  3, -1.8, 0.3],
                  [-1.15,  3, -6.1, 0.3],
                  [2.95,  9.3, -6.1, 0.3],
                  [8.95,  3, -8.4, 0.3],
                  [7.3,  0.3, -1.5, 3],
                  [11.95,  0.3, -6.1, 7.9],
                  [6.85,  0.3, -8.1, 2],
                  [4.15,  0.3, -9.9, 2]]
        rects = []
        for p in params:
            rects.append(patches.Rectangle((p[0], p[2]), p[1], p[3], facecolor='red'))
        rects.append(patches.Circle((4.1, -3.8), 0.35, facecolor='red'))
        for rect in rects:
            ax1.add_patch(rect)

    
    # Iterate over the list of headings
    for i, (h, t) in enumerate(headings):
        ax2.plot(trajectories[i][0], h, label=labels[i])
    
    # Add legend for trajectory labels
    if task != 'rooms':
        handles, labels = ax1.get_legend_handles_labels()
        handles.append(arrow)
        labels.append('User Input')
        ax1.legend(handles=handles, labels=labels, handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)}, fontsize=label_font_size, loc='upper right')
    
    fig.tight_layout()
    fig.savefig(name+'.pdf', format='pdf', bbox_inches='tight')
    extent = full_extent(ax1).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(name+'no_heading.pdf', format='pdf', bbox_inches=extent)
    # ax2.savefig('plots/'+name+'_headings.png', format='png', bbox_inches='tight')
    #plt.show()

# define bag names and csv names
bag_names = ['results_bags/box/box1_straight/model1.bag',
             'results_bags/box/box2_straight/model1.bag',
             'results_bags/box/box4_straight/model1.bag']
bag_names = ['results_bags/box/box2_angle/model1.bag',
             'results_bags/box/box4_angle/model1.bag']

folder = 'results_bags/rooms_col/t3'
task = 'rooms'
rectangle_width = 1.0  # Change the width of the rectangle here
rectangle_height = 2.0  # Change the height of the rectangle here
door_width = 1.0 # Change the width of the door here
start = 0.0
TITLE = 'Target 3'

bag_names = []
for file in os.listdir(folder):
    if file.endswith('.bag') and '38' not in file:
        bag_names.append(folder+'/'+file)

bag_names.sort()
plot_names = []
for bag_name in bag_names:
    plot_names.append(bag_name.split('/')[-1].split('.')[0])

starts = [start for _ in range(len(bag_names))]

csv_names = [
    'mycsv.csv',
    ]

csv_names = []

assert len(bag_names) == len(starts)

trajectory_name = folder + '/' + folder.split('/')[-1]
heading_name = folder + '/heading'

trajectories = []
des_velocities = []
headings = []
targets = []
finishing_times = []
for i in range(len(bag_names)):
    bag_name = bag_names[i]
    start = starts[i]
    bag = rosbag.Bag(bag_name)  # Replace with the path to your rosbag file
    x = []
    y = []
    yaw = []
    time = []
    old_time = 0
    old_x = 0
    dt = 0.1
    time_round = 1
    for topic, msg, t in bag.read_messages(topics=['/odometry/filtered', '/odom']):
    #for topic, msg, t in bag.read_messages(topics=['/ridgeback_velocity_controller/odom']):
    # for topic, msg, t in bag.read_messages(topics=['/odom']):
        # Extract the data you need from the rosbag message
        timestamp = round(t.secs + t.nsecs * 1e-9, time_round)
        if timestamp != old_time and abs(msg.pose.pose.position.x) > 1e-5 and abs(msg.pose.pose.position.x - old_x) > 0.001:
            if 'rds' in bag_name and msg.pose.pose.position.x < 0.5:
                continue
            # if 'model1' in bag_name and msg.pose.pose.position.y < 1.2 and msg.pose.pose.position.x > 2:
            #     continue
            y.append(msg.pose.pose.position.y)
            x.append(msg.pose.pose.position.x)
            if 'rds' in bag_name:
                x[-1] = x[-1] - 0.5
            yaw.append(msg.pose.pose.orientation.z)
            time.append(timestamp)
            old_time = timestamp
            old_x = msg.pose.pose.position.x
    des_x = []
    des_y = []
    des_t = []
    old_time = 0
    for topic, msg, t in bag.read_messages(topics=['/desired_velocity']):
        timestamp = round(t.secs + t.nsecs * 1e-9, time_round)
        try:
            if timestamp != old_time and (abs(msg.vector.x) > 0 or abs(msg.vector.y) > 0):
                des_y.append(msg.vector.y)
                des_x.append(msg.vector.x)
                des_t.append(timestamp)
                old_time = timestamp
        except:
            if timestamp != old_time and (abs(msg.linear.x) > 0 or abs(msg.linear.y) > 0):
                des_y.append(msg.linear.y)
                des_x.append(msg.linear.x)
                des_t.append(timestamp)
                old_time = timestamp

    h = []
    h_time = []
    old_time = 0
    for topic, msg, t in bag.read_messages(topics=['/heading']):
        # Extract the data you need from the rosbag message
        timestamp = round(t.secs + t.nsecs * 1e-9, time_round)
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
        timestamp = round(t.secs + t.nsecs * 1e-9, time_round)
        if timestamp != old_time:
            target_y.append(msg.point.y+start)
            target_x.append(msg.point.x)
            target_time.append(timestamp)
            old_time = timestamp

    if len(des_t) > 0:
        time_list = [time, h_time, des_t]
    else:
        time_list = [time, h_time]
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

    tmp = [des_t[i] for i in range(len(des_t)) if des_x[i] != 0 or des_y[i] != 0]
    finishing_times.append(tmp[-1]-tmp[0])

    time = [t - time[0] for t in time]
    # x = [-i for i in x]
    x = [x[i]-x[0] for i in range(len(x))]
    y = [y[i]-y[0]+start for i in range(len(y))]
    trajectories.append((x, y, time))

    h_time = [t - h_time[0] for t in h_time]
    headings.append((h, h_time))

    target_time = [t - target_time[0] for t in target_time]
    # target_x = [-i f or i in target_x]
    targets.append((target_x, target_y, target_time))
    des_t = [t - des_t[0] for t in des_t]
    for i in range(len(des_x)):
        vel = np.array([des_x[i], des_y[i]])
        angle = yaw[i]
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        vel1 = np.dot(rotation_matrix, vel)
        des_x[i] = vel1[0]
        des_y[i] = vel1[1]
    des_velocities.append((des_x, des_y, des_t))

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

for i in range(len(trajectories)):
    print('Model', i+1)
    print('Finishing time:', finishing_times[i])
plot_trajectory(trajectories, headings, des_velocities, targets, task, rectangle_width, rectangle_height, door_width, starts, labels=plot_names, name=trajectory_name)
# plot_headings(headings, labels=plot_names, name=heading_name)