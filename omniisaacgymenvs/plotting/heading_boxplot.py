import rosbag
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import numpy as np
import os

import pdb
import csv
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

LABELS = [f'Model {i+1}' for i in range(6)]
LABELS = ['FC', 'LFC', 'CLFC', 'CLFC_D', 'SCLFC_D_R1', 'SCLFC_D_R2']
TITLES = ['Box 1m [0°]', 'Box 2m [0°]', 'Box 2m [20°]', 'Box 4m [0°]', 'Box 4m [20°]',
          'Door 1m [0°]', 'Door 1m [20°]', 'Door 1.25m [20°]', 'Door 1.25m [0°]', 'Door 1.25m [20°]']
WIDTH = 4
LENGTH = 4
NROWS = 3
NCOLS = 3
def heading_boxplot(all_headings, name, ylabel, figwidth, figlength, nrows, ncols):
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(figwidth, figlength), squeeze=False)
    title_font_size = 8
    label_font_size = 8
    max_ticks = None
    max_y_tick = 0.0
    for i in range(nrows):
        axs[i,0].set_ylabel(ylabel, fontsize=label_font_size)
    
    for i, headings in enumerate(all_headings):
        row = min(int(i/nrows), nrows-1)
        col = int(i%ncols)
        headings = [np.absolute(h[0]) for h in headings]
        # if len(headings) == 3:
        #     while len(headings) < 6:
        #         headings.insert(0, [])

        axs[row,col].set_title(TITLES[i], fontsize=title_font_size)
        axs[row,col].tick_params(axis='both', which='major', labelsize=label_font_size)
        axs[row,col].grid(True)
        plot_labels = LABELS
        if len(LABELS) > len(headings):
            plot_labels = LABELS[i*len(headings):(i+1)*len(headings)]
        if row == nrows-1:
            axs[row,col].boxplot(headings, labels=plot_labels, flierprops=dict(markersize=3))
            axs[row,col].tick_params(axis='x', rotation=90)
        else:
            axs[row,col].boxplot(headings, flierprops=dict(markersize=3))
            axs[row,col].set_xticklabels([])
        
        # Set the y-axis ticks to be the same for all subplots
        axs[row,col].set_ylim(ymin=0)
        if max(axs[row, col].get_yticks()) > max_y_tick:
            max_ticks = axs[row, col].get_yticks()
            max_y_tick = max(max_ticks)

    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].set_yticks(max_ticks)
            axs[i,j].set_ylim(0, max_y_tick)
    
    fig.tight_layout(h_pad=0)
    fig.savefig(name+'.pdf', format='pdf', bbox_inches='tight')
    #plt.show()


folders = ['results_bags/box/', 'results_bags/door/']
folders = ['results_bags/rooms_col/']
folders = ['daav_bags/reward2_box_door_5trials/']
folders = ['daav_bags/model_comparison_box_door/']
folders = ['daav_bags/videos/']
completion_check = False
rds = False
rds_tests = ['box1_straight', 'box2_xangle', 'door1_25_straight', 'door1_25_xangle']
if rds:
    LABELS = ['SCLFC_D_R2', 'RDS']
    TITLES = ['Box 1m [0°]', 'Box 2m [20°]', 'Door 1.25m [0°]', 'Door 1.25m [20°]']


all_headings = []
all_velocities = []
all_accs = []
all_jerks = []
heading_name = 'results_bags/heading'
vel_name = 'results_bags/velocity'
acc_name = 'results_bags/acceleration'
jerk_name = 'results_bags/jerk'
if rds:
    heading_name = 'results_bags/heading_rds'
    vel_name = 'results_bags/velocity_rds'
    acc_name = 'results_bags/acceleration_rds'
    jerk_name = 'results_bags/jerk_rds'
if 'rooms_col' in folders[0]:
    LABELS = [f'Trial {i+1}' for i in range(5)]
    TITLES = [f'Target {i+1}' for i in range(3)]

LABELS = [f'Trial {i+1}' for i in range(5)]
LABELS = ['CLFC_D', 'SCLFC_D_R1', 'SCLFC_D_R2 [0.4]', 'SCLFC_D_R2 [0.5]', 'SCLFC_D_R2 [0.6]']
LABELS = ['SCLFC_D_R1 Small Box', 'SCLFC_D_R2 Small Box', 'SCLFC_D_R2 Large Box', 'SCLDF_D_R1 Door', 'SCLFC_D_R2 Door', 'SCLFC_D_R2 Door + Desk']
TITLES = ['Box [~0°]', 'Door [~20°]']
WIDTH = 4
LENGTH = 3
NROWS = 1
NCOLS = 2
if len(folders) == 1:
    heading_name = folders[0] + 'heading'
    vel_name = folders[0] + 'velocity'
    acc_name = folders[0] + 'acceleration'
    jerk_name = folders[0] + 'jerk'
for test in folders:
    subtests = [test + subtest + '/' for subtest in os.listdir(test) if os.path.isdir(test + subtest) and '1_5' not in subtest]
    if rds:
        subtests = [test + subtest + '/' for subtest in os.listdir(test) if os.path.isdir(test + subtest) and subtest in rds_tests]
    subtests.sort()
    for folder in subtests:

        bag_names = []
        for file in os.listdir(folder):
            if file.endswith('.bag') and '1_5' not in file and not rds and 'rds' not in file:
                bag_names.append(folder+'/'+file)
            elif file.endswith('.bag') and rds and ('rds' in file or '6' in file):
                bag_names.append(folder+'/'+file)

        bag_names.sort()
        plot_names = []
        for bag_name in bag_names:
            plot_names.append(bag_name.split('/')[-1].split('.')[0])

        headings = []
        velocities = []
        accs = []
        jerks = []
        targets = []
        finishing_times = []
        for i in range(len(bag_names)):
            bag_name = bag_names[i]
            bag = rosbag.Bag(bag_name)  # Replace with the path to your rosbag file
            x = []
            x_vel = []
            y_vel = []
            ang_vel = []
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
                # if timestamp != old_time and abs(msg.pose.pose.position.x) > 1e-5 and abs(msg.pose.pose.position.x - old_x) > 0.001:
                if timestamp != old_time:
                    y.append(msg.pose.pose.position.y)
                    x.append(msg.pose.pose.position.x)
                    x_vel.append(msg.twist.twist.linear.x)
                    y_vel.append(msg.twist.twist.linear.y)
                    ang_vel.append(msg.twist.twist.angular.z)
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
                    target_y.append(msg.point.y)
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
            x_vel = [x_vel[i] for i in range(len(time)) if time[i] in all_times]
            y_vel = [y_vel[i] for i in range(len(time)) if time[i] in all_times]
            ang_vel = [ang_vel[i] for i in range(len(time)) if time[i] in all_times]
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
            x = [x[i]-x[0] for i in range(len(x))]
            y = [y[i]-y[0] for i in range(len(y))]

            h_time = [t - h_time[0] for t in h_time]
            h_red = [h[i] for i in range(len(h)) if des_x[i] != 0 or des_y[i] != 0]
            if np.all(np.array(x) < 4.0) and completion_check: h_red=[]
            headings.append((h_red, h_time))

            x_vel_red = [x_vel[i] for i in range(len(x_vel)) if des_x[i] != 0 or des_y[i] != 0]
            y_vel_red = [y_vel[i] for i in range(len(y_vel)) if des_x[i] != 0 or des_y[i] != 0]
            vel_time_red = [time[i] for i in range(len(time)) if des_x[i] != 0 or des_y[i] != 0]
            ang_vel_red = [ang_vel[i] for i in range(len(ang_vel)) if des_x[i] != 0 or des_y[i] != 0]
            vel = [np.sqrt(x_vel_red[i]**2 + y_vel_red[i]**2 + ang_vel_red[i]**2) for i in range(len(x_vel_red))]
            # acc = [abs((vel[i+1] - vel[i])/(vel_time_red[i+1] - vel_time_red[i])) for i in range(len(vel)-1)]
            # jerk = [abs((acc[i+1] - acc[i])/(vel_time_red[i+1] - vel_time_red[i])) for i in range(len(acc)-1)]

            # dt_vel = [vel_time_red[i+1] - vel_time_red[i] for i in range(len(vel_time_red)-1)]
            x_acc = [(x_vel_red[i+1] - x_vel_red[i])/dt for i in range(len(x_vel_red)-1)]
            y_acc = [(y_vel_red[i+1] - y_vel_red[i])/dt for i in range(len(y_vel_red)-1)]
            ang_acc = [(ang_vel_red[i+1] - ang_vel_red[i])/dt for i in range(len(ang_vel_red)-1)]
            acc = [np.sqrt(x_acc[i]**2 + y_acc[i]**2 + ang_acc[i]**2) for i in range(len(x_acc))]

            x_jerk = [(x_acc[i+1] - x_acc[i])/dt for i in range(len(x_acc)-1)]
            y_jerk = [(y_acc[i+1] - y_acc[i])/dt for i in range(len(y_acc)-1)]
            ang_jerk = [(ang_acc[i+1] - ang_acc[i])/dt for i in range(len(ang_acc)-1)]
            jerk = [np.sqrt(x_jerk[i]**2 + y_jerk[i]**2 + ang_jerk[i]**2) for i in range(len(x_jerk))]

            if np.all(np.array(x) < 4.0) and completion_check: 
                vel, acc, jerk =[], [], []

            velocities.append((vel, vel_time_red))
            accs.append((acc, vel_time_red[:-1]))
            jerks.append((jerk, vel_time_red[:-2]))

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

            bag.close()
        
        all_headings.append(headings)
        all_velocities.append(velocities)
        all_accs.append(accs)
        all_jerks.append(jerks)

heading_boxplot(all_headings, heading_name, 'Abs Heading [°]', WIDTH, LENGTH, NROWS, NCOLS)
heading_boxplot(all_velocities, vel_name, 'Velocity', WIDTH, LENGTH, NROWS, NCOLS)
heading_boxplot(all_accs, acc_name, 'Acceleration', WIDTH, LENGTH, NROWS, NCOLS)
heading_boxplot(all_jerks, jerk_name, 'Jerk', WIDTH, LENGTH, NROWS, NCOLS)
# plot_headings(headings, labels=plot_names, name=heading_name)