import numpy as np
import os
import numpy as np
import csv
import sys
import time
import json
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')

# file_path = os.path.dirname(os.path.abspath(__file__))
# cur_path = os.path.dirname(file_path)+'/'
cur_path = os.getcwd() + '/data/drone_data/'
sys.path.append(cur_path)
from utils import data_load, coordinate_conversion, get_nearest_link

print('Starting KAIST dataset processing')
print('Data list loading ...\n')

file_list = os.listdir(cur_path + 'raw/landmark')
file_list_int = np.zeros(len(file_list), dtype=int)
for i in range(len(file_list)):
    file_list_int[i] = int(file_list[i][0:4])

print('------------------------------------------------------------')
for i in range(len(file_list_int)):
    print('File_id : ' + str(file_list_int[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')
selected_file_index = input('Select data file index from above :')

selected_scenario_id = file_list_int[int(selected_file_index)]

print('scenario ' + str(selected_scenario_id) + ' is selected')

print('\n')
print('Data loading ....')

landmark, recordingMeta, tracks, tracksMeta, tracksClass = data_load(cur_path, selected_scenario_id)
scale = 1.5
origin_GT = [[4311.99951171875, 697.5087890625],
             [4323.337890625, 668.513671875],
             [4299.55908203125, 659.2275390625]]

new_tracks = coordinate_conversion(scale, tracks, landmark, recordingMeta, origin_GT)

with open(cur_path + 'map/' + str(selected_scenario_id) + '/link_set_mod.json') as json_file:
    links = json.load(json_file)
with open(cur_path + 'map/' + str(selected_scenario_id) + '/node_set_new.json') as json_file:
    nodes = json.load(json_file)

for i in range(len(links)):
    points_np = np.array(links[i]['points'])[:, :2]
    plt.plot(np.array(points_np)[:, 0], np.array(points_np)[:, 1], c='k')

plt.axis('equal')
plt.show()

veh_idx = tracksMeta[(tracksMeta[:, 6] > 0) & (tracksMeta[:, 4] > 0), 1]
for i in range(len(veh_idx)):
    traj = new_tracks[new_tracks[:, 1] == veh_idx[i], 4:6]
    move_check = 0
    for j in range(len(traj) - 5):
        displacement = np.linalg.norm(traj[j + 5] - traj[j])
        if displacement > 0.1:
            if move_check == 0:
                start_index_cand = j
            move_check = move_check + 1
            if move_check == 10:
                start_index = start_index_cand
                break
        else:
            move_check = 0
    traj = traj[start_index:]

    seg_list = []
    for jjj in range(len(traj)):
        init_seg_int, _ = get_nearest_link(links, traj[jjj,:])
        seg_list.append(init_seg_int)
    init_pos = traj[0, :]
    init_seg_int, init_seg = get_nearest_link(links, init_pos)
    end_pos = traj[-1, :]
    end_seg_int, end_seg = get_nearest_link(links, end_pos)
    if min(seg_list)==0 and end_seg_int > 0:
        origin_point = init_seg['points'][-1]
        origin_heading = np.arctan2(init_seg['points'][-1][1] - init_seg['points'][-2][1],
                                    init_seg['points'][-1][0] - init_seg['points'][-2][0])
        origin_heading = np.mod(np.rad2deg(origin_heading), 360)
        rot = np.asarray([[np.cos(np.deg2rad(origin_heading)), -np.sin(np.deg2rad(origin_heading))],
                          [np.sin(np.deg2rad(origin_heading)), np.cos(np.deg2rad(origin_heading))]])

        target_point = end_seg['points'][0]
        target_heading = np.arctan2(end_seg['points'][1][1] - end_seg['points'][0][1],
                                    init_seg['points'][1][0] - init_seg['points'][0][0])
        target_heading = np.mod(np.rad2deg(target_heading), 360)

        traj_conv = np.matmul(traj - origin_point, rot)
        plt.plot(traj[:, 0], traj[:, 1])


for i in range(len(links)):
    points_np = np.array(links[i]['points'])[:, :2]
    points_np_conv = np.matmul(points_np - origin_point, rot)
    plt.plot(points_np_conv[:, 0], points_np_conv[:, 1], c='r')
# plt.plot(traj[:,0], traj[:,1])
plt.scatter(traj_conv[:, 0], traj_conv[:, 1])
plt.axis('equal')
