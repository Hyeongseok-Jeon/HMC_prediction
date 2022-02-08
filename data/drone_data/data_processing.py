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

# cur_path = os.path.dirname(os.path.abspath(__file__))+'/'
cur_path = os.getcwd() + '/data/drone_data/'
print(cur_path)
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

# issued index
# conversion error = [1010, 1003] [10, 6]
for file_name_index in range(len(file_list_int)):

    # selected_file_index = input('Select data file index from above :')
    # selected_file_index = input('Proceed?: ')

    selected_scenario_id = file_list_int[file_name_index]

    if '10' in str(selected_scenario_id):
        print('scenario ' + str(selected_scenario_id) + ' is selected')

        print('\n')
        print('Data loading ....')

        landmark, recordingMeta, tracks, tracksMeta, tracksClass = data_load(cur_path, selected_scenario_id)
        if selected_scenario_id == 1013:
            landmark[29,2:] = landmark[30,2:]
        if '10' in str(selected_scenario_id):
            representative_id = '1001'

        origin_GT = []
        with open(cur_path + 'map/' + representative_id + '/csv/LandMark.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                row_mod = row[0].split(',')
                if row_mod[0] == 'id':
                    pass
                else:
                    origin_GT.append([float(row_mod[1]), float(row_mod[2])])

        if selected_scenario_id == 3001:
            scale = 1.4
        else:
            scale = 1

        print('Data Converting ....')

        new_tracks = coordinate_conversion(scale, tracks, landmark, recordingMeta, origin_GT)

        with open(cur_path + 'map/' + representative_id + '/link_set.json') as json_file:
            links = json.load(json_file)
        with open(cur_path + 'map/' + representative_id + '/node_set.json') as json_file:
            nodes = json.load(json_file)
        maneuver_table = np.load(cur_path + 'map/' + representative_id + '/maneuver_table.npy')

        plt.figure(str(selected_scenario_id)+'_1')
        for i in range(len(links)):
            points_np = np.array(links[i]['points'])[:, :2]
            plt.plot(np.array(points_np)[:, 0], np.array(points_np)[:, 1], c='k')
        # plt.scatter(new_tracks[:,4], new_tracks[:,5])
        plt.axis('equal')
        plt.show()
        for i in range(47):
            x = new_tracks[new_tracks[:,1]==i,4]
            y = new_tracks[new_tracks[:,1]==i,5]
            plt.plot(x, y, linewidth=3)
            try:
                plt.text(x[0], y[0], str(int(new_tracks[new_tracks[:,1]==i,1][0])))
            except:
                pass
            # plt.pause(0.05)
            # input('press enter')
        plt.pause(0.05)

        print('\n')
        print('Data Extracting ....')

        plt.figure(str(selected_scenario_id)+'_2')
        veh_idx = tracksMeta[(tracksMeta[:, 6] > 0) & (tracksMeta[:, 4] > 0), 1]
        for i in range(len(veh_idx)):
            if selected_scenario_id == 1001 and i == 2:
                pass
            elif selected_scenario_id == 1010 and i == 51:
                pass
            else:
                if i+1 == len(veh_idx):
                    print(str(int(10000 * (i + 1) / len(veh_idx)) / 100) + ' % is completed')
                else:
                    print(str(int(10000 * (i + 1) / len(veh_idx)) / 100) + ' % is completed', end='\r')

                index_mask = '0000'
                veh_id = str(int(veh_idx[i]))
                veh_id_sort = index_mask[:-len(veh_id)] + veh_id
                try:
                    traj = new_tracks[new_tracks[:, 1] == veh_idx[i], 4:6]
                    heading = new_tracks[new_tracks[:, 1] == veh_idx[i], 6:7]
                    move_check = 0
                    for j in range(len(traj) - 5):
                        displacement = np.linalg.norm(traj[j + 5] - traj[j])
                        if displacement > 0.1:
                            if move_check == 0:
                                start_index_cand = j
                            move_check = move_check + 1
                            if move_check == 20:
                                start_index = start_index_cand
                                break
                        else:
                            move_check = 0

                    while True:
                        init_pos = traj[start_index, :]
                        init_seg_int, init_seg, min_dist = get_nearest_link(links, init_pos)
                        if min_dist < 1.5:
                            break
                        else:
                            start_index = start_index+1

                    traj = traj[start_index:]
                    heading = heading[start_index:]

                    seg_list = []
                    for jjj in range(len(traj)):
                        init_seg_int, _, _ = get_nearest_link(links, traj[jjj,:])
                        seg_list.append(init_seg_int)

                    for init_index in range(len(traj)):
                        init_pos = traj[init_index, :]
                        init_seg_int, init_seg, _ = get_nearest_link(links, init_pos)
                        if init_seg_int == -1:
                            pass
                        else:
                            break

                    if selected_scenario_id == 1014 and i == 28:
                        init_seg_int = 5
                        for asdf in range(len(links)):
                            seg = links[asdf]
                            if seg['idx_int'] == init_seg_int:
                                init_seg = seg

                    end_pos = traj[-1, :]
                    end_seg_int, end_seg, _ = get_nearest_link(links, end_pos)
                    if 0 in seg_list and end_seg_int != 0 and init_seg_int != 0:
                        # plt.plot(traj[:, 0], traj[:, 1])

                        origin_point = init_seg['points'][-1]
                        origin_heading = np.arctan2(init_seg['points'][-1][1] - init_seg['points'][-2][1],
                                                    init_seg['points'][-1][0] - init_seg['points'][-2][0])
                        origin_heading = np.mod(np.rad2deg(origin_heading), 360)
                        rot = np.asarray([[np.cos(np.deg2rad(-origin_heading)), -np.sin(np.deg2rad(-origin_heading))],
                                          [np.sin(np.deg2rad(-origin_heading)), np.cos(np.deg2rad(-origin_heading))]])

                        outlet_node = end_seg['points'][0]
                        target_index = np.argmin(np.linalg.norm(traj - outlet_node,axis=1))

                        traj_conv = np.matmul(rot, (traj - origin_point).T).T
                        heading_conv = heading + 90 - origin_heading
                        inlet_idx = np.argmin(np.linalg.norm(traj_conv, axis=1))
                        # TODO: hist_traj길이를 intersection 중간지점까지 확장할 필요도 있을듯

                        hist_traj = np.concatenate((traj_conv[:inlet_idx+1, :], heading_conv[:inlet_idx+1]), axis=1)
                        outlet_state = np.concatenate((traj_conv[target_index:target_index+1, :], heading_conv[target_index:target_index+1]), axis=1)
                        total_traj = np.concatenate((traj_conv, heading_conv), axis=1)
                        maneuver_index = np.zeros(shape=(4,))
                        maneuver_index[int(maneuver_table[init_seg_int, end_seg_int])] = 1

                        file_name = str(selected_scenario_id) + '_'+ veh_id_sort
                        with open(cur_path + 'processed/hist_traj/' + file_name + '.npy', "wb") as f:
                            np.save(f, hist_traj)
                        with open(cur_path + 'processed/maneuver_index/' + file_name + '.npy', "wb") as f:
                            np.save(f, maneuver_index)
                        with open(cur_path + 'processed/outlet_state/' + file_name + '.npy', "wb") as f:
                            np.save(f, outlet_state)
                        with open(cur_path + 'processed/total_traj/' + file_name + '.npy', "wb") as f:
                            np.save(f, total_traj)
                        plt.scatter(traj_conv[:, 0], traj_conv[:, 1])
                        plt.text(traj_conv[-1, 0], traj_conv[-1,1], veh_id_sort)
                        # plt.scatter(hist_traj[:,0], hist_traj[:,1])
                except:
                    pass
        plt.pause(0.05)

        print('\n')
        print('Done!!')

#
#
# for i in range(len(links)):
#     points_np = np.array(links[i]['points'])[:, :2]
#     points_np_conv = np.matmul(points_np - origin_point, rot)
#     plt.plot(points_np_conv[:, 0], points_np_conv[:, 1], c='r')
# # plt.plot(traj[:,0], traj[:,1])
# plt.scatter(traj_conv[:, 0], traj_conv[:, 1])
# plt.scatter(traj_conv[inlet_idx, 0], traj_conv[inlet_idx,1])
# plt.scatter(hist_traj[:,0], hist_traj[:,1])
# plt.axis('equal')

'''
np.load('data/drone_data/processed/maneuver_index/1001_0022.npy')

'''