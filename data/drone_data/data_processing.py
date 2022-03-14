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
val_rate = 0.2
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
# plt.figure()

for file_name_index in range(len(file_list_int)):
    # if file_list_int[file_name_index] == 1003:
    #     print(file_name_index)
    # selected_file_index = input('Select data file index from above :')
    # selected_file_index = input('Proceed?: ')

# 1011_0027
# 1016_0001
# 1016_0002
# 1014_0006
# 1014_0049
# 1014_0013
# 1010_0017


    selected_scenario_id = file_list_int[file_name_index]

    if '10' in str(selected_scenario_id):
        print('scenario ' + str(selected_scenario_id) + ' is selected')

        print('\n')
        print('Data loading ....')

        landmark, recordingMeta, tracks, tracksMeta, tracksClass = data_load(cur_path, selected_scenario_id)
        if selected_scenario_id == 1013:
            landmark[29, 2:] = landmark[30, 2:]
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
        new_tracks_tmp = new_tracks.copy()
        if selected_scenario_id == 1003:
            new_tracks[:, 5] = new_tracks[:, 5] - 13
        with open(cur_path + 'map/' + representative_id + '/link_set.json') as json_file:
            links = json.load(json_file)
        with open(cur_path + 'map/' + representative_id + '/node_set.json') as json_file:
            nodes = json.load(json_file)
        maneuver_table = np.load(cur_path + 'map/' + representative_id + '/maneuver_table.npy')

        # for i in range(len(links)):
        #     points_np = np.array(links[i]['points'])[:, :2]
        #     plt.plot(np.array(points_np)[:, 1], -np.array(points_np)[:, 0], c='k', zorder=100, linewidth=3)
        # # plt.scatter(new_tracks[:,4], new_tracks[:,5])
        # plt.axis('equal')
        # plt.show()
        # for i in range(int(max(new_tracks[:, 1])) + 1):
        #     try:
        #         idxe = int(new_tracks[new_tracks[:, 1] == i, 1][0])
        #         obj_class = tracksClass[np.where(tracksMeta[:,1] == idxe)[0][0]]
        #         x = new_tracks[new_tracks[:, 1] == i, 4]
        #         y = new_tracks[new_tracks[:, 1] == i, 5]
        #         if obj_class == 'pedestrian':
        #             plt.plot(y, -x, color='c',linewidth=1)
        #         elif obj_class == 'bicycle':
        #             plt.plot(y, -x, color='b',linewidth=1)
        #         else:
        #             plt.plot(y, -x, color='lightcoral',linewidth=1)
        #         # plt.text(y[0], -x[0], str(int(new_tracks[new_tracks[:, 1] == i, 1][0])))
        #     except:
        #         pass
        #     # plt.pause(0.05)
        #     # input('press enter')
        # plt.pause(0.05)

        print('\n')
        print('Data Extracting ....')

        # plt.figure(str(selected_scenario_id) + '_2')
        veh_idx = tracksMeta[(tracksMeta[:, 6] > 0) & (tracksMeta[:, 4] > 0), 1]
        for i in range(len(veh_idx)):
            if i + 1 == len(veh_idx):
                print(str(int(10000 * (i + 1) / len(veh_idx)) / 100) + ' % is completed')
            else:
                print(str(int(10000 * (i + 1) / len(veh_idx)) / 100) + ' % is completed', end='\r')

            index_mask = '0000'
            veh_id = str(int(veh_idx[i]))
            veh_id_sort = index_mask[:-len(veh_id)] + veh_id
            traj = new_tracks[new_tracks[:, 1] == veh_idx[i], 4:6]
            heading = new_tracks[new_tracks[:, 1] == veh_idx[i], 6:7]

            data_gen = 1
            seg_list = []
            for jjj in range(len(traj)):
                init_seg_int, _, _ = get_nearest_link(links, traj[jjj, :])
                seg_list.append(init_seg_int)

            if not (0 in seg_list):
                continue

            idx_in = seg_list.index(0)
            seg_list.reverse()
            if seg_list.index(0) == 0:
                continue

            idx_out = len(seg_list) - 1 - seg_list.index(0)
            seg_list.reverse()

            if idx_in == 0:
                continue

            inlet_index = seg_list[idx_in - 1]
            outlet_index = seg_list[idx_out + 1]

            if inlet_index == 4:
                inlet_index_for_seg = 5
            elif inlet_index == 7:
                inlet_index_for_seg = 8
            elif inlet_index == 11:
                inlet_index_for_seg = 12
            elif inlet_index == 14:
                inlet_index_for_seg = 1
            else:
                inlet_index_for_seg = inlet_index

            if outlet_index == 5:
                outlet_index_for_seg = 4
            elif outlet_index == 8:
                outlet_index_for_seg = 7
            elif outlet_index == 12:
                outlet_index_for_seg = 11
            elif outlet_index == 1:
                outlet_index_for_seg = 14
            else:
                outlet_index_for_seg = outlet_index

            for i in range(len(links)):
                if links[i]['idx_int'] == inlet_index_for_seg:
                    init_seg = links[i]
                if links[i]['idx_int'] == outlet_index_for_seg:
                    end_seg = links[i]

            #
            # for init_index in range(len(traj)):
            #     init_pos = traj[init_index, :]
            #     init_seg_int, init_seg, _ = get_nearest_link(links, init_pos)
            #     if init_seg_int == -1:
            #         pass
            #     else:
            #         break
            #
            # if selected_scenario_id == 1014 and i == 28:
            #     init_seg_int = 5
            #     for asdf in range(len(links)):
            #         seg = links[asdf]
            #         if seg['idx_int'] == init_seg_int:
            #             init_seg = seg
            #
            # if selected_scenario_id == 1010 and veh_id == '72':
            #     init_seg_int = 5
            #     for asdf in range(len(links)):
            #         seg = links[asdf]
            #         if seg['idx_int'] == init_seg_int:
            #             init_seg = seg
            # end_pos = traj[-1, :]
            # end_seg_int, end_seg, _ = get_nearest_link(links, end_pos)

            if data_gen == 1:
                # plt.plot(traj[:, 0], traj[:, 1])

                origin_point = init_seg['points'][-1]
                origin_heading = np.arctan2(init_seg['points'][-1][1] - init_seg['points'][-2][1],
                                            init_seg['points'][-1][0] - init_seg['points'][-2][0])
                origin_heading = np.mod(np.rad2deg(origin_heading), 360)
                rot = np.asarray([[np.cos(np.deg2rad(-origin_heading)), -np.sin(np.deg2rad(-origin_heading))],
                                  [np.sin(np.deg2rad(-origin_heading)), np.cos(np.deg2rad(-origin_heading))]])

                outlet_seg_points = np.asarray(end_seg['points'].copy())
                target_index = np.argmin(np.linalg.norm(traj - outlet_seg_points[0], axis=1))

                traj_conv = np.matmul(rot, (traj - origin_point).T).T
                outlet_seg_conv = np.matmul(rot, (outlet_seg_points - origin_point).T).T
                outlet_seg_heading = np.rad2deg(np.arctan2(outlet_seg_conv[1][1] - outlet_seg_conv[0][1],
                                                           outlet_seg_conv[1][0] - outlet_seg_conv[0][0]))
                outlet_seg_heading = np.expand_dims(outlet_seg_heading, axis=0)
                outlet_seg_heading = np.expand_dims(outlet_seg_heading, axis=1)

                heading_conv = heading + 90 - origin_heading
                outlet_node_state = np.concatenate((outlet_seg_conv[0:1, :],outlet_seg_heading),axis=1)

                inlet_idx = np.argmin(np.linalg.norm(traj_conv, axis=1))
                # TODO: hist_traj길이를 intersection 중간지점까지 확장할 필요도 있을듯

                hist_traj = np.concatenate((traj_conv[:inlet_idx + 1, :], heading_conv[:inlet_idx + 1]), axis=1)
                nearest_outlet_state = np.concatenate(
                    (traj_conv[target_index:target_index + 1, :], heading_conv[target_index:target_index + 1]),
                    axis=1)
                total_traj = np.concatenate((traj_conv, heading_conv), axis=1)
                maneuver_index = np.zeros(shape=(4,))
                maneuver_index[int(maneuver_table[inlet_index, outlet_index])] = 1

                file_name = str(selected_scenario_id) + '_' + veh_id_sort
                # with open(cur_path + 'processed/hist_traj/' + file_name + '.npy', "wb") as f:
                #     np.save(f, hist_traj)
                if np.random.rand() < 0.2:
                    cat = 'val'
                else:
                    cat = 'train'
                with open(cur_path + 'processed/' + cat + '/maneuver_index/' + file_name + '.npy', "wb") as f:
                    np.save(f, maneuver_index)
                with open(cur_path + 'processed/' + cat + '/nearest_outlet_state/' + file_name + '.npy', "wb") as f:
                    np.save(f, nearest_outlet_state)
                with open(cur_path + 'processed/' + cat + '/outlet_node_state/' + file_name + '.npy', "wb") as f:
                    np.save(f, outlet_node_state)
                with open(cur_path + 'processed/' + cat + '/total_traj/' + file_name + '.npy', "wb") as f:
                    np.save(f, total_traj)
                with open(cur_path + 'processed/' + cat + '/link_idx/' + file_name + '.npy', "wb") as f:
                    np.save(f, seg_list)
                with open(cur_path + 'processed/' + cat + '/conversion/' + file_name + '.npy', "wb") as f:
                    np.save(f, np.asarray([origin_point] + rot.tolist()))
                # plt.scatter(traj_conv[:, 0], traj_conv[:, 1])
                # plt.text(traj_conv[-1, 0], traj_conv[-1, 1], veh_id_sort)
                # plt.scatter(hist_traj[:,0], hist_traj[:,1])

        # plt.pause(0.05)

        print('\n')
        print('Done!!')

print('program finished')
# plt.pause(100000)
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
