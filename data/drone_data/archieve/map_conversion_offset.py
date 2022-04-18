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
ddx = np.linspace(-10, 10, 50)
ddy = np.linspace(-10, 10, 50)
for ii in range(len(ddx)):
    for jj in range(len(ddy)):
        dx = ddx[ii]
        dy = ddy[jj]
        origin_GT = [[4312.063+dx, 697.521+dy],
                     [4323.276+dx, 668.434+dy],
                     [4299.597+dx, 659.215+dy]]

        new_tracks = coordinate_conversion(tracks, landmark, recordingMeta, origin_GT)

        with open(cur_path + 'map/' + str(selected_scenario_id) + '/link_set_mod.json') as json_file:
            links = json.load(json_file)

        for i in range(len(links)):
            points_np = np.array(links[i]['points'])[:, :2]
            plt.plot(np.array(points_np)[:, 0], np.array(points_np)[:, 1], c='k')

        plt.axis('equal')
        plt.show()

        veh_idx = tracksMeta[(tracksMeta[:, 6] > 0) & (tracksMeta[:, 4] > 0), 1]

        for i in range(len(veh_idx)):
            traj = new_tracks[new_tracks[:, 1] == veh_idx[i], 4:6]
            plt.plot(traj[:, 0], traj[:, 1])
        plt.savefig(str(dx) + '_' + str(dy) + '.png')
        plt.close()

    init_pos = traj[0,:]
    init_seg = get_nearest_link(links, init_pos)
    end_pos = traj[-1,:]
    end_seg = get_nearest_link(links, end_pos)
