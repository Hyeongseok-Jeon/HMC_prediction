import glob
import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import sys
# file_path = os.path.dirname(os.path.abspath(__file__))
# cur_path = os.path.dirname(file_path)+'/'
cur_path = os.getcwd() + '/data/drone_data/'
data_bag = glob.glob(cur_path + '/raw/tracks/*.csv')
sys.path.append(cur_path)
from utils import point_regen

data_idx = 8
data_id = os.path.basename(data_bag[data_idx]).split("_")[0]
if '10' in data_id:
    data_id = '1001'

with open(cur_path + 'MORAI_map/' + data_id + '/link_set_new.json') as json_file:
    links = json.load(json_file)

with open(cur_path + 'MORAI_map/' + data_id + '/node_set_new.json') as json_file:
    nodes = json.load(json_file)

branch_cnt = 0
for i in range(len(links)):
    points_np = np.array(links[i]['points'])[:, :2]
    nodes_in = points_np[0,:]
    nodes_in_idx = links[i]['from_node_idx']
    nodes_out = points_np[-1,:]
    nodes_out_idx = links[i]['to_node_idx']

    list_pt = []
    for j in range(len(points_np)-1):
        dist_to_next = np.linalg.norm(points_np[j+1]-points_np[j])
        list_pt.append(points_np[j,:])
        if dist_to_next > 1:
            inter_num = int(np.ceil(dist_to_next))
            for z in range(inter_num-1):
                x = points_np[j,0] + (points_np[j+1,0]-points_np[j,0])*z/inter_num
                y = points_np[j,1] + (points_np[j+1,1]-points_np[j,1])*z/inter_num
                list_pt.append(np.asarray([x,y]))
    list_pt.append(points_np[-1,:])
    points_np_interpolate = np.asarray(list_pt)

    pt_list = point_regen(points_np_interpolate)

    links[i]['points'] = pt_list.tolist()
    plt.scatter(np.array(pt_list)[:, 0], np.array(pt_list)[:, 1])
    plt.text((np.array(pt_list)[0, 0]),
             (np.array(pt_list)[0, 1]),
             str(links[i]['idx_int']))
    if links[i]['idx_int']>0:
        branch_cnt = branch_cnt+1
    init = 0

plt.axis('equal')
plt.show()

with open(cur_path + 'MORAI_map/' + data_id + '/link_set_mod.json', "w") as json_file:
    json.dump(links, json_file)

maneuver_table = np.zeros([15,15])
maneuver_table[1,11] = 1
maneuver_table[1,12] = 1
maneuver_table[2,11] = 1
maneuver_table[2,12] = 1
maneuver_table[3,11] = 1
maneuver_table[3,12] = 1
maneuver_table[1,6] = 2
maneuver_table[1,7] = 2
maneuver_table[1,8] = 2
maneuver_table[1,9] = 2
maneuver_table[1,10] = 2
maneuver_table[2,6] = 2
maneuver_table[2,7] = 2
maneuver_table[2,8] = 2
maneuver_table[2,9] = 2
maneuver_table[2,10] = 2
maneuver_table[3,6] = 2
maneuver_table[3,7] = 2
maneuver_table[3,8] = 2
maneuver_table[3,9] = 2
maneuver_table[3,10] = 2
maneuver_table[1,4] = 3
maneuver_table[1,5] = 3
maneuver_table[2,4] = 3
maneuver_table[2,5] = 3
maneuver_table[3,4] = 3
maneuver_table[3,5] = 3

maneuver_table[4,1] = 1
maneuver_table[4,2] = 1
maneuver_table[4,3] = 1
maneuver_table[4,13] = 1
maneuver_table[4,14] = 1
maneuver_table[5,1] = 1
maneuver_table[5,2] = 1
maneuver_table[5,3] = 1
maneuver_table[5,13] = 1
maneuver_table[5,14] = 1
maneuver_table[4,11] = 2
maneuver_table[4,12] = 2
maneuver_table[5,11] = 2
maneuver_table[5,12] = 2
maneuver_table[4,6] = 3
maneuver_table[4,7] = 3
maneuver_table[4,8] = 3
maneuver_table[4,9] = 3
maneuver_table[4,10] = 3
maneuver_table[5,6] = 3
maneuver_table[5,7] = 3
maneuver_table[5,8] = 3
maneuver_table[5,9] = 3
maneuver_table[5,10] = 3

maneuver_table[8,4] = 1
maneuver_table[8,5] = 1
maneuver_table[9,4] = 1
maneuver_table[9,5] = 1
maneuver_table[10,4] = 1
maneuver_table[10,5] = 1
maneuver_table[8,1] = 2
maneuver_table[8,2] = 2
maneuver_table[8,3] = 2
maneuver_table[8,13] = 2
maneuver_table[8,14] = 2
maneuver_table[9,1] = 2
maneuver_table[9,2] = 2
maneuver_table[9,3] = 2
maneuver_table[9,13] = 2
maneuver_table[9,14] = 2
maneuver_table[10,1] = 2
maneuver_table[10,2] = 2
maneuver_table[10,3] = 2
maneuver_table[10,13] = 2
maneuver_table[10,14] = 2
maneuver_table[8,11] = 3
maneuver_table[8,12] = 3
maneuver_table[9,11] = 3
maneuver_table[9,12] = 3
maneuver_table[10,11] = 3
maneuver_table[10,12] = 3

maneuver_table[11,6] = 1
maneuver_table[11,7] = 1
maneuver_table[11,8] = 1
maneuver_table[11,9] = 1
maneuver_table[11,10] = 1
maneuver_table[12,6] = 1
maneuver_table[12,7] = 1
maneuver_table[12,8] = 1
maneuver_table[12,9] = 1
maneuver_table[12,10] = 1
maneuver_table[11,4] = 2
maneuver_table[11,5] = 2
maneuver_table[12,4] = 2
maneuver_table[12,5] = 2
maneuver_table[11,1] = 3
maneuver_table[11,2] = 3
maneuver_table[11,3] = 3
maneuver_table[11,13] = 3
maneuver_table[11,14] = 3
maneuver_table[11,1] = 3
maneuver_table[11,2] = 3
maneuver_table[11,3] = 3
maneuver_table[12,13] = 3
maneuver_table[12,14] = 3


with open(cur_path + 'MORAI_map/' + data_id + '/maneuver_table.npy', "wb") as f:
    np.save(f, maneuver_table)
