import glob
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
matplotlib.use('TkAgg')
# file_path = os.path.dirname(os.path.abspath(__file__))
# cur_path = os.path.dirname(file_path)+'/'
cur_path = os.getcwd() + '/data/drone_data/'
data_bag = glob.glob(cur_path + '/raw/tracks/*.csv')
sys.path.append(cur_path)
from utils import point_regen

data_idx = 17
data_id = os.path.basename(data_bag[data_idx]).split("_")[0]

with open(cur_path + 'map/' + data_id + '/link_set_new.json') as json_file:
    links = json.load(json_file)

with open(cur_path + 'map/' + data_id + '/node_set_new.json') as json_file:
    nodes = json.load(json_file)

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
    if i == 7:
        pt_list = pt_list[127:]
    elif i == 8:
        pt_list = pt_list[131:]
    elif i == 9:
        pt_list = pt_list[132:]

    links[i]['points'] = pt_list.tolist()
    plt.scatter(np.array(pt_list)[:, 0], np.array(pt_list)[:, 1])

    init = 0

plt.axis('equal')
plt.show()

with open(cur_path + 'map/' + data_id + '/link_set_mod.json', "w") as json_file:
    json.dump(links, json_file)


