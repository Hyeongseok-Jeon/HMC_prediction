import glob
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import csv
# matplotlib.use('TkAgg')
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

with open(cur_path + 'map/' + data_id + '/template/link_set_new.json') as json_file:
    links = json.load(json_file)

with open(cur_path + 'map/' + data_id + '/template/node_set_new.json') as json_file:
    nodes = json.load(json_file)

nodes_csv = []
with open(cur_path + 'map/' + data_id + '/csv/node_set.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row_mod = row[0].split(',')
        node_id = row_mod[0]
        if node_id == 'id':
            pass
        else:
            x_pos = float(row_mod[1])
            y_pos = float(row_mod[2])
            nodes_csv.append([node_id, x_pos, y_pos])

links_csv = []
link_bag = glob.glob(cur_path + 'map/' + data_id + '/csv/*_link.csv')
for i in range(len(link_bag)):
    link_id = os.path.basename(link_bag[i])[:-9]
    points = []
    with open(link_bag[i], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row_mod = row[0].split(',')
            x = row_mod[3]
            y = row_mod[4]
            if (x == 'x') | (x=='"x"'):
                pass
            else:
                points.append([x,y])
    links_csv.append([link_id, points])


if len(nodes) > len(nodes_csv):
    nodes = nodes[:len(nodes_csv)]
else:
    while len(nodes) < len(nodes_csv):
        nodes.append(nodes[-1])

for i in range(len(nodes_csv)):
    idx = nodes_csv[i][0][1:-1]
    node_type = 1
    junction = []
    point = [nodes_csv[i][1], nodes_csv[i][2], 0]
    on_stop_line = False
    nodes[i]['idx'] = idx
    nodes[i]['node_type'] = node_type
    nodes[i]['junction'] = junction
    nodes[i]['point'] = point
    nodes[i]['on_stop_line'] = on_stop_line

if len(links) > len(links_csv):
    links = nodes[:len(links_csv)]
else:
    while len(links) < len(links_csv):
        links.append(links[-1].copy())

for i in range(len(links_csv)):
    idx = links_csv[i][0]
    idx_int = int(idx[-2:])
    if idx_int > 14:
        if idx_int > 17:
            idx_int = 0
        else:
            idx_int = -1
    point = links_csv[i][1]
    from_node_idx = [nodes[j]['idx'] for j in range(len(nodes)) if (nodes[j]['point'][0] == float(point[0][0])) & (nodes[j]['point'][1] == float(point[0][1]))][0]
    to_node_idx = [nodes[j]['idx'] for j in range(len(nodes)) if (nodes[j]['point'][0] == float(point[-1][0])) & (nodes[j]['point'][1] == float(point[-1][1]))][0]
    points = [[float(point[j][0]), float(point[j][1])] for j in range(len(point))]

    links[i]['idx'] = idx
    links[i]['idx_int'] = idx_int
    links[i]['from_node_idx'] = from_node_idx
    links[i]['to_node_idx'] = to_node_idx
    links[i]['points'] = points

with open(cur_path + 'map/' + data_id + '/link_set.json', "w") as json_file:
    json.dump(links, json_file)

with open(cur_path + 'map/' + data_id + '/node_set.json', "w") as json_file:
    json.dump(nodes, json_file)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(len(links)):
    x = np.asarray(links[i]['points'])[:,0]
    y = np.asarray(links[i]['points'])[:,1]
    id = links[i]['idx']
    ax.plot(x,y, zorder=0)
    ax.text((x[0]+x[-1])/2, (y[0]+y[-1])/2, id, size='xx-small')

for i in range(len(nodes)):
    x = nodes[i]['point'][0]
    y = nodes[i]['point'][1]
    ax.scatter(x,y,c='k', zorder=10)
plt.show()

maneuver_table = np.zeros([15,15])
maneuver_table[1,11] = 1
maneuver_table[1,12] = 1
maneuver_table[2,11] = 1
maneuver_table[2,12] = 1
maneuver_table[3,11] = 1
maneuver_table[3,12] = 1
maneuver_table[1,6] = 2
maneuver_table[1,7] = 2
maneuver_table[2,6] = 2
maneuver_table[2,7] = 2
maneuver_table[3,6] = 2
maneuver_table[3,7] = 2
maneuver_table[1,4] = 3
maneuver_table[1,5] = 3
maneuver_table[2,4] = 3
maneuver_table[2,5] = 3
maneuver_table[3,4] = 3
maneuver_table[3,5] = 3

maneuver_table[4,13] = 1
maneuver_table[4,14] = 1
maneuver_table[5,13] = 1
maneuver_table[5,14] = 1
maneuver_table[4,11] = 2
maneuver_table[4,12] = 2
maneuver_table[5,11] = 2
maneuver_table[5,12] = 2
maneuver_table[4,6] = 3
maneuver_table[4,7] = 3
maneuver_table[5,6] = 3
maneuver_table[5,7] = 3

maneuver_table[8,4] = 1
maneuver_table[8,5] = 1
maneuver_table[9,4] = 1
maneuver_table[9,5] = 1
maneuver_table[10,4] = 1
maneuver_table[10,5] = 1
maneuver_table[8,13] = 2
maneuver_table[8,14] = 2
maneuver_table[9,13] = 2
maneuver_table[9,14] = 2
maneuver_table[10,13] = 2
maneuver_table[10,14] = 2
maneuver_table[8,11] = 3
maneuver_table[8,12] = 3
maneuver_table[9,11] = 3
maneuver_table[9,12] = 3
maneuver_table[10,11] = 3
maneuver_table[10,12] = 3

maneuver_table[11,7] = 1
maneuver_table[11,8] = 1
maneuver_table[12,7] = 1
maneuver_table[12,8] = 1
maneuver_table[11,4] = 2
maneuver_table[11,5] = 2
maneuver_table[12,4] = 2
maneuver_table[12,5] = 2
maneuver_table[11,13] = 3
maneuver_table[11,14] = 3
maneuver_table[12,13] = 3
maneuver_table[12,14] = 3


with open(cur_path + 'map/' + data_id + '/maneuver_table.npy', "wb") as f:
    np.save(f, maneuver_table)
