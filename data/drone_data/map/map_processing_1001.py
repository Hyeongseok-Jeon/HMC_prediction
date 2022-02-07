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

with open(cur_path + 'map/' + data_id + '/link_set_new.json') as json_file:
    links = json.load(json_file)

with open(cur_path + 'map/' + data_id + '/node_set_new.json') as json_file:
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
            if x == 'x':
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
        links.append(links[-1])

for i in range(len(links_csv)):
    idx = links_csv[i][0][1:-1]
    node_type = 1
    junction = []
    point = [nodes_csv[i][1], nodes_csv[i][2], 0]
    on_stop_line = False
    nodes[i]['idx'] = idx
    nodes[i]['node_type'] = node_type
    nodes[i]['junction'] = junction
    nodes[i]['point'] = point
    nodes[i]['on_stop_line'] = on_stop_line

with open(cur_path + 'map/' + data_id + '/link_set_mod.json', "w") as json_file:
    json.dump(links, json_file)

maneuver_table = np.zeros([branch_cnt,branch_cnt])
maneuver_table[1,7] = 2
maneuver_table[1,8] = 2
maneuver_table[1,12] = 1
maneuver_table[1,13] = 1
maneuver_table[4,16] = 1
maneuver_table[4,15] = 1
maneuver_table[4,14] = 1
maneuver_table[5,16] = 1
maneuver_table[5,15] = 1
maneuver_table[5,14] = 1
maneuver_table[6,12] = 2
maneuver_table[6,13] = 2
maneuver_table[9,3] = 1
maneuver_table[9,2] = 1
maneuver_table[9,16] = 2
maneuver_table[10,3] = 1
maneuver_table[10,2] = 1
maneuver_table[10,16] = 2
maneuver_table[10,15] = 2
maneuver_table[10,14] = 2
maneuver_table[11,16] = 2
maneuver_table[11,15] = 2
maneuver_table[11,14] = 2
maneuver_table[11,12] = 3
maneuver_table[13,8] = 1
maneuver_table[13,7] = 1
maneuver_table[13,14] = 3
maneuver_table[13,15] = 3
maneuver_table[13,16] = 3
with open(cur_path + 'map/' + data_id + '/maneuver_table.npy', "wb") as f:
    np.save(f, maneuver_table)
