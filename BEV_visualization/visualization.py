import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import glob
import os

from data.drone_data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config_enc import config
from model.representation_learning.Net_enc import BackBone
import torch
import warnings
import time
import numpy as np
from openTSNE import TSNE
import imageio
import scipy.stats as st
import csv
from scipy.optimize import minimize


def func(x):
    trans_x = x[0]
    trans_y = x[1]
    rot = x[2]

    theta_1 = np.rad2deg(np.arctan2(landmark1[0][1], landmark1[0][0]))
    theta_2 = np.rad2deg(np.arctan2(landmark2[0][1], landmark2[0][0]))
    theta_3 = np.rad2deg(np.arctan2(landmark3[0][1], landmark3[0][0]))
    theta_4 = np.rad2deg(np.arctan2(landmark4[0][1], landmark4[0][0]))

    x_1 = trans_x + np.sqrt(landmark1[0][0] ** 2 + landmark1[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_1))
    y_1 = trans_y + np.sqrt(landmark1[0][0] ** 2 + landmark1[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_1))

    x_2 = trans_x + np.sqrt(landmark2[0][0] ** 2 + landmark2[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_2))
    y_2 = trans_y + np.sqrt(landmark2[0][0] ** 2 + landmark2[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_2))

    x_3 = trans_x + np.sqrt(landmark3[0][0] ** 2 + landmark3[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_3))
    y_3 = trans_y + np.sqrt(landmark3[0][0] ** 2 + landmark3[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_3))

    x_4 = trans_x + np.sqrt(landmark4[0][0] ** 2 + landmark4[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_4))
    y_4 = trans_y + np.sqrt(landmark4[0][0] ** 2 + landmark4[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_4))

    landmark1_trans = np.asarray([[x_1, y_1]])
    landmark2_trans = np.asarray([[x_2, y_2]])
    landmark3_trans = np.asarray([[x_3, y_3]])
    landmark4_trans = np.asarray([[x_4, y_4]])

    return np.linalg.norm(landmark1_GT - landmark1_trans) + \
           np.linalg.norm(landmark2_GT - landmark2_trans) + \
           np.linalg.norm(landmark3_GT - landmark3_trans) + \
           np.linalg.norm(landmark4_GT - landmark4_trans)


def data_load(root, data_id):
    # background = mpimg.imread('../data/background/' + str(data_id) + '.jpg')
    landmark = np.genfromtxt(root + 'raw/landmark/' + str(data_id) + '_landmarks.csv', skip_header=1, delimiter=',',
                             dtype=int)
    landmark[:, 1] = landmark[:, 1] % 10000
    recordingMeta = np.genfromtxt(root + 'raw/recordingMeta/' + str(data_id) + '_recordingMeta.csv', skip_header=1,
                                  delimiter=',')
    recordingMeta[3] = recordingMeta[3] % 10000
    tracks = np.genfromtxt(root + 'raw/tracks/' + str(data_id) + '_tracks.csv', skip_header=1, delimiter=',')
    tracksMeta = np.genfromtxt(root + 'raw/tracksMeta/' + str(data_id) + '_trackMeta.csv', skip_header=1, delimiter=',')
    tracksMeta = np.delete(tracksMeta, -1, -1)
    tracksClass = []
    with open(root + 'raw/tracksMeta/' + str(data_id) + '_trackMeta.csv', "r") as tmp_file:
        csvReader = csv.reader(tmp_file)
        header = next(csvReader)
        class_index = header.index("class")
        for row in csvReader:
            class_tmp = row[class_index]
            tracksClass.append(class_tmp)

    return landmark, recordingMeta, tracks, tracksMeta, tracksClass


def coordinate_conversion(scale, tracks, landmark, recordingMeta, origin_GT):
    global center_GT
    global landmark1_GT
    global landmark2_GT
    global landmark3_GT
    global landmark4_GT
    global landmark1
    global landmark2
    global landmark3
    global landmark4

    meter_per_pixel = scale * recordingMeta[15]
    new_tracks = np.zeros_like(tracks)
    new_tracks[:] = tracks[:]
    landmark1_GT = np.asarray([origin_GT[0]])
    landmark2_GT = np.asarray([origin_GT[1]])
    landmark3_GT = np.asarray([origin_GT[2]])
    landmark4_GT = np.asarray([origin_GT[3]])
    center_GT = [(landmark1_GT[0, 0] + landmark2_GT[0, 0] + landmark3_GT[0, 0] + landmark4_GT[0, 0]) / 4,
                 (landmark1_GT[0, 1] + landmark2_GT[0, 1] + landmark3_GT[0, 1] + landmark4_GT[0, 1]) / 4]

    for i in range(len(landmark)):
        if i + 1 == len(landmark):
            print(str(int(10000 * (i + 1) / len(landmark)) / 100) + ' % is completed')
        else:
            print(str(int(10000 * (i + 1) / len(landmark)) / 100) + ' % is completed', end='\r')
        cur_frame = landmark[i, 1]
        landmark1 = np.asarray([[landmark[i, 2] * meter_per_pixel, -landmark[i, 3] * meter_per_pixel]])
        landmark2 = np.asarray([[landmark[i, 4] * meter_per_pixel, -landmark[i, 5] * meter_per_pixel]])
        landmark3 = np.asarray([[landmark[i, 6] * meter_per_pixel, -landmark[i, 7] * meter_per_pixel]])
        landmark4 = np.asarray([[landmark[i, 8] * meter_per_pixel, -landmark[i, 9] * meter_per_pixel]])
        center = [(landmark1[0, 0] + landmark2[0, 0] + landmark3[0, 0] + landmark4[0, 0]) / 4,
                  (landmark1[0, 1] + landmark2[0, 1] + landmark3[0, 1] + landmark4[0, 1]) / 4]

        res = minimize(func, [center_GT[0] - center[0], center_GT[1] - center[1], 0], method='Nelder-Mead', tol=1e-10)

        trans_x = res.x[0]
        trans_y = res.x[1]
        rot = res.x[2]

        veh_list = np.where(tracks[:, 2] == cur_frame)[0]
        for j in range(len(veh_list)):
            cur_pos = scale * np.asarray([tracks[veh_list[j], 4:6]])
            theta_1 = np.rad2deg(np.arctan2(cur_pos[0][1], cur_pos[0][0]))

            x_1 = trans_x + np.sqrt(cur_pos[0][0] ** 2 + cur_pos[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_1))
            y_1 = trans_y + np.sqrt(cur_pos[0][0] ** 2 + cur_pos[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_1))

            new_tracks[veh_list[j], 4:6] = np.asarray([x_1, y_1])
            new_tracks[veh_list[j], 6] = new_tracks[veh_list[j], 6] + rot - 90

    return new_tracks


def get_veh_in_frame(tracks, frame, px2m, hist_time):
    vehs = dict()
    vehs['id'] = list(tracks[(tracks[:, 2] == frame) & (tracks[:, 7] > 0), 1])
    vehs['x'] = np.asarray(list(tracks[(tracks[:, 2] == frame) & (tracks[:, 7] > 0), 4])) / px2m
    vehs['y'] = -np.asarray(list(tracks[(tracks[:, 2] == frame) & (tracks[:, 7] > 0), 5])) / px2m
    vehs['heading'] = -np.asarray(list(tracks[(tracks[:, 2] == frame) & (tracks[:, 7] > 0), 6]))
    vehs['width'] = np.asarray(list(tracks[(tracks[:, 2] == frame) & (tracks[:, 7] > 0), 7])) / px2m
    vehs['length'] = np.asarray(list(tracks[(tracks[:, 2] == frame) & (tracks[:, 7] > 0), 8])) / px2m

    hist_x = []
    hist_y = []
    if frame < 30 * hist_time:
        for i in range(len(vehs['id'])):
            hist_x.append(np.asarray(list(tracks[(tracks[:, 1] == vehs['id'][i]) & (tracks[:, 2] <= frame), 4])) / px2m)
            hist_y.append(-np.asarray(list(tracks[(tracks[:, 1] == vehs['id'][i]) & (tracks[:, 2] <= frame), 5])) / px2m)
    else:
        for i in range(len(vehs['id'])):
            hist_x.append(np.asarray(list(tracks[(tracks[:, 1] == vehs['id'][i]) & (tracks[:, 2] <= frame) & (tracks[:, 2] > frame - 30 * hist_time), 4])) / px2m)
            hist_y.append(-np.asarray(list(tracks[(tracks[:, 1] == vehs['id'][i]) & (tracks[:, 2] <= frame) & (tracks[:, 2] > frame - 30 * hist_time), 5])) / px2m)
    vehs['hist_x'] = hist_x
    vehs['hist_y'] = hist_y

    hist_x_tot = []
    hist_y_tot = []
    for i in range(len(vehs['id'])):
        hist_x_tot.append(np.asarray(list(tracks[(tracks[:, 1] == vehs['id'][i]) & (tracks[:, 2] <= frame), 4])) / px2m)
        hist_y_tot.append(-np.asarray(list(tracks[(tracks[:, 1] == vehs['id'][i]) & (tracks[:, 2] <= frame), 5])) / px2m)
    vehs['hist_x_tot'] = hist_x_tot
    vehs['hist_y_tot'] = hist_y_tot

    return vehs


def get_veh_polygon(vehs):
    vehicle_x = []
    vehicle_y = []
    for j in range(len(vehs['id'])):
        x = vehs['x'][j]
        y = vehs['y'][j]
        heading = vehs['heading'][j]
        width = vehs['width'][j]
        length = vehs['length'][j]

        polygon = [[x + 0.5 * length * np.cos(np.deg2rad(heading)) + 0.5 * width * np.sin(np.deg2rad(heading)),
                    y + 0.5 * length * np.sin(np.deg2rad(heading)) - 0.5 * width * np.cos(np.deg2rad(heading))],
                   [x + 0.5 * length * np.cos(np.deg2rad(heading)) - 0.5 * width * np.sin(np.deg2rad(heading)),
                    y + 0.5 * length * np.sin(np.deg2rad(heading)) + 0.5 * width * np.cos(np.deg2rad(heading))],
                   [x - 0.5 * length * np.cos(np.deg2rad(heading)) - 0.5 * width * np.sin(np.deg2rad(heading)),
                    y - 0.5 * length * np.sin(np.deg2rad(heading)) + 0.5 * width * np.cos(np.deg2rad(heading))],
                   [x - 0.5 * length * np.cos(np.deg2rad(heading)) + 0.5 * width * np.sin(np.deg2rad(heading)),
                    y - 0.5 * length * np.sin(np.deg2rad(heading)) - 0.5 * width * np.cos(np.deg2rad(heading))]]
        polygon.append(polygon[0])
        xs, ys = zip(*polygon)
        vehicle_x.append(xs)
        vehicle_y.append(ys)

    return vehicle_x, vehicle_y


def get_bags(dataloader, model):
    for i, data in enumerate(dataloader_train):
        trajectory, traj_length, conversion, maneuvers = data
        maneuvers = maneuvers[0].numpy()
        trajectory = trajectory.float().cuda()
        with torch.no_grad():
            representation_time_bag_1 = model_2(trajectory, traj_length, mode='val')
        if i == 0:
            context_bag_train_1 = [None if repres is None else repres.cpu().detach().numpy() for repres in representation_time_bag_1]
            maneuver_bag_train_1 = [None if context_bag_train_1[i] is None else np.repeat(maneuvers, config["val_augmentation"], axis=0) for i in range(11)]
        else:
            for j in range(len(representation_time_bag_1)):
                if representation_time_bag_1[j] is None:
                    pass
                else:
                    if context_bag_train_1[j] is None:
                        context_bag_train_1[j] = representation_time_bag_1[j].cpu().detach().numpy()
                        maneuver_bag_train_1[j] = np.repeat(maneuvers, config["val_augmentation"], axis=0)
                    else:
                        context_bag_train_1[j] = np.concatenate((context_bag_train_1[j], representation_time_bag_1[j].cpu().detach().numpy()), axis=0)
                        maneuver_bag_train_1[j] = np.concatenate((maneuver_bag_train_1[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
    return [context_bag_train_1, maneuver_bag_train_1]


def get_gaussian_kde(context_bag_tot_2, maneuver_bag_tot_2):
    n_components = 2
    tsne_hist_2 = TSNE(n_components=n_components,
                       perplexity=30,
                       verbose=True)
    contexts_outlet_2 = context_bag_tot_2[-2]
    context_tsne_2 = tsne_hist_2.fit(contexts_outlet_2)

    contexts = context_bag_tot_2[9]
    context_low_dim = context_tsne_2.transform(contexts)
    context_low_dim_left_turn = context_low_dim[maneuver_bag_tot_2[9][:, 1] == 1]
    context_low_dim_go_straight = context_low_dim[maneuver_bag_tot_2[9][:, 2] == 1]
    context_low_dim_right_turn = context_low_dim[maneuver_bag_tot_2[9][:, 3] == 1]

    x = context_low_dim_left_turn[:, 0]
    y = context_low_dim_left_turn[:, 1]
    values = np.vstack([x, y])
    kernel_LT = st.gaussian_kde(values)

    x = context_low_dim_right_turn[:, 0]
    y = context_low_dim_right_turn[:, 1]
    values = np.vstack([x, y])
    kernel_RT = st.gaussian_kde(values)

    x = context_low_dim_go_straight[:, 0]
    y = context_low_dim_go_straight[:, 1]
    values = np.vstack([x, y])
    kernel_ST = st.gaussian_kde(values)

    return kernel_LT, kernel_ST, kernel_RT, context_tsne_2, [context_low_dim_left_turn, context_low_dim_go_straight, context_low_dim_right_turn]


def get_arrows(x_cen, y_cen, heading, length, maneuver):
    arr_ST_x_init = x_cen + (0.5 * length + 10) * np.cos(np.deg2rad(heading))
    arr_ST_y_init = y_cen + (0.5 * length + 10) * np.sin(np.deg2rad(heading))
    arr_ST_x_end = x_cen + (0.5 * length + 150) * np.cos(np.deg2rad(heading))
    arr_ST_y_end = y_cen + (0.5 * length + 150) * np.sin(np.deg2rad(heading))
    arr_ST = matplotlib.patches.FancyArrowPatch((arr_ST_x_init, arr_ST_y_init), (arr_ST_x_end, arr_ST_y_end),
                                                mutation_scale=20,
                                                facecolor='w',
                                                edgecolor='k',
                                                alpha=maneuver[1][0])

    arr_LT_1_x_init = x_cen + (0.5 * length + 15) * np.cos(np.deg2rad(heading)) + 30 * np.sin(np.deg2rad(heading))
    arr_LT_1_y_init = y_cen + (0.5 * length + 15) * np.sin(np.deg2rad(heading)) - 30 * np.cos(np.deg2rad(heading))
    arr_LT_1_x_end = x_cen + (0.5 * length + 70) * np.cos(np.deg2rad(heading)) + 30 * np.sin(np.deg2rad(heading))
    arr_LT_1_y_end = y_cen + (0.5 * length + 70) * np.sin(np.deg2rad(heading)) - 30 * np.cos(np.deg2rad(heading))

    points = []
    points.append([arr_LT_1_x_init + 8 * np.sin(np.deg2rad(heading)), arr_LT_1_y_init - 8 * np.cos(np.deg2rad(heading))])
    points.append([arr_LT_1_x_init - 8 * np.sin(np.deg2rad(heading)), arr_LT_1_y_init + 8 * np.cos(np.deg2rad(heading))])
    points.append([arr_LT_1_x_end - 8 * np.sin(np.deg2rad(heading)), arr_LT_1_y_end + 8 * np.cos(np.deg2rad(heading))])
    for i in range(100):
        origin = [arr_LT_1_x_end + 20 * np.sin(np.deg2rad(heading)), arr_LT_1_y_end - 20 * np.cos(np.deg2rad(heading))]
        theta = heading - 90 * (i / 99)
        x_circle = origin[0] - 28 * np.sin(np.deg2rad(theta))
        y_circle = origin[1] + 28 * np.cos(np.deg2rad(theta))
        points.append([x_circle, y_circle])
    points.append([points[-1][0] + 15 * np.sin(np.deg2rad(heading)), points[-1][1] - 15 * np.cos(np.deg2rad(heading))])
    points.append([points[-1][0] + 15 * np.sin(np.deg2rad(heading + 90)), points[-1][1] - 15 * np.cos(np.deg2rad(heading + 90))])
    points.append([points[-2][0] - 8 * np.sin(np.deg2rad(heading + 90)) + 35 * np.sin(np.deg2rad(heading)),
                   points[-2][1] + 8 * np.cos(np.deg2rad(heading + 90)) - 35 * np.cos(np.deg2rad(heading))])
    points.append([points[-3][0] - 31 * np.sin(np.deg2rad(heading + 90)), points[-3][1] + 31 * np.cos(np.deg2rad(heading + 90))])
    points.append([points[-1][0] + 15 * np.sin(np.deg2rad(heading + 90)), points[-1][1] - 15 * np.cos(np.deg2rad(heading + 90))])
    points.append([points[-1][0] - 15 * np.sin(np.deg2rad(heading)), points[-1][1] + 15 * np.cos(np.deg2rad(heading))])

    for i in range(100):
        origin = [arr_LT_1_x_end + 20 * np.sin(np.deg2rad(heading)), arr_LT_1_y_end - 20 * np.cos(np.deg2rad(heading))]
        theta = heading - 90 + 90 * (i / 99)
        x_circle = origin[0] - 12 * np.sin(np.deg2rad(theta))
        y_circle = origin[1] + 12 * np.cos(np.deg2rad(theta))
        points.append([x_circle, y_circle])
    points.append([points[0][0], points[0][1]])

    arr_LT = matplotlib.patches.Polygon(xy=np.asarray(points),
                                        facecolor='w',
                                        edgecolor='k',
                                        alpha=maneuver[0][0])

    arr_RT_1_x_init = x_cen + (0.5 * length + 15) * np.cos(np.deg2rad(heading)) - 30 * np.sin(np.deg2rad(heading))
    arr_RT_1_y_init = y_cen + (0.5 * length + 15) * np.sin(np.deg2rad(heading)) + 30 * np.cos(np.deg2rad(heading))
    arr_RT_1_x_end = x_cen + (0.5 * length + 70) * np.cos(np.deg2rad(heading)) - 30 * np.sin(np.deg2rad(heading))
    arr_RT_1_y_end = y_cen + (0.5 * length + 70) * np.sin(np.deg2rad(heading)) + 30 * np.cos(np.deg2rad(heading))
    points = []
    points.append([arr_RT_1_x_init - 8 * np.sin(np.deg2rad(heading)), arr_RT_1_y_init + 8 * np.cos(np.deg2rad(heading))])
    points.append([arr_RT_1_x_init + 8 * np.sin(np.deg2rad(heading)), arr_RT_1_y_init - 8 * np.cos(np.deg2rad(heading))])
    points.append([arr_RT_1_x_end + 8 * np.sin(np.deg2rad(heading)), arr_RT_1_y_end - 8 * np.cos(np.deg2rad(heading))])
    for i in range(100):
        origin = [arr_RT_1_x_end - 20 * np.sin(np.deg2rad(heading)), arr_RT_1_y_end + 20 * np.cos(np.deg2rad(heading))]
        theta = 180 + heading + 90 * (i / 99)
        x_circle = origin[0] - 28 * np.sin(np.deg2rad(theta))
        y_circle = origin[1] + 28 * np.cos(np.deg2rad(theta))
        points.append([x_circle, y_circle])
    points.append([points[-1][0] - 15 * np.sin(np.deg2rad(heading)), points[-1][1] + 15 * np.cos(np.deg2rad(heading))])
    points.append([points[-1][0] + 15 * np.sin(np.deg2rad(heading + 90)), points[-1][1] - 15 * np.cos(np.deg2rad(heading + 90))])
    points.append([points[-2][0] - 8 * np.sin(np.deg2rad(heading + 90)) - 35 * np.sin(np.deg2rad(heading)),
                   points[-2][1] + 8 * np.cos(np.deg2rad(heading + 90)) + 35 * np.cos(np.deg2rad(heading))])
    points.append([points[-3][0] - 31 * np.sin(np.deg2rad(heading + 90)), points[-3][1] + 31 * np.cos(np.deg2rad(heading + 90))])
    points.append([points[-1][0] + 15 * np.sin(np.deg2rad(heading + 90)), points[-1][1] - 15 * np.cos(np.deg2rad(heading + 90))])
    points.append([points[-1][0] + 15 * np.sin(np.deg2rad(heading)), points[-1][1] - 15 * np.cos(np.deg2rad(heading))])

    for i in range(100):
        origin = [arr_RT_1_x_end - 20 * np.sin(np.deg2rad(heading)), arr_RT_1_y_end + 20 * np.cos(np.deg2rad(heading))]
        theta = heading + 270 - 90 * (i / 99)
        x_circle = origin[0] - 12 * np.sin(np.deg2rad(theta))
        y_circle = origin[1] + 12 * np.cos(np.deg2rad(theta))
        points.append([x_circle, y_circle])
    points.append([points[0][0], points[0][1]])

    arr_RT = matplotlib.patches.Polygon(xy=np.asarray(points),
                                        facecolor='w',
                                        edgecolor='k',
                                        alpha=maneuver[2][0])

    return arr_LT, arr_ST, arr_RT


scene_id = '1001'
save_path = 'BEV_visualization\\video_results\\' + scene_id
os.makedirs(save_path, exist_ok=True)
landmark, recordingMeta, tracks, tracksMeta, tracksClass = data_load('data/drone_data/', scene_id)
origin_GT = []
with open('data/drone_data/' + 'map/' + '1001' + '/csv/LandMark.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row_mod = row[0].split(',')
        if row_mod[0] == 'id':
            pass
        else:
            origin_GT.append([float(row_mod[1]), float(row_mod[2])])
new_tracks = coordinate_conversion(1, tracks, landmark, recordingMeta, origin_GT)
veh_idx = tracksMeta[(tracksMeta[:, 6] > 0) & (tracksMeta[:, 4] > 0), 1]

img_path = 'BEV_visualization\\' + scene_id + '\\images\\' + scene_id + '_'

px2m = recordingMeta[15]
frame_list = list(tracks[:, 2])
frame_list = list(dict.fromkeys(frame_list))
frame_list = ['0' * (4 - len(str(int(frame_list[i]) + 1))) + str(int(frame_list[i]) + 1) for i in range(len(frame_list))]
hist_time = 2

GPU_NUM = config["GPU_id"]
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# file_id_1 = 'maneuver_prediction-2022-03-10_04_02_34'
# ckpt_dir_1 = config['ckpt_dir'] + file_id_1
# weight_1 = 'model_140.pt'

file_id_2 = 'maneuver_prediction-2022-03-17_07_07_52'
ckpt_dir_2 = config['ckpt_dir'] + file_id_2
weight_2 = 'model_50.pt'
warnings.filterwarnings("ignore", category=UserWarning)

config["splicing_num"] = 1
config["occlusion_rate"] = 0
config["batch_size"] = 1
config["LC_multiple"] = 1
config["LK_multiple"] = 0.1

dataset_train = pred_loader_1(config, 'train', mode='val')
dataset_val = pred_loader_1(config, 'val', mode='val')

dataloader_train = DataLoader(dataset_train,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)

# model_1 = BackBone(config).cuda()
# weights = torch.load(ckpt_dir_1 + '/' + weight_1)
# model_1.load_state_dict(weights['model_state_dict'])
model_2 = BackBone(config).cuda()
weights = torch.load(ckpt_dir_2 + '/' + weight_2)
model_2.load_state_dict(weights['model_state_dict'])

correct_num_tot = 0
full_length_num_tot = 0
loss_tot = 0
loss_calc_num_tot = 0
epoch_time = time.time()

# [context_bag_train_1, maneuver_bag_train_1] = get_bags(dataloader_train, model_1)
# [context_bag_val_1, maneuver_bag_val_1] = get_bags(dataloader_val, model_1)
[context_bag_train_2, maneuver_bag_train_2] = get_bags(dataloader_train, model_2)
[context_bag_val_2, maneuver_bag_val_2] = get_bags(dataloader_val, model_2)

# context_bag_tot_1 = [np.concatenate((context_bag_train_1[i], context_bag_val_1[i]))for i in range(len(context_bag_train_1))]
context_bag_tot_2 = [np.concatenate((context_bag_train_2[i], context_bag_val_2[i])) for i in range(len(context_bag_train_2))]
# maneuver_bag_tot_1 = [np.concatenate((maneuver_bag_train_1[i], maneuver_bag_val_1[i]))for i in range(len(maneuver_bag_train_1))]
maneuver_bag_tot_2 = [np.concatenate((maneuver_bag_train_2[i], maneuver_bag_val_2[i])) for i in range(len(maneuver_bag_train_2))]

kernel_LT, kernel_ST, kernel_RT, tsne, context_low_dims = get_gaussian_kde(context_bag_tot_2, maneuver_bag_tot_2)

config["LK_multiple"] = 1
dataset_train = pred_loader_1(config, 'train', mode='val')
dataset_val = pred_loader_1(config, 'val', mode='val')

dataloader_train = DataLoader(dataset_train,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)

plt.figure()
colors = ['g', 'b', 'c']
for i in range(3):
    plt.figure()

    x = context_low_dims[i][:, 0]
    y = context_low_dims[i][:, 1]
    xx, yy = np.mgrid[-60:60:100j, -60:80:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    if i == 0:
        val = kernel(positions).T
    else:
        val = val + kernel(positions).T

    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = plt.contourf(xx, yy, f, cmap='coolwarm', label='Left Turn')

for i in range(3):
    plt.scatter(context_low_dims[i][:, 0], context_low_dims[i][:, 1], s=2, c=colors[i], label='Left Turn', alpha=0.2)

plt.figure(figsize=(8.1 * 2, 5.4 * 2))
for i in range(len(frame_list)):
    print(int(10000 * i / len(frame_list)) / 100)
    frame = int(frame_list[i]) - 1
    img = mpimg.imread(img_path + frame_list[i] + '.jpg')
    vehs = get_veh_in_frame(tracks, frame, px2m, hist_time)
    x, y = get_veh_polygon(vehs)

    maneuver_kde = []
    for j in range(len(vehs['id'])):
        print(vehs['id'][j])
        traj_network = new_tracks[new_tracks[:, 1] == vehs['id'][j], 4:6]
        heading_network = new_tracks[new_tracks[:, 1] == vehs['id'][j], 6:7]
        traj = np.concatenate((traj_network, heading_network), axis=1)
        traj = traj[:vehs['hist_x_tot'][j].shape[0]]
        trajectory = torch.from_numpy(traj).unsqueeze(0).cuda().float()
        traj_length = [trajectory.shape[1]]
        if traj_length[0] > 4:
            representation_time_bag_1 = model_2(trajectory, traj_length, mode='val', vis=True)
            low_dim = tsne.transform(representation_time_bag_1[-1])
            LT_pdf = kernel_LT.pdf(low_dim)
            ST_pdf = kernel_ST.pdf(low_dim)
            RT_pdf = kernel_RT.pdf(low_dim)
            pdfs = [LT_pdf, ST_pdf, RT_pdf]
            maneuver_kde.append(pdfs / sum(pdfs))
        else:
            maneuver_kde.append([None, None, None])

    arr_LTs, arr_STs, arr_RTs = [], [], []
    for j in range(len(vehs['id'])):
        x_cen = vehs['x'][j]
        y_cen = vehs['y'][j]
        heading = vehs['heading'][j]
        length = vehs['length'][j]
        maneuver = maneuver_kde[j]
        arr_LT, arr_ST, arr_RT = get_arrows(x_cen, y_cen, heading, length, maneuver)
        arr_LTs.append(arr_LT)
        arr_STs.append(arr_ST)
        arr_RTs.append(arr_RT)
    #
    # arr_LT = matplotlib.patches.FancyArrowPatch((arr_LT_1_x_init, arr_LT_1_y_init), (arr_LT_1_x_end, arr_LT_1_y_end),
    #                                             mutation_scale=20,
    #                                             facecolor='w',
    #                                             edgecolor='k',
    #                                             alpha=1)


    plt.imshow(img)
    for j in range(len(x)):
        plt.plot(x[j], y[j], 'k', alpha=0.5)
        plt.fill(x[j], y[j], 'c', alpha=0.5)
    for j in range(len(vehs['id'])):
        hist_traj_x = vehs['hist_x'][j]
        hist_traj_y = vehs['hist_y'][j]
        for k in range(len(hist_traj_x) - 1):
            plt.plot(hist_traj_x[k:k + 2], hist_traj_y[k:k + 2], 'r', alpha=k / (len(hist_traj_x) - 1))
    for j in range(len(vehs['id'])):
        plt.gca().add_patch(arr_LTs[j])
        plt.gca().add_patch(arr_STs[j])
        plt.gca().add_patch(arr_RTs[j])

    plt.axis('off')
    plt.ylim([2160, 0])
    plt.xlim([0, 3840])

    fig_name = save_path + '\\' + frame_list[i] + '.png'
    plt.savefig(fig_name, dpi=200)
    plt.clf()

import cv2
import numpy as np

# choose codec according to format needed
img_list = glob.glob('BEV_visualization\\video_results\\' + scene_id + '\\*.png')

img_array = []
for filename in img_list:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('BEV_visualization\\video_results\\' + scene_id + '\\result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
