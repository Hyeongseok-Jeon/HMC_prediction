import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import sys
from data.drone_data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config_enc import config
from model.representation_learning.Net_enc import BackBone
from model.maneuver_classification.Net_dec import Downstream
from model.maneuver_classification.config_dec import config as config_dec
import torch
import warnings
import time
import numpy as np
from openTSNE import TSNE
import scipy.stats as st
import csv
from scipy.optimize import minimize
import shutil
import json
from scipy import interpolate
import cv2
from PIL import Image, ImageDraw
import random

cur_path = os.getcwd() + '/data/drone_data/'


def B_spline(waypoints):
    x = []
    y = []

    for i in range(waypoints.shape[0]):
        x.append(waypoints[i, 0])
        y.append(waypoints[i, 1])

    tck, *rest = interpolate.splprep([x, y], s=1000000)
    u = np.linspace(0, 1, num=50)
    smooth = interpolate.splev(u, tck)
    smooth_np = np.asarray(smooth)
    return smooth_np.T


def seg_check(x, y, traj):
    center = [1858, 1096]
    seg1 = [[2142, 598], [2473, 1063]]
    seg2 = [[1467, 639], [1697, 491]]
    seg3 = [[1273, 1862], [965, 1401]]
    seg4 = [[2384, 1491], [2133, 1676]]
    line1_inter = ((seg1[1][1] - seg1[0][1]) / (seg1[1][0] - seg1[0][0])) * (center[0] - seg1[0][0]) + seg1[0][1] - center[1]
    line2_inter = ((seg2[1][1] - seg2[0][1]) / (seg2[1][0] - seg2[0][0])) * (center[0] - seg2[0][0]) + seg2[0][1] - center[1]
    line3_inter = ((seg3[1][1] - seg3[0][1]) / (seg3[1][0] - seg3[0][0])) * (center[0] - seg3[0][0]) + seg3[0][1] - center[1]
    line4_inter = ((seg4[1][1] - seg4[0][1]) / (seg4[1][0] - seg4[0][0])) * (center[0] - seg4[0][0]) + seg4[0][1] - center[1]

    line1_eval = ((seg1[1][1] - seg1[0][1]) / (seg1[1][0] - seg1[0][0])) * (x - seg1[0][0]) + seg1[0][1] - y
    line2_eval = ((seg2[1][1] - seg2[0][1]) / (seg2[1][0] - seg2[0][0])) * (x - seg2[0][0]) + seg2[0][1] - y
    line3_eval = ((seg3[1][1] - seg3[0][1]) / (seg3[1][0] - seg3[0][0])) * (x - seg3[0][0]) + seg3[0][1] - y
    line4_eval = ((seg4[1][1] - seg4[0][1]) / (seg4[1][0] - seg4[0][0])) * (x - seg4[0][0]) + seg4[0][1] - y

    if line1_inter * line1_eval < 0:
        if traj[-1] > traj[0]:
            return False
        else:
            return True
    if line2_inter * line2_eval < 0:
        if traj[-1] < traj[0]:
            return False
        else:
            return True
    if line3_inter * line3_eval < 0:
        if traj[-1] < traj[0]:
            return False
        else:
            return True
    if line4_inter * line4_eval < 0:
        if traj[-1] > traj[0]:
            return False
        else:
            return True
    return True


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


def coordinate_conversion(scale, landmark, meter_per_pixel, origin_GT):
    global center_GT
    global landmark1_GT
    global landmark2_GT
    global landmark3_GT
    global landmark4_GT
    global landmark1
    global landmark2
    global landmark3
    global landmark4

    landmark1_GT = np.asarray([origin_GT[0]])
    landmark2_GT = np.asarray([origin_GT[1]])
    landmark3_GT = np.asarray([origin_GT[2]])
    landmark4_GT = np.asarray([origin_GT[3]])
    center_GT = [(landmark1_GT[0, 0] + landmark2_GT[0, 0] + landmark3_GT[0, 0] + landmark4_GT[0, 0]) / 4,
                 (landmark1_GT[0, 1] + landmark2_GT[0, 1] + landmark3_GT[0, 1] + landmark4_GT[0, 1]) / 4]

    landmark1 = np.asarray([[landmark[2] * meter_per_pixel, -landmark[3] * meter_per_pixel]])
    landmark2 = np.asarray([[landmark[4] * meter_per_pixel, -landmark[5] * meter_per_pixel]])
    landmark3 = np.asarray([[landmark[6] * meter_per_pixel, -landmark[7] * meter_per_pixel]])
    landmark4 = np.asarray([[landmark[8] * meter_per_pixel, -landmark[9] * meter_per_pixel]])
    center = [(landmark1[0, 0] + landmark2[0, 0] + landmark3[0, 0] + landmark4[0, 0]) / 4,
              (landmark1[0, 1] + landmark2[0, 1] + landmark3[0, 1] + landmark4[0, 1]) / 4]

    res = minimize(func, [center_GT[0] - center[0], center_GT[1] - center[1], 0], method='Nelder-Mead', tol=1e-10)

    return res


def angle_diff_max(distance, vehs, jj, direction, cur_pos=None, heading_c=None):
    if cur_pos == None:
        x = vehs['x'][jj]
        y = vehs['y'][jj]
        heading = -vehs['heading'][jj]
    else:
        x = cur_pos[0]
        y = cur_pos[1]
        heading = heading_c

    wheel_base = 0.6472491909385114 * vehs['length'][jj]
    if direction == 'ccw':
        max_steer = 30
        rear_dist = wheel_base / np.tan(np.deg2rad(max_steer))
        mid_dist = np.sqrt(rear_dist ** 2 + (wheel_base * 0.5) ** 2)
        angle = 180 - np.rad2deg(np.arccos(0.5 * wheel_base / mid_dist)) + heading
        origin_x = x + mid_dist * np.cos(np.deg2rad(angle))
        origin_y = y - mid_dist * np.sin(np.deg2rad(angle))
        init_angle = angle - 180
        for i in range(0, 3600, 1):
            angle = init_angle + 0.1 * i
            x_arc = origin_x + mid_dist * np.cos(np.deg2rad(angle))
            y_arc = origin_y - mid_dist * np.sin(np.deg2rad(angle))
            disp = np.linalg.norm([y_arc - y, x_arc - x])
            if disp > distance:
                ang = np.rad2deg(np.arctan2(y - y_arc, x_arc - x))
                if ang < 0:
                    ang = ang + 360
                return ang - heading, [x_arc, y_arc]
    else:
        max_steer = -30
        rear_dist = -wheel_base / np.tan(np.deg2rad(max_steer))
        mid_dist = np.sqrt(rear_dist ** 2 + (wheel_base * 0.5) ** 2)
        angle = -(180 - np.rad2deg(np.arccos(0.5 * wheel_base / mid_dist)) - heading)
        origin_x = x + mid_dist * np.cos(np.deg2rad(angle))
        origin_y = y - mid_dist * np.sin(np.deg2rad(angle))
        init_angle = angle - 180
        for i in range(0, 3600, 1):
            angle = init_angle - 0.1 * i
            x_arc = origin_x + mid_dist * np.cos(np.deg2rad(angle))
            y_arc = origin_y - mid_dist * np.sin(np.deg2rad(angle))
            disp = np.linalg.norm([y_arc - y, x_arc - x])
            if disp > distance:
                ang = np.rad2deg(np.arctan2(y - y_arc, x_arc - x))
                if ang < 0:
                    ang = ang + 360
                return ang - heading, [x_arc, y_arc]


def kinematic_update(cur_pos, LT_path_mod, threshold, vehs, jj, direction, LT_heading_mod):
    LT_path_new = [cur_pos]
    angle_diff = [None, None]
    for i in range(len(LT_path_mod)):
        if i == 0:
            distance = 60
            _, new_pos = angle_diff_max(distance, vehs, jj, direction)
            LT_path_new.append(new_pos)
        else:
            distance = 60
            if i == 1:
                heading = np.rad2deg(np.arctan2(-(LT_path_new[-1][1] - cur_pos[1]), LT_path_new[-1][0] - cur_pos[0]))
                if heading < 0:
                    heading = heading + 360
            else:
                heading = np.rad2deg(np.arctan2(-(LT_path_new[-1][1] - LT_path_new[-2][1]), LT_path_new[-1][0] - LT_path_new[-2][0]))
                if heading < 0:
                    heading = heading + 360
            _, new_pos = angle_diff_max(distance, vehs, jj, direction, cur_pos=LT_path_new[-1], heading_c=heading)
            LT_path_new.append(new_pos)
        heading = np.rad2deg(np.arctan2(-(LT_path_new[-1][1] - LT_path_new[-2][1]), LT_path_new[-1][0] - LT_path_new[-2][0]))
        nearest_point = np.argmin(np.linalg.norm(LT_path_mod - new_pos, axis=1))
        nearest_heading = LT_heading_mod[nearest_point]
        angle_diff[0] = angle_diff[1]
        angle_diff[1] = nearest_heading - heading
        if np.abs(nearest_heading - heading) < 10:
            break
        if angle_diff[0] != None and angle_diff[1] != None:
            if angle_diff[0] * angle_diff[1] < 0:
                break
    return np.asarray(LT_path_new)[1:, :], LT_path_mod[nearest_point:]


def path_mod_LT(LT_points, cur_pos_sub, cur_idx, LT_path, LT_heading_mod, turn_start_idx, merge_idx):
    disp_cen = LT_points[cur_idx, :] - LT_points[0, :]
    angle_cen = np.rad2deg(np.arctan2(-disp_cen[1], disp_cen[0]))
    if angle_cen < 0:
        angle_cen = angle_cen + 360
    disp_cur = cur_pos_sub - LT_points[0, :]
    angle_cur = np.rad2deg(np.arctan2(-disp_cur[1], disp_cur[0]))
    if angle_cur < 0:
        angle_cur = angle_cur + 360
    angle_diff = angle_cen - angle_cur
    cur_err = np.linalg.norm(LT_path[0:1] - cur_pos_sub, axis=1)[0]
    if angle_diff > 0:
        cur_err = -cur_err
    ang_tmp_cur = np.rad2deg(np.arctan2(-(cur_pos_sub[1] - LT_path[0, 1]), cur_pos_sub[0] - LT_path[0, 0]))
    if ang_tmp_cur < 0:
        ang_tmp_cur = ang_tmp_cur + 360
    LT_path_mod = LT_path.copy()
    if cur_err < 0:
        ang_tmp_tar = LT_heading_mod[0] - 90
        err_curve_coord = -cur_err * np.cos(np.deg2rad(ang_tmp_cur - ang_tmp_tar))
        for i in range(turn_start_idx):
            LT_path_mod[i, 0] = LT_path_mod[i, 0] + err_curve_coord * np.cos(np.deg2rad(LT_heading_mod[i] - 90))
            LT_path_mod[i, 1] = LT_path_mod[i, 1] - err_curve_coord * np.sin(np.deg2rad(LT_heading_mod[i] - 90))

        for i in range(turn_start_idx, merge_idx):
            LT_path_mod[i, 0] = LT_path_mod[i, 0] + ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.cos(np.deg2rad(LT_heading_mod[i] - 90))
            LT_path_mod[i, 1] = LT_path_mod[i, 1] - ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.sin(np.deg2rad(LT_heading_mod[i] - 90))
    else:
        ang_tmp_tar = LT_heading_mod[0] + 90
        err_curve_coord = cur_err * np.cos(np.deg2rad(ang_tmp_cur - ang_tmp_tar))
        for i in range(turn_start_idx):
            LT_path_mod[i, 0] = LT_path_mod[i, 0] + err_curve_coord * np.cos(np.deg2rad(LT_heading_mod[i] + 90))
            LT_path_mod[i, 1] = LT_path_mod[i, 1] - err_curve_coord * np.sin(np.deg2rad(LT_heading_mod[i] + 90))

        for i in range(turn_start_idx, merge_idx):
            LT_path_mod[i, 0] = LT_path_mod[i, 0] + ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.cos(np.deg2rad(LT_heading_mod[i] + 90))
            LT_path_mod[i, 1] = LT_path_mod[i, 1] - ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.sin(np.deg2rad(LT_heading_mod[i] + 90))

    return LT_path_mod


def path_mod_ST(ST_points, cur_pos_sub, cur_idx, ST_path, ST_heading_mod, turn_start_idx, merge_idx):
    disp_cen = ST_points[cur_idx, :] - ST_points[0, :]
    angle_cen = np.rad2deg(np.arctan2(-disp_cen[1], disp_cen[0]))
    if angle_cen < 0:
        angle_cen = angle_cen + 360
    disp_cur = cur_pos_sub - ST_points[0, :]
    angle_cur = np.rad2deg(np.arctan2(-disp_cur[1], disp_cur[0]))
    if angle_cur < 0:
        angle_cur = angle_cur + 360
    angle_diff = angle_cen - angle_cur
    cur_err = np.linalg.norm(ST_path[0:1] - cur_pos_sub, axis=1)[0]
    if angle_diff > 0:
        cur_err = -cur_err
    ang_tmp_cur = np.rad2deg(np.arctan2(-(cur_pos_sub[1] - ST_path[0, 1]), cur_pos_sub[0] - ST_path[0, 0]))
    if ang_tmp_cur < 0:
        ang_tmp_cur = ang_tmp_cur + 360
    ST_path_mod = ST_path.copy()
    if cur_err < 0:
        ang_tmp_tar = ST_heading_mod[0] - 90
        err_curve_coord = -cur_err * np.cos(np.deg2rad(ang_tmp_cur - ang_tmp_tar))
        for i in range(turn_start_idx):
            ST_path_mod[i, 0] = ST_path_mod[i, 0] + err_curve_coord * np.cos(np.deg2rad(ST_heading_mod[i] - 90))
            ST_path_mod[i, 1] = ST_path_mod[i, 1] - err_curve_coord * np.sin(np.deg2rad(ST_heading_mod[i] - 90))

        for i in range(turn_start_idx, merge_idx):
            ST_path_mod[i, 0] = ST_path_mod[i, 0] + ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.cos(np.deg2rad(ST_heading_mod[i] - 90))
            ST_path_mod[i, 1] = ST_path_mod[i, 1] - ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.sin(np.deg2rad(ST_heading_mod[i] - 90))
    else:
        ang_tmp_tar = ST_heading_mod[0] + 90
        err_curve_coord = cur_err * np.cos(np.deg2rad(ang_tmp_cur - ang_tmp_tar))
        for i in range(turn_start_idx):
            ST_path_mod[i, 0] = ST_path_mod[i, 0] + err_curve_coord * np.cos(np.deg2rad(ST_heading_mod[i] + 90))
            ST_path_mod[i, 1] = ST_path_mod[i, 1] - err_curve_coord * np.sin(np.deg2rad(ST_heading_mod[i] + 90))

        for i in range(turn_start_idx, merge_idx):
            ST_path_mod[i, 0] = ST_path_mod[i, 0] + ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.cos(np.deg2rad(ST_heading_mod[i] + 90))
            ST_path_mod[i, 1] = ST_path_mod[i, 1] - ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.sin(np.deg2rad(ST_heading_mod[i] + 90))

    return ST_path_mod


def path_mod_RT(RT_points, cur_pos_sub, cur_idx, RT_path, RT_heading_mod, turn_start_idx, merge_idx):
    disp_cen = RT_points[cur_idx, :] - RT_points[0, :]
    angle_cen = np.rad2deg(np.arctan2(-disp_cen[1], disp_cen[0]))
    if angle_cen < 0:
        angle_cen = angle_cen + 360
    disp_cur = cur_pos_sub - RT_points[0, :]
    angle_cur = np.rad2deg(np.arctan2(-disp_cur[1], disp_cur[0]))
    if angle_cur < 0:
        angle_cur = angle_cur + 360
    angle_diff = angle_cen - angle_cur
    cur_err = np.linalg.norm(RT_path[0:1] - cur_pos_sub, axis=1)[0]
    if angle_diff > 0:
        cur_err = -cur_err
    ang_tmp_cur = np.rad2deg(np.arctan2(-(cur_pos_sub[1] - RT_path[0, 1]), cur_pos_sub[0] - RT_path[0, 0]))
    if ang_tmp_cur < 0:
        ang_tmp_cur = ang_tmp_cur + 360
    RT_path_mod = RT_path.copy()
    if cur_err < 0:
        ang_tmp_tar = RT_heading_mod[0] - 90
        err_curve_coord = -cur_err * np.cos(np.deg2rad(ang_tmp_cur - ang_tmp_tar))
        for i in range(turn_start_idx):
            RT_path_mod[i, 0] = RT_path_mod[i, 0] + err_curve_coord * np.cos(np.deg2rad(RT_heading_mod[i] - 90))
            RT_path_mod[i, 1] = RT_path_mod[i, 1] - err_curve_coord * np.sin(np.deg2rad(RT_heading_mod[i] - 90))

        for i in range(turn_start_idx, merge_idx):
            RT_path_mod[i, 0] = RT_path_mod[i, 0] + ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.cos(np.deg2rad(RT_heading_mod[i] - 90))
            RT_path_mod[i, 1] = RT_path_mod[i, 1] - ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.sin(np.deg2rad(RT_heading_mod[i] - 90))
    else:
        ang_tmp_tar = RT_heading_mod[0] + 90
        err_curve_coord = cur_err * np.cos(np.deg2rad(ang_tmp_cur - ang_tmp_tar))
        for i in range(turn_start_idx):
            RT_path_mod[i, 0] = RT_path_mod[i, 0] + err_curve_coord * np.cos(np.deg2rad(RT_heading_mod[i] + 90))
            RT_path_mod[i, 1] = RT_path_mod[i, 1] - err_curve_coord * np.sin(np.deg2rad(RT_heading_mod[i] + 90))

        for i in range(turn_start_idx, merge_idx):
            RT_path_mod[i, 0] = RT_path_mod[i, 0] + ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.cos(np.deg2rad(RT_heading_mod[i] + 90))
            RT_path_mod[i, 1] = RT_path_mod[i, 1] - ((merge_idx - i) / (merge_idx - turn_start_idx)) * err_curve_coord * np.sin(np.deg2rad(RT_heading_mod[i] + 90))

    return RT_path_mod


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


def get_local_path_before_inlet(cur_pos, links, vehs, jj):
    threshold = 200
    vis_length = 800
    cur_pos_sub = np.asarray(cur_pos).copy()
    for i in range(3):
        if i == 0:
            LT_points = links[0][i]['points_BEV']
            ST_points = links[1][i]['points_BEV']
            RT_points = links[2][i]['points_BEV']
        else:
            LT_points = LT_points + links[0][i]['points_BEV'][1:]
            ST_points = ST_points + links[1][i]['points_BEV'][1:]
            RT_points = RT_points + links[2][i]['points_BEV'][1:]

    LT_points = np.asarray(LT_points)
    LT_heading = np.zeros_like(LT_points[1:-1, 0])
    for i in range(LT_heading.shape[0]):
        heading_prev = np.rad2deg(np.arctan2(-(LT_points[i + 1, 1] - LT_points[i, 1]), LT_points[i + 1, 0] - LT_points[i, 0]))
        heading_next = np.rad2deg(np.arctan2(-(LT_points[i + 2, 1] - LT_points[i + 1, 1]), LT_points[i + 2, 0] - LT_points[i + 1, 0]))
        if np.abs(heading_prev - heading_next) > 180:
            heading = (heading_next + heading_prev + 360) / 2
        else:
            heading = (heading_prev + heading_next) / 2
        LT_heading[i] = heading
    LT_points = LT_points[1:-1]

    ST_points = np.asarray(ST_points)
    ST_heading = np.zeros_like(ST_points[1:-1, 0])
    for i in range(ST_heading.shape[0]):
        heading_prev = np.rad2deg(np.arctan2(-(ST_points[i + 1, 1] - ST_points[i, 1]), ST_points[i + 1, 0] - ST_points[i, 0]))
        heading_next = np.rad2deg(np.arctan2(-(ST_points[i + 2, 1] - ST_points[i + 1, 1]), ST_points[i + 2, 0] - ST_points[i + 1, 0]))
        if np.abs(heading_prev - heading_next) > 180:
            heading = (heading_next + heading_prev + 360) / 2
        else:
            heading = (heading_prev + heading_next) / 2
        ST_heading[i] = heading
    ST_points = ST_points[1:-1]

    RT_points = np.asarray(RT_points)
    RT_heading = np.zeros_like(RT_points[1:-1, 0])
    for i in range(RT_heading.shape[0]):
        heading_prev = np.rad2deg(np.arctan2(-(RT_points[i + 1, 1] - RT_points[i, 1]), RT_points[i + 1, 0] - RT_points[i, 0]))
        heading_next = np.rad2deg(np.arctan2(-(RT_points[i + 2, 1] - RT_points[i + 1, 1]), RT_points[i + 2, 0] - RT_points[i + 1, 0]))
        if np.abs(heading_prev - heading_next) > 180:
            heading = (heading_next + heading_prev + 360) / 2
        else:
            heading = (heading_prev + heading_next) / 2
        RT_heading[i] = heading
    RT_points = RT_points[1:-1]

    # LT_path calc
    cur_idx = np.argmin(np.linalg.norm(LT_points - cur_pos_sub, axis=1))
    if np.linalg.norm(LT_points[cur_idx] - cur_pos_sub) < np.linalg.norm(LT_points[cur_idx + 1] - cur_pos_sub):
        cur_idx = cur_idx + 1
    LT_path = LT_points[cur_idx:].copy()
    turn_start_idx = np.argmin(np.linalg.norm(LT_path - links[0][0]['points_BEV'][-1], axis=1))
    LT_heading_mod = LT_heading[cur_idx:].copy()
    merge_idx = np.argmin(np.linalg.norm(LT_path - links[0][-1]['points_BEV'][0], axis=1))
    LT_path_mod = path_mod_LT(LT_points, cur_pos_sub, cur_idx, LT_path, LT_heading_mod, turn_start_idx, merge_idx)

    heading_LT_init = np.rad2deg(np.arctan2(-(LT_path_mod[0, 1] - cur_pos_sub[1]), LT_path_mod[0, 0] - cur_pos_sub[0]))
    if heading_LT_init < 0:
        heading_LT_init = heading_LT_init + 360
    if heading_LT_init + vehs['heading'][jj] > 0:
        direction = 'ccw'
    else:
        direction = 'cw'
    if merge_idx == 0:
        heading_LT_init = - vehs['heading'][jj]
    if np.min(np.linalg.norm(LT_points - cur_pos_sub, axis=1)) > threshold:
        LT_path_mod = None
        LT_exist = False
    else:
        LT_exist = True
        distance = np.linalg.norm(LT_path_mod[0] - cur_pos_sub)
        max_angle_diff, _ = angle_diff_max(distance, vehs, jj, direction)

        if np.abs(heading_LT_init + vehs['heading'][jj]) > np.abs(max_angle_diff):
            LT_path_mod_1, remain = kinematic_update(cur_pos_sub, LT_path_mod, threshold, vehs, jj, direction, LT_heading_mod)
            cur_pos_sub = LT_path_mod_1[-1]
            cur_idx = np.argmin(np.linalg.norm(LT_path_mod - cur_pos_sub, axis=1))
            if np.linalg.norm(LT_path_mod[cur_idx] - cur_pos_sub) < np.linalg.norm(LT_path_mod[cur_idx + 1] - cur_pos_sub):
                cur_idx = cur_idx + 1
            LT_path = LT_path_mod[cur_idx:].copy()
            turn_start_idx = np.argmin(np.linalg.norm(LT_path - links[0][0]['points_BEV'][-1], axis=1))
            LT_heading_mod = LT_heading_mod[cur_idx:].copy()
            merge_idx = np.argmin(np.linalg.norm(LT_path - links[0][-1]['points_BEV'][0], axis=1))
            LT_path_mod_2 = path_mod_LT(LT_path_mod, cur_pos_sub, cur_idx, LT_path, LT_heading_mod, turn_start_idx, merge_idx)
            LT_path_mod = np.concatenate((LT_path_mod_1, LT_path_mod_2), axis=0)
        else:
            pass

    # ST_path calc
    cur_pos_sub = np.asarray(cur_pos).copy()
    cur_idx = np.argmin(np.linalg.norm(ST_points - cur_pos_sub, axis=1))
    if np.linalg.norm(ST_points[cur_idx] - cur_pos_sub) < np.linalg.norm(ST_points[cur_idx + 1] - cur_pos_sub):
        cur_idx = cur_idx + 1
    ST_path = ST_points[cur_idx:].copy()
    turn_start_idx = np.argmin(np.linalg.norm(ST_path - links[1][0]['points_BEV'][-1], axis=1))
    ST_heading_mod = ST_heading[cur_idx:].copy()
    merge_idx = np.argmin(np.linalg.norm(ST_path - links[1][-1]['points_BEV'][0], axis=1))
    ST_path_mod = path_mod_ST(ST_points, cur_pos_sub, cur_idx, ST_path, ST_heading_mod, turn_start_idx, merge_idx)
    heading_ST_init = np.rad2deg(np.arctan2(-(ST_path_mod[0, 1] - cur_pos_sub[1]), ST_path_mod[0, 0] - cur_pos_sub[0]))
    if heading_ST_init < 0:
        heading_ST_init = heading_ST_init + 360
    if heading_ST_init + vehs['heading'][jj] > 0:
        direction = 'ccw'
    else:
        direction = 'cw'
    if merge_idx == 0:
        heading_ST_init = - vehs['heading'][jj]
    if np.min(np.linalg.norm(ST_points - cur_pos_sub, axis=1)) > threshold:
        ST_path_mod = None
        ST_exist = False
    else:
        ST_exist = True
        distance = np.linalg.norm(ST_path_mod[0] - cur_pos_sub)
        max_angle_diff, _ = angle_diff_max(distance, vehs, jj, direction)
        if np.abs(heading_ST_init + vehs['heading'][jj]) > np.abs(max_angle_diff):
            ST_path_mod_1, remain = kinematic_update(cur_pos_sub, ST_path_mod, threshold, vehs, jj, direction, ST_heading_mod)
            cur_pos_sub = ST_path_mod_1[-1]
            cur_idx = np.argmin(np.linalg.norm(ST_path_mod - cur_pos_sub, axis=1))
            if np.linalg.norm(ST_path_mod[cur_idx] - cur_pos_sub) < np.linalg.norm(ST_path_mod[cur_idx + 1] - cur_pos_sub):
                cur_idx = cur_idx + 1
            ST_path = ST_path_mod[cur_idx:].copy()
            turn_start_idx = np.argmin(np.linalg.norm(ST_path - links[1][0]['points_BEV'][-1], axis=1))
            ST_heading_mod = ST_heading_mod[cur_idx:].copy()
            merge_idx = np.argmin(np.linalg.norm(ST_path - links[1][-1]['points_BEV'][0], axis=1))
            ST_path_mod_2 = path_mod_ST(ST_path_mod, cur_pos_sub, cur_idx, ST_path, ST_heading_mod, turn_start_idx, merge_idx)
            ST_path_mod = np.concatenate((ST_path_mod_1, ST_path_mod_2), axis=0)
        else:
            pass

    # RT_path calc
    cur_pos_sub = np.asarray(cur_pos).copy()
    cur_idx = np.argmin(np.linalg.norm(RT_points - cur_pos_sub, axis=1))
    if np.linalg.norm(RT_points[cur_idx] - cur_pos_sub) < np.linalg.norm(RT_points[cur_idx + 1] - cur_pos_sub):
        cur_idx = cur_idx + 1
    RT_path = RT_points[cur_idx:].copy()
    turn_start_idx = np.argmin(np.linalg.norm(RT_path - links[2][0]['points_BEV'][-1], axis=1))
    RT_heading_mod = RT_heading[cur_idx:].copy()
    merge_idx = np.argmin(np.linalg.norm(RT_path - links[2][-1]['points_BEV'][0], axis=1))
    RT_path_mod = path_mod_RT(RT_points, cur_pos_sub, cur_idx, RT_path, RT_heading_mod, turn_start_idx, merge_idx)
    heading_RT_init = np.rad2deg(np.arctan2(-(RT_path_mod[0, 1] - cur_pos_sub[1]), RT_path_mod[0, 0] - cur_pos_sub[0]))

    if heading_RT_init < 0:
        heading_RT_init = heading_RT_init + 360
    if heading_RT_init + vehs['heading'][jj] > 0:
        direction = 'ccw'
    else:
        direction = 'cw'
    if merge_idx == 0:
        heading_RT_init = - vehs['heading'][jj]
    if np.min(np.linalg.norm(RT_points - cur_pos_sub, axis=1)) > threshold:
        RT_path_mod = None
        RT_exist = False
    else:
        RT_exist = True
        distance = np.linalg.norm(RT_path_mod[0] - cur_pos_sub)
        max_angle_diff, _ = angle_diff_max(distance, vehs, jj, direction)
        if np.abs(heading_RT_init + vehs['heading'][jj]) > np.abs(max_angle_diff):
            RT_path_mod_1, _ = kinematic_update(cur_pos_sub, RT_path_mod, threshold, vehs, jj, direction, RT_heading_mod)
            cur_pos_sub = RT_path_mod_1[-1]
            cur_idx = np.argmin(np.linalg.norm(RT_path_mod - cur_pos_sub, axis=1))
            if np.linalg.norm(RT_path_mod[cur_idx] - cur_pos_sub) < np.linalg.norm(RT_path_mod[cur_idx + 1] - cur_pos_sub):
                cur_idx = cur_idx + 1
            RT_path = RT_path_mod[cur_idx:].copy()
            turn_start_idx = np.argmin(np.linalg.norm(RT_path - links[2][0]['points_BEV'][-1], axis=1))
            RT_heading_mod = RT_heading_mod[cur_idx:].copy()
            merge_idx = np.argmin(np.linalg.norm(RT_path - links[2][-1]['points_BEV'][0], axis=1))
            RT_path_mod_2 = path_mod_RT(RT_path_mod, cur_pos_sub, cur_idx, RT_path, RT_heading_mod, turn_start_idx, merge_idx)
            RT_path_mod = np.concatenate((RT_path_mod_1, RT_path_mod_2), axis=0)
        else:
            pass

    cur_pos_sub = np.asarray(cur_pos).copy()

    length = 0
    if LT_exist == False:
        pass
    else:
        for i in range(1, len(LT_path_mod)):
            length = length + np.linalg.norm(LT_path_mod[i] - LT_path_mod[i - 1])
            if length > vis_length:
                break
        LT_final = LT_path_mod[:i]
        LT_final = np.concatenate((np.expand_dims(cur_pos_sub, axis=0), LT_final))
        LT_final = B_spline(LT_final)
        for k in range(len(LT_path_mod) - 1):
            al = (len(LT_path_mod) - 2 - k) / (len(LT_path_mod) - 2)
            ax_main.plot(LT_final[k:k + 2, 0], LT_final[k:k + 2, 1], lw=10, color=(0, 0, 1, 0.3 * al ** 2), rasterized=True, zorder=0)

    length = 0
    if ST_exist == False:
        pass
    else:
        for i in range(1, len(ST_path_mod)):
            length = length + np.linalg.norm(ST_path_mod[i] - ST_path_mod[i - 1])
            if length > vis_length:
                break
        ST_final = ST_path_mod[:i]
        ST_final = np.concatenate((np.expand_dims(cur_pos_sub, axis=0), ST_final))
        ST_final = B_spline(ST_final)
        for k in range(len(ST_path_mod) - 1):
            al = (len(ST_path_mod) - 2 - k) / (len(ST_path_mod) - 2)
            ax_main.plot(ST_final[k:k + 2, 0], ST_final[k:k + 2, 1], lw=10, color=(0, 0, 1, 0.3 * al ** 2), rasterized=True, zorder=0)

    length = 0
    if RT_exist == False:
        pass
    else:
        for i in range(1, len(RT_path_mod)):
            length = length + np.linalg.norm(RT_path_mod[i] - RT_path_mod[i - 1])
            if length > vis_length:
                break
        RT_final = RT_path_mod[:i]
        RT_final = np.concatenate((np.expand_dims(cur_pos_sub, axis=0), RT_final))
        RT_final = B_spline(RT_final)
        for k in range(len(RT_path_mod) - 1):
            al = (len(RT_path_mod) - 2 - k) / (len(RT_path_mod) - 2)
            ax_main.plot(RT_final[k:k + 2, 0], RT_final[k:k + 2, 1], lw=10, color=(0, 0, 1, 0.3 * al ** 2), rasterized=True, zorder=0)


data_id = '1001'
with open(cur_path + 'MORAI_map/' + data_id + '/link_set_mod.json') as json_file:
    links_sim_coord = json.load(json_file)

with open(cur_path + 'MORAI_map/' + data_id + '/node_set_mod.json') as json_file:
    nodes_sim_coord = json.load(json_file)

scene_list = glob.glob(os.getcwd() + '\\BEV_visualization\\10*')
for i in range(len(scene_list)):
    scenario_id = scene_list[i].split('\\')[-1]
    landmark, recordingMeta, tracks, tracksMeta, tracksClass = data_load(cur_path, scenario_id)
    origin_GT = []
    with open(cur_path + 'MORAI_map/' + '1001' + '/csv/LandMark.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row_mod = row[0].split(',')
            if row_mod[0] == 'id':
                pass
            else:
                origin_GT.append([float(row_mod[1]), float(row_mod[2])])
    origin_GT_np = np.asarray(origin_GT)
    disp_1_2_sim = np.linalg.norm(origin_GT_np[0, :] - origin_GT_np[1, :])
    disp_2_3_sim = np.linalg.norm(origin_GT_np[1, :] - origin_GT_np[2, :])
    disp_3_4_sim = np.linalg.norm(origin_GT_np[2, :] - origin_GT_np[3, :])
    disp_4_1_sim = np.linalg.norm(origin_GT_np[3, :] - origin_GT_np[0, :])

    os.makedirs(os.getcwd() + '\\BEV_visualization\\' + scenario_id + '\\mgeos', exist_ok=True)

    img_path = 'BEV_visualization\\' + scenario_id + '\\images\\' + scenario_id + '_'

    frame_list = list(tracks[:, 2])
    frame_list = list(dict.fromkeys(frame_list))
    frame_list = ['0' * (4 - len(str(int(frame_list[i]) + 1))) + str(int(frame_list[i]) + 1) for i in range(len(frame_list))]
    hist_time = 2

    plt.figure(figsize=(64 / 4, 50 / 4))
    ax_main = plt.subplot2grid((50, 70), (0, 0), colspan=70, rowspan=36)
    ax_prob_1 = plt.subplot2grid((50, 70), (38, 0), colspan=16, rowspan=12)
    ax_prob_1.set_facecolor((0, 0.5, 0, 0.5))
    ax_prob_2 = plt.subplot2grid((50, 70), (38, 18), colspan=16, rowspan=12)
    ax_prob_2.set_facecolor((0.5, 0.5, 0, 0.5))
    ax_prob_3 = plt.subplot2grid((50, 70), (38, 36), colspan=16, rowspan=12)
    ax_prob_3.set_facecolor((0.5, 0, 0.5, 0.5))
    ax_prob_4 = plt.subplot2grid((50, 70), (38, 54), colspan=16, rowspan=12)
    ax_prob_4.set_facecolor((0, 0.5, 0.5, 0.5))
    ax_probs = [ax_prob_1, ax_prob_2, ax_prob_3, ax_prob_4]
    save_path = 'BEV_visualization\\tmp\\'
    os.makedirs(save_path, exist_ok=True)
    prob_bag = [[], [], [], []]
    plot_idx = [None, None, None, None]
    for j in range(1, landmark.shape[0]):
        print(j/landmark.shape[0])
        ax_main.axis('off')

        for axs in ax_probs:
            axs.set_ylim([0, 12 / 9])
            axs.set_xlim([-10, 0.2])
            axs.plot(np.asarray([0, 0]), np.asarray([0, 1]), c=(0.5, 0.5, 0.5, 0.5), ls='--')

        frame = int(frame_list[j]) - 1
        # img = mpimg.imread(img_path + frame_list[j] + '.jpg')
        # plt.imshow(img)

        im = Image.open(img_path + frame_list[j] + '.jpg').convert("RGBA")
        im_array = np.asarray(im)
        ax_main.imshow(im_array)

        disp_1_2_GT = np.linalg.norm(np.asarray([landmark[j, 2], landmark[j, 3]]) - np.asarray([landmark[j, 4], landmark[j, 5]]))
        disp_2_3_GT = np.linalg.norm(np.asarray([landmark[j, 4], landmark[j, 5]]) - np.asarray([landmark[j, 6], landmark[j, 7]]))
        disp_3_4_GT = np.linalg.norm(np.asarray([landmark[j, 6], landmark[j, 7]]) - np.asarray([landmark[j, 8], landmark[j, 9]]))
        disp_4_1_GT = np.linalg.norm(np.asarray([landmark[j, 8], landmark[j, 9]]) - np.asarray([landmark[j, 2], landmark[j, 3]]))
        meter_per_pixel = np.mean([disp_1_2_sim / disp_1_2_GT, disp_2_3_sim / disp_2_3_GT, disp_3_4_sim / disp_3_4_GT, disp_4_1_sim / disp_4_1_GT])

        vehs = get_veh_in_frame(tracks, frame, recordingMeta[15], hist_time)
        veh_x, veh_y = get_veh_polygon(vehs)

        res_tot = coordinate_conversion(1, landmark[j, :], meter_per_pixel, origin_GT)
        trans_x = res_tot.x[0]
        trans_y = res_tot.x[1]
        rot = res_tot.x[2]

        link_waypoint = []
        link_id = []

        for k in range(len(links_sim_coord)):
            pts = np.asarray(links_sim_coord[k]['points'])
            pts[:, 0] = pts[:, 0] - trans_x
            pts[:, 1] = pts[:, 1] - trans_y
            theta = np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0]))
            if links_sim_coord[k]['idx'] == '00001106' or links_sim_coord[k]['idx'] == '00001107':
                x = (np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) * np.cos(np.deg2rad(theta - rot + 2.5))) / meter_per_pixel - 70
                y = -(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) * np.sin(np.deg2rad(theta - rot + 2.5))) / meter_per_pixel + 80
            elif links_sim_coord[k]['idx'] == '00001119' or links_sim_coord[k]['idx'] == '00001120' or links_sim_coord[k]['idx'] == '00001124' or links_sim_coord[k]['idx'] == '00001129':
                x = (np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) * np.cos(np.deg2rad(theta - rot + 2.5))) / meter_per_pixel - 30
                y = -(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) * np.sin(np.deg2rad(theta - rot + 2.5))) / meter_per_pixel + 105
                for way_point in range(len(x)):
                    x[way_point] = x[way_point] - 40 * (way_point / (len(x) - 1))
                    y[way_point] = y[way_point] - 25 * (way_point / (len(x) - 1))
            else:
                x = (np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) * np.cos(np.deg2rad(theta - rot + 2.5))) / meter_per_pixel - 30
                y = -(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) * np.sin(np.deg2rad(theta - rot + 2.5))) / meter_per_pixel + 105
            link_waypoint.append([x, y])
            links_sim_coord[k]['points_BEV'] = [[x[pt_idx], y[pt_idx]] for pt_idx in range(len(x))]
            link_id.append(links_sim_coord[k])
            # plt.text(x[0], y[0], s=links_sim_coord[k]['idx'])
            # plt.scatter(x, y, s=1)
        target_cand = []
        plot_idx_frame = [None, None, None, None]
        LT_line = [None, None, None, None]
        RT_line = [None, None, None, None]
        ST_line = [None, None, None, None]
        for jj in range(len(veh_x)):
            # create mask
            polygon = [(veh_x[jj][i], veh_y[jj][i]) for i in range(5)]
            maskIm = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
            ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
            mask = np.array(maskIm)
            # assemble new image (uint8: 0-255)
            newImArray = np.empty(im_array.shape, dtype='uint8')
            # colors (three first columns, RGB)
            newImArray[:, :, :3] = im_array[:, :, :3]
            # transparency (4th column)
            newImArray[:, :, 3] = mask * 255
            ax_main.imshow(newImArray, zorder=100)

            cur_pos_x = np.mean(veh_x[jj][:-1])
            cur_pos_y = np.mean(veh_y[jj][:-1])
            heading = np.rad2deg(np.arctan2(-(veh_y[jj][0] - veh_y[jj][3]), veh_x[jj][0] - veh_x[jj][3]))
            cur_heading = heading.copy()
            if heading < 0:
                heading = heading + 360
            ax_main.text(cur_pos_x, cur_pos_y, str(jj), ha='center', va='center', size='x-small')

            dist_to_link = [np.min(np.linalg.norm(np.asarray([xy_link[0] - cur_pos_x, xy_link[1] - cur_pos_y]).T, axis=1)) for xy_link in link_waypoint]
            sort = np.argsort(dist_to_link)
            cur_link_tmp = [link_id[sort[0]]]
            link_heading = np.rad2deg(np.arctan2(-(cur_link_tmp[0]['points_BEV'][-1][1] - cur_link_tmp[0]['points_BEV'][0][1]), cur_link_tmp[0]['points_BEV'][-1][0] - cur_link_tmp[0]['points_BEV'][0][0]))
            if link_heading < 0:
                link_heading = link_heading + 360
            if np.abs(np.abs(link_heading - heading) - 180) < 10:
                cur_link = [link_id[sort[1]]]
            else:
                cur_link = [cur_link_tmp[0]]

            if cur_link[0]['idx_int'] > 0:
                pass
            elif cur_link[0]['idx_int'] == 0:
                cur_pos_x = vehs['hist_x_tot'][jj][0]
                cur_pos_y = vehs['hist_y_tot'][jj][0]
                heading = np.rad2deg(np.arctan2(-(vehs['hist_y_tot'][jj][1] - vehs['hist_y_tot'][jj][0]), vehs['hist_x_tot'][jj][1] - vehs['hist_x_tot'][jj][0]))
                if heading < 0:
                    heading = heading + 360
                dist_to_link = [np.min(np.linalg.norm(np.asarray([xy_link[0] - cur_pos_x, xy_link[1] - cur_pos_y]).T, axis=1)) for xy_link in link_waypoint]
                sort = np.argsort(dist_to_link)
                cur_link_tmp = [link_id[sort[0]]]
                link_heading = np.rad2deg(np.arctan2(-(cur_link_tmp[0]['points_BEV'][-1][1] - cur_link_tmp[0]['points_BEV'][0][1]), cur_link_tmp[0]['points_BEV'][-1][0] - cur_link_tmp[0]['points_BEV'][0][0]))
                if link_heading < 0:
                    link_heading = link_heading + 360
                if np.abs(np.abs(link_heading - heading) - 180) < 10:
                    cur_link = [link_id[sort[1]]]
                else:
                    cur_link = [cur_link_tmp[0]]

            prev_link = [link for link in links_sim_coord if link['to_node_idx'] == cur_link[0]['from_node_idx']]
            if len(prev_link) > 0:
                prev_left_link = [link for link in links_sim_coord if link['to_node_idx'] == prev_link[0]['from_node_idx']]
                prev_right_link = [link for link in links_sim_coord if link['to_node_idx'] == prev_link[0]['from_node_idx']]
            else:
                prev_left_link = []
                prev_right_link = []
            left_link = [link for link in links_sim_coord if link['idx'] == cur_link[0]['left_lane_change_dst_link_idx']]
            right_link = [link for link in links_sim_coord if link['idx'] == cur_link[0]['right_lane_change_dst_link_idx']]
            next_link = [link for link in links_sim_coord if link['from_node_idx'] == cur_link[0]['to_node_idx']]
            if len(left_link) > 0:
                left_next_link = [link for link in links_sim_coord if link['from_node_idx'] == left_link[0]['to_node_idx']]
            else:
                left_next_link = []
            if len(right_link) > 0:
                right_next_link = [link for link in links_sim_coord if link['from_node_idx'] == right_link[0]['to_node_idx']]
            else:
                right_next_link = []

            if cur_link[0]['idx_int'] in [4, 6, 7, 11, 13, 14]:
                veh_idx = str(int(vehs['id'][jj]))
                if veh_idx in plot_idx:
                    plot_pos = plot_idx.index(veh_idx)
                    plot_idx[plot_pos] = None
                    prob_bag[plot_pos] = []

            elif cur_link[0]['idx_int'] > 0 & next_link[0]['idx_int'] == 0:
                LT_prob = np.random.rand()
                ST_prob = np.random.rand()
                RT_prob = np.random.rand()
                LT_prob_norm = LT_prob / (LT_prob + ST_prob + RT_prob)
                ST_prob_norm = ST_prob / (LT_prob + ST_prob + RT_prob)
                RT_prob_norm = RT_prob / (LT_prob + ST_prob + RT_prob)
                probs = [LT_prob_norm, ST_prob_norm, RT_prob_norm]
                veh_idx = str(int(vehs['id'][jj]))
                x_cen = vehs['x'][jj]
                y_cen = vehs['y'][jj]
                ax_main.text(x_cen, y_cen, str(int(vehs['id'][jj])), ha='center', va='center', size='x-large', zorder = 50000)

                if veh_idx in plot_idx:
                    plot_pos = plot_idx.index(veh_idx)
                    plot_idx_frame[plot_pos] = jj
                    prob_bag[plot_pos].append(probs)
                else:
                    if None in plot_idx:
                        plot_pos = plot_idx.index(None)
                        plot_idx[plot_pos] = veh_idx
                        plot_idx_frame[plot_pos] = jj
                        prob_bag[plot_pos].append(probs)
                    else:
                        pass

                if plot_pos == 0:
                    ax_main.plot(veh_x[jj], veh_y[jj], 'g', alpha=0.5, zorder=10000)
                    ax_main.fill(veh_x[jj], veh_y[jj], 'g', alpha=0.5, zorder=10000)
                elif plot_pos == 1:
                    ax_main.plot(veh_x[jj], veh_y[jj], 'y', alpha=0.5, zorder=10000)
                    ax_main.fill(veh_x[jj], veh_y[jj], 'y', alpha=0.5, zorder=10000)
                elif plot_pos == 2:
                    ax_main.plot(veh_x[jj], veh_y[jj], 'm', alpha=0.5, zorder=10000)
                    ax_main.fill(veh_x[jj], veh_y[jj], 'm', alpha=0.5, zorder=10000)
                elif plot_pos == 3:
                    ax_main.plot(veh_x[jj], veh_y[jj], 'c', alpha=0.5, zorder=10000)
                    ax_main.fill(veh_x[jj], veh_y[jj], 'c', alpha=0.5, zorder=10000)


                LT_prob_seq = [prob[0] for prob in prob_bag[plot_pos]]
                ST_prob_seq = [prob[1] for prob in prob_bag[plot_pos]]
                RT_prob_seq = [prob[2] for prob in prob_bag[plot_pos]]
                x = 0.1*np.asarray([-i for i in range(len(LT_prob_seq))])
                x.sort()
                ax_probs[plot_pos].plot(x, LT_prob_seq, 'o-', c='r', markersize=0, alpha=0.8, label='Left Turn Prob.')
                LT_line[plot_pos] = ax_probs[plot_pos].scatter(x, LT_prob_seq, c='r', s=5)
                ax_probs[plot_pos].plot(x, ST_prob_seq, 'o-', c='g', markersize=0, alpha=0.8, label='Go Straight Prob.')
                ST_line[plot_pos] = ax_probs[plot_pos].scatter(x, ST_prob_seq, c='g', s=5)
                ax_probs[plot_pos].plot(x, RT_prob_seq, 'o-', c='b', markersize=0, alpha=0.8, label='Right Turn Prob.')
                RT_line[plot_pos] = ax_probs[plot_pos].scatter(x, RT_prob_seq, c='b', s=5)
                title_font = {
                    'fontsize': 10,
                }
                ax_probs[plot_pos].set_title('Maneuver probability of the vehicle '+ plot_idx[plot_pos], fontdict=title_font)

                target_cand.append(jj)
                if len(next_link) == 1:
                    if next_link[0]['dir'] == 'ST':
                        ST_links = [cur_link[0], next_link[0]]
                        ST_links.append([link for link in links_sim_coord if link['from_node_idx'] == next_link[0]['to_node_idx']][0])
                        LT_links = [left_link[0]]
                        LT_links.append([link for link in links_sim_coord if link['from_node_idx'] == left_link[0]['to_node_idx']][0])
                        LT_links.append([link for link in links_sim_coord if link['from_node_idx'] == LT_links[-1]['to_node_idx']][0])
                        RT_links = [right_link[0]]
                        RT_links.append([link for link in links_sim_coord if ((link['from_node_idx'] == right_link[0]['to_node_idx']) & (link['dir'] == 'RT'))][0])
                        RT_links.append([link for link in links_sim_coord if link['from_node_idx'] == RT_links[-1]['to_node_idx']][0])
                    elif next_link[0]['dir'] == 'LT':
                        LT_links = [cur_link[0], next_link[0]]
                        LT_links.append([link for link in links_sim_coord if link['from_node_idx'] == next_link[0]['to_node_idx']][0])
                        ST_links = [right_link[0]]
                        ST_links.append([link for link in links_sim_coord if link['from_node_idx'] == right_link[0]['to_node_idx']][0])
                        ST_links.append([link for link in links_sim_coord if link['from_node_idx'] == ST_links[-1]['to_node_idx']][0])
                        double_right_link = [link for link in links_sim_coord if link['idx'] == right_link[0]['right_lane_change_dst_link_idx']]
                        RT_links = [double_right_link[0]]
                        RT_links.append([link for link in links_sim_coord if ((link['from_node_idx'] == double_right_link[0]['to_node_idx']) & (link['dir'] == 'RT'))][0])
                        RT_links.append([link for link in links_sim_coord if link['from_node_idx'] == RT_links[-1]['to_node_idx']][0])
                elif len(next_link) == 2:
                    ST_links = [cur_link[0]]
                    RT_links = [cur_link[0]]
                    for next in range(len(next_link)):
                        if next_link[next]['dir'] == 'ST':
                            ST_links.append(next_link[next])
                        elif next_link[next]['dir'] == 'RT':
                            RT_links.append(next_link[next])
                    ST_links.append([link for link in links_sim_coord if link['from_node_idx'] == ST_links[-1]['to_node_idx']][0])
                    RT_links.append([link for link in links_sim_coord if link['from_node_idx'] == RT_links[-1]['to_node_idx']][0])
                    double_left_link = [link for link in links_sim_coord if link['idx'] == left_link[0]['left_lane_change_dst_link_idx']]
                    LT_links = [double_left_link[0]]
                    LT_links.append([link for link in links_sim_coord if link['from_node_idx'] == double_left_link[0]['to_node_idx']][0])
                    LT_links.append([link for link in links_sim_coord if link['from_node_idx'] == LT_links[-1]['to_node_idx']][0])
                elif len(next_link) == 3:
                    ST_links = [cur_link[0]]
                    RT_links = [cur_link[0]]
                    LT_links = [cur_link[0]]
                    for next in range(len(next_link)):
                        if next_link[next]['dir'] == 'ST':
                            ST_links.append(next_link[next])
                        elif next_link[next]['dir'] == 'RT':
                            RT_links.append(next_link[next])
                        elif next_link[next]['dir'] == 'LT':
                            LT_links.append(next_link[next])
                    ST_links.append([link for link in links_sim_coord if link['from_node_idx'] == ST_links[-1]['to_node_idx']][0])
                    RT_links.append([link for link in links_sim_coord if link['from_node_idx'] == RT_links[-1]['to_node_idx']][0])
                    LT_links.append([link for link in links_sim_coord if link['from_node_idx'] == LT_links[-1]['to_node_idx']][0])
                cur_pos = [vehs['x'][jj], vehs['y'][jj]]
                links = [LT_links, ST_links, RT_links]
                get_local_path_before_inlet(cur_pos, links, vehs, jj)

            hist_traj_x = vehs['hist_x'][jj]
            hist_traj_y = vehs['hist_y'][jj]
            for k in range(len(hist_traj_x) - 1):
                ax_main.plot(hist_traj_x[k:k + 2], hist_traj_y[k:k + 2], 'r', alpha=k / (len(hist_traj_x) - 1), lw=2.5)
        for zxcv in range(len(ax_probs)):
            ax_probs[zxcv].set_xlim([-10, 0.2])
            ax_probs[zxcv].set_ylim([0, 12/9])
            ax_probs[zxcv].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax_probs[zxcv].set_xticks([-10, -8, -6, -4, -2, 0])
            if LT_line[zxcv] != None:
                ax_probs[zxcv].legend((LT_line[zxcv], ST_line[zxcv], RT_line[zxcv]), ('Left Turn Prob.', 'Go Straight Prob.', 'Right Turn Prob.'), fontsize='x-small')

        ax_main.set_ylim([2160, 0])
        ax_main.set_xlim([0, 3840])
        fig_name = save_path + '\\' + frame_list[j] + '_gauss.png'
        plt.savefig(fig_name, dpi=100)
        ax_main.clear()
        ax_probs[0].clear()
        ax_probs[1].clear()
        ax_probs[2].clear()
        ax_probs[3].clear()

    img_list = glob.glob('BEV_visualization\\tmp\\*_gauss.png')

    img_array = []
    for filename in img_list:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('BEV_visualization\\tmp\\gauss.mp4', cv2.VideoWriter_fourcc(*'FMP4'), 10, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
