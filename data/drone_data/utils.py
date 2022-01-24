import numpy as np
from scipy.optimize import minimize
import csv
import os


def polyfit_with_fixed_points(n, x, y, xf, yf):
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x ** np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf ** np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]


def point_regen(points_np):
    x_new = []
    y_new = []
    seg_length = 1
    t = np.linspace(0, 1, len(points_np))
    x = points_np[:, 0]
    y = points_np[:, 1]

    n, d, f = len(t), 7, 2
    params_x = polyfit_with_fixed_points(d, t[1:-1], x[1:-1], np.asarray([t[0], t[-1]]), np.asarray([x[0], x[-1]]))
    poly_x = np.polynomial.Polynomial(params_x)

    params_y = polyfit_with_fixed_points(d, t[1:-1], y[1:-1], np.asarray([t[0], t[-1]]), np.asarray([y[0], y[-1]]))
    poly_y = np.polynomial.Polynomial(params_y)

    t_cand = np.linspace(0, 1, 1001)
    x_cands = poly_x(t_cand)
    y_cands = poly_y(t_cand)
    x_cands[0] = points_np[0, 0]
    x_cands[-1] = points_np[-1, 0]
    y_cands[0] = points_np[0, 1]
    y_cands[-1] = points_np[-1, 1]

    cnt = 0
    for i in range(len(t_cand)):
        if i == 0:
            x_new.append(x_cands[i])
            y_new.append(y_cands[i])
            cnt = cnt + 1
        else:
            dist_cnt = np.sqrt(np.sum((x_cands[i] - x_new[-1]) ** 2 + (y_cands[i] - y_new[-1]) ** 2))
            if dist_cnt >= seg_length:
                x_new.append(x_cands[i])
                y_new.append(y_cands[i])
                cnt = cnt + 1
    x_new.append(x_cands[-1])
    y_new.append(y_cands[-1])
    return np.transpose(np.concatenate((np.array([x_new]), np.array([y_new])), axis=0))


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
    global landmark1
    global landmark2
    global landmark3

    meter_per_pixel = scale * recordingMeta[15]
    new_tracks = np.zeros_like(tracks)
    new_tracks[:] = tracks[:]
    landmark1_GT = np.asarray([origin_GT[0]])
    landmark2_GT = np.asarray([origin_GT[1]])
    landmark3_GT = np.asarray([origin_GT[2]])
    center_GT = [(landmark1_GT[0, 0] + landmark2_GT[0, 0] + landmark3_GT[0, 0]) / 3,
                 (landmark1_GT[0, 1] + landmark2_GT[0, 1] + landmark3_GT[0, 1]) / 3]

    for i in range(len(landmark)):
        print(i)
        cur_frame = landmark[i, 1]
        landmark1 = np.asarray([[landmark[i, 2] * meter_per_pixel, -landmark[i, 3] * meter_per_pixel]])
        landmark2 = np.asarray([[landmark[i, 4] * meter_per_pixel, -landmark[i, 5] * meter_per_pixel]])
        landmark3 = np.asarray([[landmark[i, 6] * meter_per_pixel, -landmark[i, 7] * meter_per_pixel]])
        center = [(landmark1[0, 0] + landmark2[0, 0] + landmark3[0, 0]) / 3,
                  (landmark1[0, 1] + landmark2[0, 1] + landmark3[0, 1]) / 3]

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


def func(x):
    trans_x = x[0]
    trans_y = x[1]
    rot = x[2]

    theta_1 = np.rad2deg(np.arctan2(landmark1[0][1], landmark1[0][0]))
    theta_2 = np.rad2deg(np.arctan2(landmark2[0][1], landmark2[0][0]))
    theta_3 = np.rad2deg(np.arctan2(landmark3[0][1], landmark3[0][0]))

    x_1 = trans_x + np.sqrt(landmark1[0][0] ** 2 + landmark1[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_1))
    y_1 = trans_y + np.sqrt(landmark1[0][0] ** 2 + landmark1[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_1))

    x_2 = trans_x + np.sqrt(landmark2[0][0] ** 2 + landmark2[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_2))
    y_2 = trans_y + np.sqrt(landmark2[0][0] ** 2 + landmark2[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_2))

    x_3 = trans_x + np.sqrt(landmark3[0][0] ** 2 + landmark3[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_3))
    y_3 = trans_y + np.sqrt(landmark3[0][0] ** 2 + landmark3[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_3))

    landmark1_trans = np.asarray([[x_1, y_1]])
    landmark2_trans = np.asarray([[x_2, y_2]])
    landmark3_trans = np.asarray([[x_3, y_3]])

    return np.linalg.norm(landmark1_GT - landmark1_trans) + np.linalg.norm(
        landmark2_GT - landmark2_trans) + np.linalg.norm(landmark3_GT - landmark3_trans)


def get_nearest_link(links, pos):
    min_dist = np.inf
    for i in range(len(links)):
        pt = np.asarray(links[i]['points'])
        min_dist_cand = np.min(np.linalg.norm(pt - pos, axis=1))
        if min_dist_cand < min_dist:
            min_dist = min_dist_cand
            min_seg = links[i]
            min_seg_int_idx = links[i]['idx_int']
    return min_seg_int_idx, min_seg, min_dist