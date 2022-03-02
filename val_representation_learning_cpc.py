from data.drone_data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config import config
from model.representation_learning.Net import BackBone
import torch
import os
import warnings
import time
import numpy as np
from openTSNE import TSNE
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

GPU_NUM = config["GPU_id"]
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

print('Data list loading ...\n')

file_list = os.listdir(os.getcwd() + '\logs')

print('------------------------------------------------------------')
for i in range(len(file_list)):
    print('File_id : ' + str(file_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')

while True:
    s = input('selected target models : ')
    try:
        if int(s) < len(file_list) and int(s) >= 0:
            file_index = int(s)
            file_id = file_list[file_index].split('.')[0]
            break
        else:
            pass
    except:
        pass

ckpt_dir = config['ckpt_dir'] + file_id
ckpt_list = os.listdir(ckpt_dir)
epoch_list = [int(ckpt_list[i].split('_')[1].split('.')[0]) for i in range(len(ckpt_list))]
idx = sorted(range(len(epoch_list)), key=lambda k: epoch_list[k])
ckpt_list = [ckpt_list[idx[i]] for i in range(len(idx))]

print('------------------------------------------------------------')
for i in range(len(ckpt_list)):
    print('File_id : ' + str(ckpt_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')

while True:
    s = input('selected target models : ')
    try:
        if int(s) < len(ckpt_list) and int(s) >= 0:
            weight_index = int(s)
            weight = ckpt_list[weight_index]
            break
        else:
            pass
    except:
        pass

val_dir = 'val/' + file_id + '/' + weight.split('.')[0]
tsne_dir = val_dir + '/tsne_plot'
os.makedirs(tsne_dir, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)

# dataset_original = pred_loader_1(config, 'orig')
config["splicing_num"] = 1
config["occlusion_rate"] = 0
config["batch_size"] = 1
config["LC_multiple"] = 1
dataset_train = pred_loader_1(config, 'train', mode='val')
dataset_val = pred_loader_1(config, 'val', mode='val')
dataset_tot = pred_loader_1(config, 'orig', mode='val')

dataloader_train = DataLoader(dataset_train,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)
dataloader_tot = DataLoader(dataset_tot,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)

model = BackBone(config).cuda()
weights = torch.load(ckpt_dir + '/' + weight)
model.load_state_dict(weights['model_state_dict'])

correct_num_tot = 0
full_length_num_tot = 0
loss_tot = 0
loss_calc_num_tot = 0
epoch_time = time.time()

pred_bag = [np.empty(shape=(1, 256)) for _ in range(10)]
hist_bag = [np.empty(shape=(1, 128)) for _ in range(10)]
inst_num_bag = [0 for _ in range(10)]
traj_bag = []
maneuver_bag = []
for i, data in enumerate(dataloader_tot):
    trajectory, traj_length, conversion, maneuvers = data
    trajectory = trajectory.float().cuda()

    pred, target, valuable_traj, pred_steps, hist_feature = model(trajectory, traj_length, mode='val')
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    valuable_traj = valuable_traj.cpu().detach().numpy()
    pred_steps = pred_steps.cpu().detach().numpy()
    hist_feature = hist_feature.cpu().detach().numpy()

    traj_bag.append(valuable_traj)
    if pred.shape[0] < config["max_pred_time"] * config["hz"]:
        masking_num = config["max_pred_time"] * config["hz"] - pred.shape[0]
        pred_steps = np.concatenate((np.array([config["max_pred_time"] * config["hz"] - k for k in range(masking_num)]), pred_steps), axis=0)
        pred = np.concatenate((np.zeros_like(pred[:masking_num]), pred), axis=0)
        hist_feature = np.concatenate((np.zeros_like(hist_feature[:masking_num]), hist_feature), axis=0)


    if i == 0:
        target_bag = target
        maneuver_bag = np.expand_dims(maneuvers, axis=0)
        conversion_bag = np.expand_dims(conversion, axis=0)
    else:
        target_bag_tmp = target
        target_bag = np.concatenate((target_bag, target_bag_tmp), axis=0)
        maneuver_bag_tmp = np.expand_dims(maneuvers, axis=0)
        maneuver_bag = np.concatenate((maneuver_bag, maneuver_bag_tmp), axis=0)
        conversion_bag_tmp = np.expand_dims(conversion, axis=0)
        conversion_bag = np.concatenate((conversion_bag, conversion_bag_tmp), axis=0)

    for j in range(pred.shape[0]):
        if inst_num_bag[config["max_pred_time"] * config["hz"] - pred_steps[j]] == 0:
            pred_bag[config["max_pred_time"] * config["hz"] - pred_steps[j]] = pred[j:j + 1, :]
            hist_bag[config["max_pred_time"] * config["hz"] - pred_steps[j]] = hist_feature[j:j + 1, :]
        else:
            pred_bag[config["max_pred_time"] * config["hz"] - pred_steps[j]] = np.concatenate((pred_bag[config["max_pred_time"] * config["hz"] - pred_steps[j]], pred[j:j + 1, :]), axis=0)
            hist_bag[config["max_pred_time"] * config["hz"] - pred_steps[j]] = np.concatenate((hist_bag[config["max_pred_time"] * config["hz"] - pred_steps[j]], hist_feature[j:j + 1, :]), axis=0)
        inst_num_bag[config["max_pred_time"] * config["hz"] - pred_steps[j]] += 1


n_components = 2
tsne_target = TSNE(n_components=n_components,
            perplexity=30,
            verbose=True)
tsne_hist = TSNE(n_components=n_components,
            perplexity=30,
            verbose=True)
#
# target_tsne = tsne_target.fit(target_bag)
# target_tsne_u_turn = target_tsne[maneuver_bag[:,0] == 1]
# target_tsne_left_turn = target_tsne[maneuver_bag[:,1] == 1]
# target_tsne_go_straight = target_tsne[maneuver_bag[:,2] == 1]
# target_tsne_right_turn = target_tsne[maneuver_bag[:,3] == 1]
# plt.figure()
# plt.scatter(target_tsne_u_turn[:,0], target_tsne_u_turn[:,1], c='r', label='U-Turn')
# plt.scatter(target_tsne_left_turn[:,0], target_tsne_left_turn[:,1], c='g', label='Left Turn')
# plt.scatter(target_tsne_go_straight[:,0], target_tsne_go_straight[:,1], c='b', label='Go Straight')
# plt.scatter(target_tsne_right_turn[:,0], target_tsne_right_turn[:,1], c='c', label='Right Turn')
# plt.legend(loc='upper right')
# plt.xlim(-30, 50)
# plt.ylim(-30, 50)
# plt.title('embeddings of outlet position')
# plt.savefig(tsne_dir+'/0_outlet_embedding_on_target_space.png')
# plt.close()
#
# for i in range(10):
#     maneuver_bags = maneuver_bag[pred_bag[i][:,0] != 0]
#     hists = hist_bag[i][hist_bag[i][:,0] != 0]
#     plt.figure()
#     hist_tsne = target_tsne.transform(hists)
#     hist_tsne_u_turn = hist_tsne[maneuver_bags[:, 0] == 1]
#     hist_tsne_left_turn = hist_tsne[maneuver_bags[:, 1] == 1]
#     hist_tsne_go_straight = hist_tsne[maneuver_bags[:, 2] == 1]
#     hist_tsne_right_turn = hist_tsne[maneuver_bags[:, 3] == 1]
#     plt.scatter(hist_tsne_u_turn[:, 0], hist_tsne_u_turn[:, 1], c='r', label='U-Turn')
#     plt.scatter(hist_tsne_left_turn[:, 0], hist_tsne_left_turn[:, 1], c='g', label='Left Turn')
#     plt.scatter(hist_tsne_go_straight[:, 0], hist_tsne_go_straight[:, 1], c='b', label='Go Straight')
#     plt.scatter(hist_tsne_right_turn[:, 0], hist_tsne_right_turn[:, 1], c='c', label='Right Turn')
#     plt.legend(loc='upper right')
#     plt.xlim(-30, 50)
#     plt.ylim(-30, 50)
#     plt.title('embeddings of hist observations: ' + str(0.5*(10-i)) +'sec before outlet')
#     plt.savefig(tsne_dir+'/0_hist_embedding_on_target_space_'+str(0.5*(10-i)) + 'sec_before.png')
#     plt.close()


i=9
maneuver_bags = maneuver_bag[pred_bag[i][:,0] != 0]
hists = hist_bag[i][hist_bag[i][:,0] != 0]
hist_tsne = tsne_hist.fit(hists)

for i in range(10):
    maneuver_bags = maneuver_bag[pred_bag[i][:,0] != 0]
    hists = hist_bag[i][hist_bag[i][:,0] != 0]
    plt.figure()
    hist_tsne_tmp = hist_tsne.transform(hists)
    hist_tsne_u_turn = hist_tsne_tmp[maneuver_bags[:, 0] == 1]
    hist_tsne_left_turn = hist_tsne_tmp[maneuver_bags[:, 1] == 1]
    hist_tsne_go_straight = hist_tsne_tmp[maneuver_bags[:, 2] == 1]
    hist_tsne_right_turn = hist_tsne_tmp[maneuver_bags[:, 3] == 1]
    plt.scatter(hist_tsne_u_turn[:, 0], hist_tsne_u_turn[:, 1], c='r', label='U-Turn')
    plt.scatter(hist_tsne_left_turn[:, 0], hist_tsne_left_turn[:, 1], c='g', label='Left Turn')
    plt.scatter(hist_tsne_go_straight[:, 0], hist_tsne_go_straight[:, 1], c='b', label='Go Straight')
    plt.scatter(hist_tsne_right_turn[:, 0], hist_tsne_right_turn[:, 1], c='c', label='Right Turn')
    plt.legend(loc='upper right')
    plt.xlim(-30, 40)
    plt.ylim(-40, 40)
    plt.title('embeddings of hist observations: ' + str(0.5*(10-i)) +'sec before outlet')
    plt.savefig(tsne_dir+'/1_hist_embedding_on_hist_space_'+str(0.5*(10-i)) + 'sec_before.png')
    plt.close()
#
#
# target_tsne = hist_tsne.transform(target_bag)
# target_tsne_u_turn = target_tsne[maneuver_bag[:,0] == 1]
# target_tsne_left_turn = target_tsne[maneuver_bag[:,1] == 1]
# target_tsne_go_straight = target_tsne[maneuver_bag[:,2] == 1]
# target_tsne_right_turn = target_tsne[maneuver_bag[:,3] == 1]
# plt.figure()
# plt.scatter(target_tsne_u_turn[:,0], target_tsne_u_turn[:,1], c='r', label='U-Turn')
# plt.scatter(target_tsne_left_turn[:,0], target_tsne_left_turn[:,1], c='g', label='Left Turn')
# plt.scatter(target_tsne_go_straight[:,0], target_tsne_go_straight[:,1], c='b', label='Go Straight')
# plt.scatter(target_tsne_right_turn[:,0], target_tsne_right_turn[:,1], c='c', label='Right Turn')
# plt.legend(loc='upper right')
# plt.xlim(-30, 40)
# plt.ylim(-40, 40)
# plt.title('embeddings of nearest-outlet position')
# plt.savefig(tsne_dir+'/1_outlet_embedding_on_hist_space_'+str(0.5*(10-i)) + 'sec_before.png')
# plt.close()