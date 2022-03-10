from data.drone_data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config_enc import config
from model.representation_learning.Net_enc import BackBone
import torch
import os
import warnings
import time
import numpy as np
from openTSNE import TSNE
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import imageio
import glob


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

config["splicing_num"] = 1
config["occlusion_rate"] = 0
config["batch_size"] = 1
config["LC_multiple"] = 1
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


print('------------------------------------------------------------')
for i in range(len(ckpt_list)):
    print('File_id : ' + str(ckpt_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')

'''
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
'''

ckpt_list_seq = [5*i for i in range(len(ckpt_list)) if 5*i < len(ckpt_list)]
for test in range(len(ckpt_list_seq)):
    weight = ckpt_list[test]
    print(weight)
    val_dir = 'val/' + file_id + '/' + weight.split('.')[0]
    tsne_dir = val_dir + '/tsne_plot'
    os.makedirs(tsne_dir, exist_ok=True)

    warnings.filterwarnings("ignore", category=UserWarning)

    # dataset_original = pred_loader_1(config, 'orig')


    model = BackBone(config).cuda()
    weights = torch.load(ckpt_dir + '/' + weight)
    model.load_state_dict(weights['model_state_dict'])

    correct_num_tot = 0
    full_length_num_tot = 0
    loss_tot = 0
    loss_calc_num_tot = 0
    epoch_time = time.time()

    for i, data in enumerate(dataloader_train):
        print(weight, 100*i/len(dataloader_train.dataset))
        trajectory, traj_length, conversion, maneuvers = data
        maneuvers = maneuvers[0].numpy()
        trajectory = trajectory.float().cuda()

        representation_time_bag = model(trajectory, traj_length, mode='val')
        if i == 0:
            context_bag_train = [None if repres == None else repres.cpu().detach().numpy() for repres in representation_time_bag]
            maneuver_bag_train = [None if context_bag_train[i] is None else np.repeat(maneuvers, config["val_augmentation"], axis=0) for i in range(11)]
            context_bag_tot = [None if repres == None else repres.cpu().detach().numpy() for repres in representation_time_bag]
            maneuver_bag_tot = [None if context_bag_train[i] is None else np.repeat(maneuvers, config["val_augmentation"], axis=0) for i in range(11)]
        else:
            for j in range(len(representation_time_bag)):
                if representation_time_bag[j] is None:
                    pass
                else:
                    if context_bag_train[j] is None:
                        context_bag_train[j] = representation_time_bag[j].cpu().detach().numpy()
                        maneuver_bag_train[j] = np.repeat(maneuvers, config["val_augmentation"], axis=0)
                        context_bag_tot[j] = np.concatenate((context_bag_tot[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
                        maneuver_bag_tot[j] = np.concatenate((maneuver_bag_tot[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
                    else:
                        context_bag_train[j] = np.concatenate((context_bag_train[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
                        maneuver_bag_train[j] = np.concatenate((maneuver_bag_train[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
                        context_bag_tot[j] = np.concatenate((context_bag_tot[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
                        maneuver_bag_tot[j] = np.concatenate((maneuver_bag_tot[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
'''                        
        if i == 0:
            context_bag_train = [repres.cpu().detach().numpy() for repres in representation_time_bag]
            maneuver_bag_train = [np.repeat(maneuvers, config["val_augmentation"], axis=0) for _ in range(11)]
            context_bag_tot = [repres.cpu().detach().numpy() for repres in representation_time_bag]
            maneuver_bag_tot = [np.repeat(maneuvers, config["val_augmentation"], axis=0) for _ in range(11)]
        else:
            for j in range(len(representation_time_bag)):
                if representation_time_bag[j] == None:
                    pass
                else:
                    context_bag_train[j] = np.concatenate((context_bag_train[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
                    maneuver_bag_train[j] = np.concatenate((maneuver_bag_train[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
                    context_bag_tot[j] = np.concatenate((context_bag_tot[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
                    maneuver_bag_tot[j] = np.concatenate((maneuver_bag_tot[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
'''

    for i, data in enumerate(dataloader_val):
        print(weight, 100*i/len(dataloader_val.dataset))

        trajectory, traj_length, conversion, maneuvers = data
        maneuvers = maneuvers[0].numpy()
        trajectory = trajectory.float().cuda()

        representation_time_bag = model(trajectory, traj_length, mode='val')
        if i == 0:
            context_bag_val = [None if repres == None else repres.cpu().detach().numpy() for repres in representation_time_bag]
            maneuver_bag_val = [None if context_bag_val[i] is None else np.repeat(maneuvers, config["val_augmentation"], axis=0) for i in range(11)]
            context_bag_tot[j] = np.concatenate((context_bag_tot[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
            maneuver_bag_tot[j] = np.concatenate((maneuver_bag_tot[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
        else:
            for j in range(len(representation_time_bag)):
                if representation_time_bag[j] is None:
                    pass
                else:
                    if context_bag_val[j] is None:
                        context_bag_val[j] = representation_time_bag[j].cpu().detach().numpy()
                        maneuver_bag_val[j] = np.repeat(maneuvers, config["val_augmentation"], axis=0)
                        context_bag_tot[j] = np.concatenate((context_bag_tot[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
                        maneuver_bag_tot[j] = np.concatenate((maneuver_bag_tot[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
                    else:
                        context_bag_val[j] = np.concatenate((context_bag_val[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
                        maneuver_bag_val[j] = np.concatenate((maneuver_bag_val[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)
                        context_bag_tot[j] = np.concatenate((context_bag_tot[j], representation_time_bag[j].cpu().detach().numpy()), axis=0)
                        maneuver_bag_tot[j] = np.concatenate((maneuver_bag_tot[j], np.repeat(maneuvers, config["val_augmentation"], axis=0)), axis=0)


    n_components = 2
    tsne_hist = TSNE(n_components=n_components,
                perplexity=30,
                verbose=True)

    contexts_outlet = context_bag_tot[-2]
    context_tsne = tsne_hist.fit(contexts_outlet)

    for i in range(10):
        contexts = context_bag_tot[i]
        plt.figure()
        context_low_dim = context_tsne.transform(contexts)
        context_low_dim_u_turn = context_low_dim[maneuver_bag_tot[i][:, 0] == 1]
        context_low_dim_left_turn = context_low_dim[maneuver_bag_tot[i][:, 1] == 1]
        context_low_dim_go_straight = context_low_dim[maneuver_bag_tot[i][:, 2] == 1]
        context_low_dim_right_turn = context_low_dim[maneuver_bag_tot[i][:, 3] == 1]
        plt.scatter(context_low_dim_go_straight[:, 0], context_low_dim_go_straight[:, 1], c='b', label='Go Straight')
        plt.scatter(context_low_dim_right_turn[:, 0], context_low_dim_right_turn[:, 1], c='c', label='Right Turn')
        plt.scatter(context_low_dim_left_turn[:, 0], context_low_dim_left_turn[:, 1], c='g', label='Left Turn')
        plt.scatter(context_low_dim_u_turn[:, 0], context_low_dim_u_turn[:, 1], c='r', label='U-Turn')
        plt.legend(loc='upper right')
        plt.xlim(-100, 120)
        plt.ylim(-100, 120)
        plt.title('embeddings of hist observations: ' + str(0.5*(10-i)) +'sec before outlet')
        plt.savefig(tsne_dir+'/0_hist_embedding_on_hist_space_'+str(0.5*(10-i)) + 'sec_before_total.png')
        plt.close()



    for i in range(10):
        contexts = context_bag_train[i]
        plt.figure()
        context_low_dim = context_tsne.transform(contexts)
        context_low_dim_u_turn = context_low_dim[maneuver_bag_train[i][:, 0] == 1]
        context_low_dim_left_turn = context_low_dim[maneuver_bag_train[i][:, 1] == 1]
        context_low_dim_go_straight = context_low_dim[maneuver_bag_train[i][:, 2] == 1]
        context_low_dim_right_turn = context_low_dim[maneuver_bag_train[i][:, 3] == 1]
        plt.scatter(context_low_dim_go_straight[:, 0], context_low_dim_go_straight[:, 1], c='b', label='Go Straight')
        plt.scatter(context_low_dim_right_turn[:, 0], context_low_dim_right_turn[:, 1], c='c', label='Right Turn')
        plt.scatter(context_low_dim_left_turn[:, 0], context_low_dim_left_turn[:, 1], c='g', label='Left Turn')
        plt.scatter(context_low_dim_u_turn[:, 0], context_low_dim_u_turn[:, 1], c='r', label='U-Turn')
        plt.legend(loc='upper right')
        plt.xlim(-100, 120)
        plt.ylim(-100, 120)
        plt.title('embeddings of hist observations: ' + str(0.5*(10-i)) +'sec before outlet')
        plt.savefig(tsne_dir+'/1_hist_embedding_on_hist_space_'+str(0.5*(10-i)) + 'sec_before_train.png')
        plt.close()

    for i in range(10):
        contexts = context_bag_val[i]
        plt.figure()
        context_low_dim = context_tsne.transform(contexts)
        context_low_dim_u_turn = context_low_dim[maneuver_bag_val[i][:, 0] == 1]
        context_low_dim_left_turn = context_low_dim[maneuver_bag_val[i][:, 1] == 1]
        context_low_dim_go_straight = context_low_dim[maneuver_bag_val[i][:, 2] == 1]
        context_low_dim_right_turn = context_low_dim[maneuver_bag_val[i][:, 3] == 1]
        plt.scatter(context_low_dim_go_straight[:, 0], context_low_dim_go_straight[:, 1], c='b', label='Go Straight')
        plt.scatter(context_low_dim_right_turn[:, 0], context_low_dim_right_turn[:, 1], c='c', label='Right Turn')
        plt.scatter(context_low_dim_left_turn[:, 0], context_low_dim_left_turn[:, 1], c='g', label='Left Turn')
        plt.scatter(context_low_dim_u_turn[:, 0], context_low_dim_u_turn[:, 1], c='r', label='U-Turn')
        plt.legend(loc='upper right')
        plt.xlim(-100, 120)
        plt.ylim(-100, 120)
        plt.title('embeddings of hist observations: ' + str(0.5*(10-i)) +'sec before outlet')
        plt.savefig(tsne_dir+'/2_hist_embedding_on_hist_space_'+str(0.5*(10-i)) + 'sec_before_val.png')
        plt.close()

    images = []
    file_list_tot = glob.glob(tsne_dir+'/0_*.png')
    file_list_train = glob.glob(tsne_dir+'/1_*.png')
    file_list_val = glob.glob(tsne_dir+'/2_*.png')
    file_list_tot.reverse()
    file_list_train.reverse()
    file_list_val.reverse()

    with imageio.get_writer(tsne_dir + '/0_total.gif', mode='I', duration=0.5) as writer:
        for filename in file_list_tot:
            image = imageio.imread(filename)
            writer.append_data(image)

    with imageio.get_writer(tsne_dir + '/1_train.gif', mode='I', duration=0.5) as writer:
        for filename in file_list_train:
            image = imageio.imread(filename)
            writer.append_data(image)

    with imageio.get_writer(tsne_dir + '/2_val.gif', mode='I', duration=0.5) as writer:
        for filename in file_list_val:
            image = imageio.imread(filename)
            writer.append_data(image)
