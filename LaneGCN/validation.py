# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from importlib import import_module
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
import importlib.util as module_loader
from numbers import Number
import os
from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
from LaneGCN.utils import Logger, load_pretrain
import pandas as pd
# cur_path = os.path.abspath(__file__)
result_path = os.getcwd() + '/LaneGCN/results/'
model_list = os.listdir(result_path)

print('------------------------------------------------------------')
for i in range(len(model_list)):
    print('File_id : ' + str(model_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')


while True:
    s = input('selected target models : ')
    try:
        if int(s) < len(model_list) and int(s) >= 0:
            file_index = int(s)
            file_id = model_list[file_index].split('.')[0]
            break
        else:
            pass
    except:
        pass

model_path = result_path + file_id + '/files/'
'''
- lanegcn-CPC_backbone: CPC encoder사용외에는 모두 lanegcn과 동일
- lanegcn_multihead : CPC encoder + multihead 들어가있는 full model이지만 maneuver scoring 및 loss는 lanegcn과 동일
- lanegcn_multihead_scoring : CPC encoder + multihead + maneuver classification header 모두 들어가 있고 loss도 classification loss포함
'''

if 'lanegcn-original' in file_id or 'lanegcn-CPC_backbone' in file_id:
    sys.path.append(model_path)
    import lanegcn as model
elif 'lanegcn_multihead_scoring' in file_id:
    sys.path.append(model_path)
    import lanegcn_multihead_scoring as model
elif 'lanegcn_multihead_pre' in file_id:
    sys.path.append(model_path)
    import lanegcn_multihead as model

print('Model is loaded from: ', end =" ")
print(model.__file__)

s = input('please check the loaded model path')

ckpt_list = os.listdir(os.getcwd() + '/LaneGCN/results/' + file_id)
ckpt_list = [x for x in ckpt_list if '.ckpt' in x]

print('------------------------------------------------------------')
for i in range(len(ckpt_list)):
    print('File_id : ' + str(ckpt_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')

f = open(os.getcwd() + '/LaneGCN/results/' + file_id + '/log', 'w')
for i in range(1, 11):
    data = "%d번째 줄입니다.\n" % i
    f.write(data)
f.close()

while True:
    s = input('selected target models : ')
    try:
        if int(s) < len(ckpt_list) and int(s) >= 0:
            file_index = int(s)
            weight = ckpt_list[file_index]
            break
        else:
            pass
    except:
        pass

global weight_dir
weight_dir = os.getcwd() + '/LaneGCN/results/' + file_id + '/' + weight

def main():
    # Import all settings for experiment.

    config, Dataset, collate_fn, net, loss, post_process, optim = model.get_model()
    weights = torch.load(weight_dir, map_location=lambda storage, loc: storage)
    load_pretrain(net, weights["state_dict"])

    config["preprocess_val"] = os.getcwd()+'/LaneGCN/dataset/preprocess/val_crs_dist6_angle90.p'
    config["val_batch_size"] = 4
    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val(config, val_loader, net, loss, post_process, 999)
    return


def val(config, val_loader, net, loss, post_process, epoch):
    val_maneuver = pd.read_csv(os.getcwd() + '/LaneGCN/dataset/preprocess/val_data.csv')
    file_list = list(val_maneuver['file name'])

    net.eval()
    start_time = time.time()
    metrics_tot = dict()
    metrics_LT = dict()
    metrics_ST = dict()
    metrics_RT = dict()
    model_name = weight_dir.split('/')[-2]
    for i, data in enumerate(val_loader):
        data = dict(data)
        maneuver = []
        for iii in range(len(data['file_name'])):
            try:
                maneuver.append(val_maneuver['target maneuver'][file_list.index(str(data['file_name'][iii]) + '.csv')])
            except:
                maneuver.append(0)

        with torch.no_grad():
            if 'lanegcn-original' == model_name:
                output = net(data)
                loss_out = loss(output, data)
                post_out = post_process(output, data)
                post_out_LT = dict()
                post_out_LT['preds'] = [post_out['preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] =='LEFT']
                post_out_LT['gt_preds'] = [post_out['gt_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'LEFT']
                post_out_LT['has_preds'] = [post_out['has_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'LEFT']
                post_out_ST = dict()
                post_out_ST['preds'] = [post_out['preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] =='go_straight']
                post_out_ST['gt_preds'] = [post_out['gt_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] =='go_straight']
                post_out_ST['has_preds'] = [post_out['has_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] =='go_straight']
                post_out_RT = dict()
                post_out_RT['preds'] = [post_out['preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'RIGHT']
                post_out_RT['gt_preds'] = [post_out['gt_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'RIGHT']
                post_out_RT['has_preds'] = [post_out['has_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'RIGHT']
                post_process.append(metrics_tot, loss_out, post_out)
                post_process.append(metrics_LT, loss_out, post_out_LT)
                post_process.append(metrics_ST, loss_out, post_out_ST)
                post_process.append(metrics_RT, loss_out, post_out_RT)

                #
                # output = net(data, mode='custom',  transfer=True, phase='val')
                # loss_out = loss(output, data, phase='val')
                # post_out = post_process(output, data, phase='val')
                # post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    post_process.display(metrics_tot, dt, epoch)
    post_process.display(metrics_LT, dt, epoch)
    post_process.display(metrics_ST, dt, epoch)
    post_process.display(metrics_RT, dt, epoch)

    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


if __name__ == "__main__":
    main()
