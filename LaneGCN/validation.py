# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import time
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
try:
    from utils import Logger, load_pretrain
except:
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
    sys.path.insert(0, model_path)
    import lanegcn as model
elif 'lanegcn_multihead_scoring' in file_id:
    sys.path.insert(0, model_path)
    import lanegcn_multihead_scoring as model
elif 'lanegcn_multihead_pre' in file_id:
    sys.path.insert(0, model_path)
    import lanegcn_multihead as model

print('Model is loaded from: ', end=" ")
print(model.__file__)

ckpt_list = os.listdir(os.getcwd() + '/LaneGCN/results/' + file_id)
ckpt_list = [x for x in ckpt_list if '.ckpt' in x]

# print('------------------------------------------------------------')
# for i in range(len(ckpt_list)):
#     print('File_id : ' + str(ckpt_list[i]), '  File_index : ' + str(i))
# print('------------------------------------------------------------')
# print('\n')

f = open(os.getcwd() + '/LaneGCN/results/' + file_id + '/log', 'r')
lines = f.readlines()
val_sig = 0
measure = [100, 100, 100, 100]
best_epoch = 0
for i in range(len(lines)):
    line = lines[i][:-1]
    if val_sig == 0:
        if 'Epoch' in line:
            best_epoch_cand = int(float(line[6:10]))
        if 'Validation' in line:
            val_sig = 1
    elif val_sig == 1:
        val_sig = 0
        ade1 = float(line[line.index('ade1 ')+5:line.index('ade1 ')+11])
        fde1 = float(line[line.index('fde1 ') + 5:line.index('fde1 ') + 11])
        ade = float(line[line.index('ade ') + 4:line.index('ade ') + 10])
        fde = float(line[line.index('fde ') + 4:line.index('fde ') + 10])
        if np.mean(measure) > np.mean([ade1, fde1, ade, fde]):
            best_epoch = best_epoch_cand
            measure = [ade1, fde1, ade, fde]
f.close()
weight = str(best_epoch)+'.000.ckpt'

print('The best epoch is : ' + weight)
#
# while True:
#     s = input('selected target models : ')
#     try:
#         if int(s) < len(ckpt_list) and int(s) >= 0:
#             file_index = int(s)
#             weight = ckpt_list[file_index]
#             break
#         else:
#             pass
#     except:
#         pass

weight_dir = os.getcwd() + '/LaneGCN/results/' + file_id + '/' + weight

config, Dataset, collate_fn, net, loss, post_process, optim = model.get_model()
try:
    loaded_weight = weight
    weights = torch.load(weight_dir, map_location=lambda storage, loc: storage)
except:
    weight_list = os.listdir(os.path.dirname(weight_dir))
    weight_list = [int(float(file[:-5])) for file in weight_list if file.endswith(".ckpt")]
    weight_idx = np.argmin(np.asarray(weight_list)-best_epoch_cand)
    loaded_weight = str(weight_list[weight_idx]) + '.000.ckpt'
    weight_dir = os.getcwd() + '/LaneGCN/results/' + file_id + '/' + loaded_weight
    weights = torch.load(weight_dir, map_location=lambda storage, loc: storage)
print('The loaded weight is : ' + loaded_weight)

load_pretrain(net, weights["state_dict"])

config["preprocess_val"] = os.getcwd() + '/LaneGCN/dataset/preprocess/val_crs_dist6_angle90.p'
config["val_batch_size"] = 32
# Data loader for evaluation
dataset = Dataset(config["val_split"], config, train=False)
val_loader = DataLoader(
    dataset,
    batch_size=config["val_batch_size"],
    num_workers=config["val_workers"],
    collate_fn=collate_fn,
    pin_memory=True,
)

val_maneuver = pd.read_csv(os.getcwd() + '/LaneGCN/dataset/preprocess/val_data.csv')
file_list = list(val_maneuver['file name'])

net.eval()
start_time = time.time()
metrics_tot = dict()
metrics_LT = dict()
metrics_ST = dict()
metrics_RT = dict()
model_name = weight_dir.split('/')[-2]

for i, data in tqdm(enumerate(val_loader)):
    data = dict(data)
    maneuver = []
    for iii in range(len(data['file_name'])):
        try:
            maneuver.append(val_maneuver['target maneuver'][file_list.index(str(data['file_name'][iii]) + '.csv')])
        except:
            maneuver.append(0)

    with torch.no_grad():
        if 'lanegcn-CPC_backbone' in model_name:
            output = net(data, mode='custom')

        elif 'lanegcn-original' == model_name:
            output = net(data)

        elif 'lanegcn-original_k3' == model_name or 'lanegcn_multihead_pretrained_weight' == model_name or 'lanegcn_multihead_scoring' in model_name:
            output = net(data, mode='custom', transfer=True, phase='train')

        loss_out = loss(output, data)
        post_out = post_process(output, data)
        post_out_LT = dict()
        post_out_LT['preds'] = [post_out['preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'LEFT']
        post_out_LT['gt_preds'] = [post_out['gt_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'LEFT']
        post_out_LT['has_preds'] = [post_out['has_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'LEFT']
        post_out_ST = dict()
        post_out_ST['preds'] = [post_out['preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'go_straight']
        post_out_ST['gt_preds'] = [post_out['gt_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'go_straight']
        post_out_ST['has_preds'] = [post_out['has_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'go_straight']
        post_out_RT = dict()
        post_out_RT['preds'] = [post_out['preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'RIGHT']
        post_out_RT['gt_preds'] = [post_out['gt_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'RIGHT']
        post_out_RT['has_preds'] = [post_out['has_preds'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'RIGHT']
        if 'lanegcn_multihead_scoring' in model_name:
            post_out_LT['pred_maneuver'] = [post_out['pred_maneuver'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'LEFT']
            post_out_ST['pred_maneuver'] = [post_out['pred_maneuver'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'go_straight']
            post_out_RT['pred_maneuver'] = [post_out['pred_maneuver'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'RIGHT']
            post_out_LT['gt_maneuver'] = [post_out['gt_maneuver'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'LEFT']
            post_out_ST['gt_maneuver'] = [post_out['gt_maneuver'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'go_straight']
            post_out_RT['gt_maneuver'] = [post_out['gt_maneuver'][j] for j in range(len(post_out['preds'])) if maneuver[j] == 'RIGHT']

        post_process.append(metrics_tot, loss_out, post_out)
        post_process.append(metrics_LT, loss_out, post_out_LT)
        post_process.append(metrics_ST, loss_out, post_out_ST)
        post_process.append(metrics_RT, loss_out, post_out_RT)

log = os.path.join(os.path.dirname(os.path.dirname(model_path)), "validation_result")
sys.stdout = Logger(log)

dt = time.time() - start_time
print('validation result for total validation set')
post_process.display(metrics_tot, dt, 0)

print('validation result for left turn validation set')
post_process.display(metrics_LT, dt, 0)

print('validation result for Go straight validation set')
post_process.display(metrics_ST, dt, 0)

print('validation result for Right turn validation set')
post_process.display(metrics_RT, dt, 0)