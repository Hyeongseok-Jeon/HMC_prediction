import os
import torch
# root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_path = os.getcwd()
data_dir = root_path + '/data/drone_data/processed/'

### config used ###
config = dict()

''' environment '''
config["GPU_id"] = 0

''' training '''
gpu_name = torch.cuda.get_device_name(0)
if '3070' in gpu_name:
    config["batch_size"] = 16
elif '3090' in gpu_name:
    config["batch_size"] = 16

config["epoch"] = 300
config["n_warmup_steps"] = 30
config["validataion_period"] = 1
config["ckpt_period"] = 5
config["ckpt_dir"] = root_path + '/ckpt/'
config["learning_rate"] = 0.000005

''' data '''
config["data_dir_train"] = data_dir + 'train/'
config["data_dir_val"] = data_dir + 'val/'
config["data_dir_orig"] = data_dir + 'archive/'
config["occlusion_rate"] = 0.2
config["splicing_num"] = 128
config["LC_multiple"] = 5
config["FOV"] = 30
config["interpolate"] = False
config["max_pred_time"] = 5
config["max_hist_time"] = 10
config["hz"] = 2
config["val_rate"] = 0.2

''' network '''
config["n_linear_layer"] = 4
config["n_hidden_after_deconv"] = 256


''' logging '''
config["log_dir"] = 'logs/'

def calc_deconv_out(config):
    ch = 1
    for i in range(config["n_deconv_layer_enc"]):
        ch = (ch - 1) * config['deconv_stride_list'][i] + config["deconv_kernel_size_list"][i]
        print(ch)
