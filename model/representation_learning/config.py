import os
import torch
# root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_path = os.getcwd()
data_dir_train = root_path + '/data/drone_data/processed/train/'
data_dir_val = root_path + '/data/drone_data/processed/val/'

### config used ###
config = dict()

''' environment '''
config["GPU_id"] = 0

''' training '''
gpu_name = torch.cuda.get_device_name(0)
if '3070' in gpu_name:
    config["batch_size"] = 64
elif '3090' in gpu_name:
    config["batch_size"] = 64

config["epoch"] = 300
config["n_warmup_steps"] = 30
config["validataion_peroid"] = 5

''' data '''
config["data_dir_train"] = data_dir_train
config["data_dir_val"] = data_dir_val
config["occlusion_rate"] = 0.2
config["splicing_num"] = 64
config["LC_multiple"] = 5
config["FOV"] = 30
config["interpolate"] = False
config["max_pred_time"] = 5
config["max_hist_time"] = 10
config["hz"] = 2
config["val_rate"] = 0.2

''' network '''
config["n_hidden_after_deconv"] = 256
config["n_convgru_layer"] = 1
config["convgru_kernel_size"] = (3, 3)

config["convgru_output_layer_num"] = 4
config["convgru_output_channel_list"] = [16, 32, 64, 128]
config["convgru_output_kernel_size_list"] = [5, 5, 5, 5]
config["convgru_kernel_size"] = (3, 3)
config["convgru_kernel_size"] = (3, 3)

''' logging '''
config["log_dir"] = 'logs/'

def calc_deconv_out(config):
    ch = 1
    for i in range(config["n_deconv_layer_enc"]):
        ch = (ch - 1) * config['deconv_stride_list'][i] + config["deconv_kernel_size_list"][i]
        print(ch)
