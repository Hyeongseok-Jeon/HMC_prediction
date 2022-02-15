import os

# root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_path = os.getcwd()
data_dir = root_path + '/data/drone_data/processed/'

### config used ###
config = dict()

''' environment '''
config["GPU_id"] = 0

''' training '''
config["batch_size"] = 2
config["epoch"] = 50
config["n_warmup_steps"] = 30

''' data '''
config["data_dir"] = data_dir
config["occlusion_rate"] = 0.2
config["splicing_num"] = 2
config["LC_multiple"] = 5
config["FOV"] = 30
config["interpolate"] = False
config["max_pred_gap"] = 20

''' network '''
config["n_deconv_layer_enc"] = 7
config["deconv_kernel_size_list"] = [3, 3, 3, 3, 3, 3, 3]
config['deconv_chennel_num_list'] = [3, 4, 4, 8, 8, 16, 16]
config['deconv_stride_list'] = [1, 2, 2, 2, 2, 2, 2, ]
config['deconv_output_padding'] = 1
config['deconv_activation'] = 'elu'
config["n_hidden_after_deconv"] = 256
config["n_convgru_layer"] = 1
config["convgru_kernel_size"] = (3, 3)

config["convgru_output_layer_num"] = 5
config["convgru_output_channel_list"] = [16, 32, 64, 128, 256]
config["convgru_output_kernel_size_list"] = [5, 5, 5, 5, 5]
config["convgru_kernel_size"] = (3, 3)
config["convgru_kernel_size"] = (3, 3)


def calc_deconv_out(config):
    ch = 1
    for i in range(config["n_deconv_layer_enc"]):
        ch = (ch - 1) * config['deconv_stride_list'][i] + config["deconv_kernel_size_list"][i]
        print(ch)
