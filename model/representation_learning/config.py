import os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = root_path+'/data/drone_data/processed/'

### config used ###
config = dict()

''' environment '''
config["GPU_id"] = 0

''' training '''
config["batch_size"] = 100
config["epoch"] = 50

''' data '''
config["data_dir"] = data_dir
config["occlusion_rate"] = 0.2
config["splicing_num"] = 2
config["FOV"] = 30
config["interpolate"] = False

''' network '''
config["n_deconv_layer_enc"] = 5
config["deconv_kernel_size_list"] = [4, 5, 5, 5, 5]
config['deconv_chennel_num_list'] = [3, 4, 8, 16, 32]
config["n_hidden_after_deconv"] = 256