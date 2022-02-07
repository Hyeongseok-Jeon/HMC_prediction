import os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = root_path+'/data/drone_data/processed/'

### config used ###
config = dict()

''' environment '''
config["GPU_id"] = 0

''' training '''
config["batch_size"] = 1
config["epoch"] = 50

''' data '''
config["data_dir"] = data_dir
config["occlusion_rate"] = 0.2
config["splicing_num"] = 128
