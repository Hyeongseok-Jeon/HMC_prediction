import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
# root_path = os.getcwd() + '/model/representation_learning/modules'
sys.path.insert(0, root_path)

from torch import nn
import torch
from ConvGRU import ConvGRU

class AutoRegressive(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(AutoRegressive, self).__init__()
        self.config = config
        self.convGRU = ConvGRU(input_size=(config["n_hidden_after_deconv"], config["n_hidden_after_deconv"]),
                               input_dim=config["deconv_chennel_num_list"][-1],
                               hidden_dim=config["deconv_chennel_num_list"][-1],
                               kernel_size=config["convgru_kernel_size"],
                               num_layers=config["n_convgru_layer"],
                               dtype=torch.cuda.FloatTensor,
                               batch_first=True,
                               return_all_layers=True)
        output = nn.ModuleList()
        for i in range(config["convgru_output_layer_num"]):
            if i == 0:
                ch_in = config['deconv_chennel_num_list'][-1]
                ch_out = config["convgru_output_channel_list"][i]
            else:
                ch_in = config["convgru_output_channel_list"][i-1]
                ch_out = config["convgru_output_channel_list"][i]

            layer = nn.Conv2d(in_channels=ch_in,
                              out_channels=ch_out,
                              kernel_size=config["convgru_output_kernel_size_list"][i],
                              padding=int((config["convgru_output_kernel_size_list"][i]-1)/2))
            output.append(layer)
            if i > 0:
                maxpool = nn.MaxPool2d(kernel_size=4,
                                       stride=4)
                output.append(maxpool)
        self.output = output


    def forward(self, ar_input, seq_len):
        ar_out, _ = self.convGRU(ar_input)
        for i in range(len(seq_len)):
            if i == 0:
                feature_map = ar_out[0][i:i+1,seq_len[i]-1]
            else:
                cand = ar_out[0][i:i+1,seq_len[i]-1]
                feature_map = torch.cat((feature_map,cand), dim=0)

        for i, l in enumerate(self.output):
            if i == 0:
                x = self.output[i](feature_map)
            else:
                x = self.output[i](x)

        out = torch.squeeze(x)
        return out