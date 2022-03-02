import os
import sys

# root_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.getcwd() + '/model/representation_learning/modules'
sys.path.insert(0, root_path)

from torch import nn
import torch


class AutoRegressive(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(AutoRegressive, self).__init__()
        self.config = config
        self.gru = nn.GRU(config["n_hidden_after_deconv"], config["n_hidden_after_deconv"], num_layers=1, bidirectional=False, batch_first=True)

    def forward(self, ar_in, seg_length):
        batch_list = []
        for i in range(len(seg_length)):
            if i == 0:
                input_tensor = ar_in[0:seg_length[i]]
            else:
                input_tensor = ar_in[sum(seg_length[:i]):sum(seg_length[:i]) + seg_length[i]]
            batch_list.append(input_tensor)

        input_tensor = torch.nn.utils.rnn.pad_sequence(batch_list, batch_first=True)
        ar_out, _ = self.gru(input_tensor)
        return ar_out
