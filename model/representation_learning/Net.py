import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from torch import nn
from modules.encoder import Encoder
from modules.autoregressive import AutoRegressive

class BackBone(nn.Module):
    def __init__(self, config):
        super(BackBone, self).__init__()
        self.encoder = Encoder(config)
        self.autoregressive = AutoRegressive(config)

    def forward(self, hist_traj):
        enc_out, seq_len = self.encoder(hist_traj)
        representation = self.autoregressive(enc_out, seq_len)

        return representation

    # def predict(self, x, x_reverse, hidden1, hidden2):
    #     batch = x.size()[0]
    #
    #     z1 = self.encoder(x)
    #
    #     z1 = z1.transpose(1, 2)
    #     output1, hidden1 = self.gru1(z1, hidden1)
    #
    #     z2 = self.encoder(x_reverse)
    #     z2 = z2.transpose(1, 2)
    #     output2, hidden2 = self.gru2(z2, hidden2)
    #
    #     return torch.cat((output1, output2), dim=2)
