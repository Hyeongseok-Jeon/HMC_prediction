import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from torch import nn
import torch


class Encoder(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        deconv = nn.ModuleList()
        for i in range(config['n_deconv_layer_enc']):
            if i == 0:
                ch_in = 3
                ch_out = config['deconv_chennel_num_list'][i]
            else:
                ch_in = config['deconv_chennel_num_list'][i - 1]
                ch_out = config['deconv_chennel_num_list'][i]

            if i == config['n_deconv_layer_enc'] - 1:
                output_padding = config['deconv_output_padding']
            else:
                output_padding = 0
            layer = nn.ConvTranspose2d(in_channels=ch_in,
                                       out_channels=ch_out,
                                       kernel_size=config['deconv_kernel_size_list'][i],
                                       stride=config['deconv_stride_list'][i],
                                       output_padding=output_padding,
                                       bias=True,
                                       dilation=1)
            deconv.append(layer)
            if config["deconv_activation"] == 'elu':
                deconv.append(nn.ELU())
            elif config["deconv_activation"] == 'relu':
                deconv.append(nn.ReLU())
        self.deconv = deconv

    def forward(self, hist_traj):
        # hist_traj = observation_fit
        seq_len = []
        if hist_traj.shape[1] > 1:
            length_idx = torch.where(hist_traj[:, :, 0] == 0)
            hist_traj[hist_traj == -1] = 0
            for i in range(hist_traj.shape[0]):
                idx = len(torch.where(length_idx[0] == i)[0])
                if idx == 0:
                    end_index = hist_traj.shape[1]
                else:
                    end_index = length_idx[1][torch.where(length_idx[0] == i)[0][0]]
                if i == 0:
                    x = hist_traj[i, :end_index, :]
                    seq_len.append(x.shape[0])
                    x = torch.unsqueeze(x, dim=-1)
                    x = torch.unsqueeze(x, dim=-1)
                else:
                    cut = hist_traj[i, :end_index, :]
                    seq_len.append(cut.shape[0])
                    cut = torch.unsqueeze(cut, dim=-1)
                    cut = torch.unsqueeze(cut, dim=-1)
                    x = torch.cat((x, cut), dim=0)
        else:
            for i in range(hist_traj.shape[0]):
                if i == 0:
                    x = hist_traj[i]
                    seq_len.append(x.shape[0])
                    x = torch.unsqueeze(x, dim=-1)
                    x = torch.unsqueeze(x, dim=-1)
                else:
                    cut = hist_traj[i]
                    seq_len.append(cut.shape[0])
                    cut = torch.unsqueeze(cut, dim=-1)
                    cut = torch.unsqueeze(cut, dim=-1)
                    x = torch.cat((x, cut), dim=0)

        for i, l in enumerate(self.deconv):
            x = self.deconv[i](x)

        for i in range(hist_traj.shape[0]):
            if i == 0:
                enc_out = x[0:seq_len[i]]
            else:
                cand = x[seq_len[i - 1]:seq_len[i - 1] + seq_len[i]]
                enc_out = torch.cat((enc_out, cand), dim=0)

        return [enc_out, seq_len]
        # return [enc_out, torch.tensor(seq_len, device=torch.device(self.config["GPU_id_text"]))]
