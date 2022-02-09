import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from torch import Tensor, nn
from typing import Dict, List, Tuple, Union
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

            if i == config['n_deconv_layer_enc']-1:
                output_padding = config['doconv_output_padding']
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
        self.deconv = deconv
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        self.generator = nn.Linear(n_actor, 2 * config["num_preds"])
        self.reconstructor = nn.Linear(n_actor, 2 * config["num_preds"])

    def forward(self, hist_traj):
        length_idx = torch.where(hist_traj[:, :, 0] == 0)
        hist_traj[hist_traj == -1] = 0
        for i in range(hist_traj.shape[0]):
            if i == 0:
                x = hist_traj[i, :length_idx[1][torch.where(length_idx[0] == i)[0][0]],:]
                x = torch.unsqueeze(x, dim=-1)
                x = torch.unsqueeze(x, dim=-1)
            else:
                cut = hist_traj[i, :length_idx[1][torch.where(length_idx[0] == i)[0][0]],:]
                cut = torch.unsqueeze(cut, dim=-1)
                cut = torch.unsqueeze(cut, dim=-1)
                x = torch.cat((x, cut), dim=0)

        for i, l in enumerate(deconv):
            x = deconv[i](x)


        preds = []
        recons = []

        hid = self.decoder(actors)
        preds.append(self.generator(hid))

        hid_for_ego = torch.cat([hid[x[0]:x[0 + 1]] for x in actor_idcs_mod])
        recons.append(self.reconstructor(hid_for_ego))

        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)
        reconstruction = torch.cat([x.unsqueeze(1) for x in recons], 1)
        reconstruction = reconstruction.view(reconstruction.size(0), reconstruction.size(1), -1, 2)

        for i in range(len(actor_idcs_mod)):
            idcs = actor_idcs_mod[i]
            ctrs = actor_ctrs_mod[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs
            reconstruction[i] = reconstruction[i] + ctrs[0]

        out = dict()
        out["reconstruction"], out["reg"] = [], []
        for i in range(len(actor_idcs_mod)):
            idcs = actor_idcs_mod[i]
            out["reg"].append(reg[idcs])
            out['reconstruction'].append(reconstruction[i, 0])
        return out
