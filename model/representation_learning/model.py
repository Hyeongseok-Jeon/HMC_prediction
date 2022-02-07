import sys
import os
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)
from modules.encoder import Encoder
from modules.autoregressive import AutoRegressive
from torch import Tensor, nn


class Net(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.encoder = Encoder
        self.autoregressive = AutoRegressive
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for i in range(timestep)])
    def forward(self, ):
        preds = []
        recons = []

        hid = self.decoder(actors)
        preds.append(self.generator(hid))

        hid_for_ego = torch.cat([hid[x[0]:x[0+1]] for x in actor_idcs_mod])
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
            out['reconstruction'].append(reconstruction[i,0])
        return out