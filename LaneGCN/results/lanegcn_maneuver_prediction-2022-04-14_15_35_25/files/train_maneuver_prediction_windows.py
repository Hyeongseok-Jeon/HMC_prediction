# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number
import os
from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    # import lanegcn as model
    # import lanegcn_multihead as model
    import lanegcn_maneuver_prediction as model
    from utils import Logger, load_pretrain

except:
    # import LaneGCN.lanegcn as model
    # import LaneGCN.lanegcn_multihead as model
    import LaneGCN.lanegcn_maneuver_prediction as model
    from LaneGCN.utils import Logger, load_pretrain

# cur_path = os.path.abspath(__file__)
cur_path = os.getcwd() + '/LaneGCN/train_window.py'
root_path = os.path.dirname(os.path.dirname(cur_path)) + '/LaneGCN'
project_root = os.path.dirname(root_path)
sys.path.append(root_path)

parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "-t", "--transfer", default='False', type=str, metavar="TRANSFER", help="transferring the pretrained encoder"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "--data_aug", default="under", type=str, metavar="WEIGHT", help="checkpoint path"
)

parser.add_argument("--port")

# decoder_training-2022-03-12_05_26_35
def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    config, Dataset, collate_fn, net, loss, post_process, optim = model.get_model()

    weight_dir = project_root + '/ckpt/maneuver_prediction-2022-03-10_04_02_34/model_15.pt'

    if args.transfer == 'True':
        weights = torch.load(weight_dir, map_location=lambda storage, loc: storage)
        load_pretrain(net.actor_net_jhs, weights["model_state_dict"])
        params = list(net.actor_net.parameters()) \
                 + list(net.mapping.parameters()) \
                 + list(net.map_net.parameters()) \
                 + list(net.a2m.parameters()) \
                 + list(net.m2m.parameters()) \
                 + list(net.m2a.parameters()) \
                 + list(net.a2a.parameters()) \
                 + list(net.pred_net.parameters())
        print('encoder weight is loaded from ' + weight_dir)
    else:
        params = list(net.parameters())
    opt = optim(params, config)

    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(net, ckpt["state_dict"])
        if args.resume:
            config["epoch"] = ckpt["epoch"]
            opt.load_state_dict(ckpt["opt_state"])

    if args.eval:
        # Data loader for evaluation
        dataset = Dataset(config["val_split"], config, train=False)
        val_loader = DataLoader(
            dataset,
            batch_size=config["val_batch_size"],
            num_workers=config["val_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val(config, val_loader, net, loss, post_process, 999)
        return

    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sys.stdout = Logger(log)

    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    config = dataset.config
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, net, loss, post_process, opt, val_loader)


def train(epoch, config, train_loader, net, loss, post_process, opt, val_loader=None):
    net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    for i, data in tqdm(enumerate(train_loader)):
        epoch += epoch_per_batch
        data = dict(data)

        output = net(data, mode='custom', transfer=True, phase='train')
        loss_out = loss(output, data, phase='train')
        post_out = post_process(output, data, phase='train')
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(net, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(config, val_loader, net, loss, post_process, epoch)
            return


def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = net(data, mode='custom',  transfer=True, phase='val')
            loss_out = loss(output, data, phase='val')
            post_out = post_process(output, data, phase='val')
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch)
    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


if __name__ == "__main__":
    main()
