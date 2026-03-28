import torch
import os
import pickle
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import *
from metrics import *
from model import *

parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1)
parser.add_argument('--n_txpcnn', type=int, default=5)
parser.add_argument('--kernel_size', type=int, default=3)

# Data params
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='nuscenes')

# Training params
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--clip_grad', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--use_lrschd', action="store_true", default=False)
parser.add_argument('--tag', default='nuscenes_run')

args = parser.parse_args()

print('*' * 30)
print("Training initiating....")
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


# Dataset
data_set = './datasets/' + args.dataset + '/'

dset_train = TrajectoryDataset(
    data_set + 'train/',
    obs_len=args.obs_seq_len,
    pred_len=args.pred_seq_len,
    skip=1,
    norm_lap_matr=True
)

loader_train = DataLoader(dset_train, batch_size=1, shuffle=True)

dset_val = TrajectoryDataset(
    data_set + 'val/',
    obs_len=args.obs_seq_len,
    pred_len=args.pred_seq_len,
    skip=1,
    norm_lap_matr=True
)

loader_val = DataLoader(dset_val, batch_size=1, shuffle=False)


# Model
model = social_stgcnn(
    n_stgcnn=args.n_stgcnn,
    n_txpcnn=args.n_txpcnn,
    output_feat=args.output_size,
    seq_len=args.obs_seq_len,
    kernel_size=args.kernel_size,
    pred_seq_len=args.pred_seq_len
).to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr)

checkpoint_dir = './checkpoint/' + args.tag + '/'
os.makedirs(checkpoint_dir, exist_ok=True)

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)


# ================= TRAIN =================
def train(epoch):
    model.train()
    total_loss = 0
    count = 0

    for batch in loader_train:
        batch = [tensor.to(device) for tensor in batch]

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        optimizer.zero_grad()

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        loss = graph_loss(V_pred, V_tr)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    print(f"TRAIN Epoch {epoch}: Loss {avg_loss}")
    return avg_loss


# ================= VALID =================
def validate(epoch):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in loader_val:
            batch = [tensor.to(device) for tensor in batch]

            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
            non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

            V_pred = V_pred.permute(0, 2, 3, 1)

            V_tr = V_tr.squeeze()
            V_pred = V_pred.squeeze()

            loss = graph_loss(V_pred, V_tr)

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    print(f"VAL Epoch {epoch}: Loss {avg_loss}")
    return avg_loss


# ================= RUN =================
print('Training started...')

best_val = float('inf')

for epoch in range(args.num_epochs):
    train_loss = train(epoch)
    val_loss = validate(epoch)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), checkpoint_dir + 'best.pth')

    print('-' * 30)