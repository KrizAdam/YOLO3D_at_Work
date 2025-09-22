"""
Script for training Regressor Model
python train_neu_mit_eval.py     --epochs 60     --batch_size 128     --num_workers 3     --save_epoch 10     --train_path ./data/daten     --model_path ./weights     --select_model resnet18     --api_key KfPzkJBufCei2Nko40NoOTYNy --bin_count 8
"""
from pytorch3d.ops import box3d_overlap
import argparse
import os
from random import shuffle
import sys
from pathlib import Path

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from comet_ml import Experiment

from script.Dataset import Dataset
from script.Model import ResNet18, VGG11, OrientationLoss

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, vgg11
from torch.utils import data

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# model factory to choose model
model_factory = {
    'resnet18': resnet18(pretrained=True),
    'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet18': ResNet18,
    'vgg11': VGG11
}
import torch.nn.functional as F
import numpy as np
from shapely.geometry import Polygon

def box3d_iou(pred_box, gt_box):
    """
    Compute 3D IoU between two 3D bounding boxes.
    pred_box, gt_box: [x, y, z, dx, dy, dz, yaw]
    """
    def get_corners_3d(box):
        x, y, z, dx, dy, dz, yaw = box
        x_corners = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
        y_corners = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
        z_corners = [dz/2, dz/2, dz/2, dz/2, -dz/2, -dz/2, -dz/2, -dz/2]
        corners = np.vstack([x_corners, y_corners, z_corners])
        R = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                      [ np.sin(yaw),  np.cos(yaw), 0],
                      [          0,           0, 1]])
        corners = R @ corners
        corners += np.array([[x], [y], [z]])
        return corners.T

    corners1, corners2 = get_corners_3d(pred_box), get_corners_3d(gt_box)
    vol1, vol2 = np.prod(pred_box[3:6]), np.prod(gt_box[3:6])

    poly1, poly2 = Polygon(corners1[:4, :2]), Polygon(corners2[:4, :2])
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    if inter_area == 0:
        return 0.0

    zmax1, zmin1 = corners1[:, 2].max(), corners1[:, 2].min()
    zmax2, zmin2 = corners2[:, 2].max(), corners2[:, 2].min()
    inter_h = max(0, min(zmax1, zmax2) - max(zmin1, zmin2))

    inter_vol = inter_area * inter_h
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0
    
def decode_yaw(orient, conf, bin_count):
    batch_size = orient.size(0)
    bin_idx = torch.argmax(conf, dim=1)  # [batch]
    chosen = orient[torch.arange(batch_size), bin_idx]  
    residual = torch.atan2(chosen[:, 1], chosen[:, 0])  
    bin_size = 2 * np.pi / bin_count
    bin_center = bin_idx.float() * bin_size + bin_size / 2  
    yaw = bin_center + residual  # [batch]
    return yaw


def compute_dim_error(pred_dim, gt_dim):
    return torch.mean(torch.abs(pred_dim - gt_dim)).item()
    
def compute_depth_error(pred_loc, gt_loc):
    return torch.mean(torch.abs(pred_loc[:, 2] - gt_loc[:, 2])).item()

def train(
    epochs=10,
    batch_size=32,
    alpha=0.6,
    w=0.4,
    num_workers=2,
    lr=0.0001,
    save_epoch=10,
    train_path=ROOT / 'data/daten',
    model_path=ROOT / 'weights/',
    select_model='resnet18',
    api_key='',
    bin_count=2
    ):

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils import data
    from tqdm import tqdm

    # directory
    train_path = str(train_path)
    model_path = str(model_path)

    # dataset
    print('[INFO] Loading dataset...')
    dataset = Dataset(train_path, bins=bin_count)

    # hyper_params
    hyper_params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'w': w,
        'num_workers': num_workers,
        'lr': lr,
        'shuffle': True
    }

    # comet ml experiment
    experiment = Experiment(api_key, project_name="YOLO3D")
    experiment.log_parameters(hyper_params)

    # data generator
    data_gen = data.DataLoader(
        dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=hyper_params['shuffle'],
        num_workers=hyper_params['num_workers'])

    # model
    base_model = model_factory[select_model]
    model = regressor_factory[select_model](model=base_model, bins=bin_count).cuda()
    
    # optimizer
    opt_SGD = torch.optim.SGD(model.parameters(), lr=hyper_params['lr'], momentum=0.9)

    # loss function
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    # load previous weights
    latest_model = None
    first_epoch = 1
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass

    if latest_model is not None:
        checkpoint = torch.load(os.path.join(model_path, latest_model))
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f'[INFO] Using previous model {latest_model} at {first_epoch} epochs')
        print('[INFO] Resuming training...')

    total_num_batches = int(len(dataset) / hyper_params['batch_size'])

    # --- helper function to decode yaw from bins --

    # --- training loop ---
    with experiment.train():
        for epoch in range(first_epoch, int(hyper_params['epochs'])+1):
            with tqdm(data_gen, unit='batch') as tepoch:
                for local_batch, local_labels in tepoch:
                    tepoch.set_description(f'Epoch {epoch}')

                    # ground-truth
                    truth_orient = local_labels['Orientation'].float().cuda()
                    truth_conf = local_labels['Confidence'].float().cuda()
                    truth_dim = local_labels['Dimensions'].float().cuda()

                    # convert to cuda
                    local_batch = local_batch.float().cuda()

                    # forward
                    orient, conf, dim = model(local_batch)

                    # loss
                    orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
                    dim_loss = dim_loss_func(dim, truth_dim)
                    truth_conf_idx = torch.max(truth_conf, dim=1)[1]
                    conf_loss = conf_loss_func(conf, truth_conf_idx)
                    loss_theta = conf_loss + w * orient_loss
                    loss = alpha * dim_loss + loss_theta

                    # --- compute metrics ---
                    gt_loc = local_labels['Location'].float().cuda() if 'Location' in local_labels else torch.zeros((len(truth_dim), 3)).cuda()
                    gt_yaw = local_labels['Orientation'].float().cuda() if 'Orientation' in local_labels else torch.zeros(len(truth_dim)).cuda()

                    # decode predicted yaw
                    pred_yaw = decode_yaw(orient, conf, bin_count)
                    pred_dim = dim
                    pred_loc = gt_loc  # replace with predicted location if available

                    # IoU
                    ious = []
                    for b in range(len(truth_dim)):
                        pred_box = [
    *pred_loc[b].detach().cpu().numpy(),
    *pred_dim[b].detach().cpu().numpy(),
    pred_yaw[b].detach().cpu().numpy()
]

                        gt_box   = [*gt_loc[b].cpu().numpy(), *truth_dim[b].cpu().numpy(), gt_yaw[:b].cpu().numpy()]
                        ious.append(box3d_iou(pred_box, gt_box))
                    mean_iou = np.mean(ious)

                    # AOS
                    cos_sim = torch.cos(pred_yaw - gt_yaw)
                    aos = torch.mean(cos_sim).item()

                    # Dimension error
                    dim_err = torch.mean(torch.abs(pred_dim - truth_dim)).item()

                    # Depth error (z coordinate)
                    depth_err = torch.mean(torch.abs(pred_loc[:, 2] - gt_loc[:, 2])).item()

                    # log metrics
                    writer.add_scalar('Loss/train', loss, epoch)
                    writer.add_scalar('IoU3D/train', mean_iou, epoch)
                    writer.add_scalar('Metrics/AOS', aos, epoch)
                    writer.add_scalar('Metrics/DimensionError', dim_err, epoch)
                    writer.add_scalar('Metrics/DepthError', depth_err, epoch)

                    experiment.log_metric('Loss/train', loss, epoch=epoch)
                    experiment.log_metric('IoU3D/train', mean_iou, epoch=epoch)
                    experiment.log_metric('AOS/train', aos, epoch=epoch)
                    experiment.log_metric('DimensionError/train', dim_err, epoch=epoch)
                    experiment.log_metric('DepthError/train', depth_err, epoch=epoch)

                    # backward
                    opt_SGD.zero_grad()
                    loss.backward()
                    opt_SGD.step()

                    tepoch.set_postfix(loss=loss.item())

            # save model every save_epoch
            if epoch % save_epoch == 0:
                model_name = os.path.join(model_path, f'{select_model}_epoch_{epoch}_bins_{bin_count}.pkl')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss
                }, model_name)
                print(f'[INFO] Saving weights as {model_name}')

    writer.flush()
    writer.close()



def parse_opt():
    parser = argparse.ArgumentParser(description='Regressor Model Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size')
    parser.add_argument('--alpha', type=float, default=0.6, help='Aplha default=0.6 DONT CHANGE')
    parser.add_argument('--w', type=float, default=0.4, help='w DONT CHANGE')
    parser.add_argument('--num_workers', type=int, default=2, help='Total # workers, for colab & kaggle use 2')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_epoch', type=int, default=10, help='Save model every # epochs')
    parser.add_argument('--train_path', type=str, default=ROOT / 'dataset/KITTI/training', help='Training path KITTI')
    parser.add_argument('--model_path', type=str, default=ROOT / 'weights', help='Weights path, for load and save model')
    parser.add_argument('--select_model', type=str, default='resnet18', help='Model selection: {resnet18, vgg11}')
    parser.add_argument('--api_key', type=str, default='', help='API key for comet.ml')
    parser.add_argument('--bin_count',type=int, default=2, help='bin ocunt, default 2')

    opt = parser.parse_args()

    return opt

def main(opt):
    train(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


