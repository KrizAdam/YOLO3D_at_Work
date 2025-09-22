"""
Script for training Regressor Model
python train_neu_mit_eval.py     --epochs 60     --batch_size 128     --num_workers 3     --save_epoch 10     --train_path ./data/daten     --model_path ./weights     --select_model resnet18     --api_key KfPzkJBufCei2Nko40NoOTYNy --bin_count 8
"""


import argparse
import os
from pathlib import Path
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from comet_ml import Experiment
from script.Dataset import Dataset
from script.Model import ResNet18, VGG11, OrientationLoss
from torchvision.models import resnet18, vgg11

writer = SummaryWriter()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# -----------------------
# Model factory
# -----------------------
model_factory = {
    'resnet18': resnet18(pretrained=True),
    'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet18': ResNet18,
    'vgg11': VGG11
}

# -----------------------
# 3D IoU axis-aligned, vectorized
# -----------------------
def box_iou_3d(box1, box2, eps=1e-7):
    x1,y1,z1,w1,h1,l1,_ = box1.chunk(7, dim=-1)
    x2,y2,z2,w2,h2,l2,_ = box2.chunk(7, dim=-1)

    box1_corners = torch.cat([x1-w1/2, y1-h1/2, z1-l1/2,
                              x1+w1/2, y1+h1/2, z1+l1/2], dim=-1)
    box2_corners = torch.cat([x2-w2/2, y2-h2/2, z2-l2/2,
                              x2+w2/2, y2+h2/2, z2+l2/2], dim=-1)

    lt = torch.max(box1_corners[...,:3].unsqueeze(-2), box2_corners[...,:3].unsqueeze(-3))
    rb = torch.min(box1_corners[...,3:].unsqueeze(-2), box2_corners[...,3:].unsqueeze(-3))
    whl = (rb - lt).clamp(min=0)
    inter = whl[...,0]*whl[...,1]*whl[...,2]

    vol1 = (w1*h1*l1).squeeze(-1)
    vol2 = (w2*h2*l2).squeeze(-1)
    union = vol1.unsqueeze(-1) + vol2.unsqueeze(-2) - inter + eps
    return inter / union

# -----------------------
# Decode yaw
# -----------------------
def decode_yaw(orient, conf, bin_count):
    bin_idx = torch.argmax(conf, dim=1)
    idx = bin_idx.view(-1,1,1).expand(-1,1,orient.size(2))
    chosen_orient = orient.gather(1, idx).squeeze(1)
    residual = torch.atan2(chosen_orient[:,1], chosen_orient[:,0])
    angle_per_bin = 2 * torch.pi / bin_count
    yaw = bin_idx * angle_per_bin + residual
    return yaw

# -----------------------
# Average Precision
# -----------------------
def average_precision(recall, precision, method="interp"):
    recall = np.array(recall)
    precision = np.array(precision)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    if method=="interp":
        x = np.linspace(0,1,101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i+1]-mrec[i]) * mpre[i+1])
    return ap, mpre, mrec

# -----------------------
# Fitness function
# -----------------------
def fitness_3d(x):
    w = [0.0, 0.0, 0.1, 0.7, 0.2]
    return (x[:,:5] * torch.tensor(w, device=x.device)).sum(1)

# -----------------------
# Training
# -----------------------
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
    train_path = str(train_path)
    model_path = str(model_path)

    print('[INFO] Loading dataset...')
    dataset = Dataset(train_path, bins=bin_count)

    hyper_params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'w': w,
        'num_workers': num_workers,
        'lr': lr,
        'shuffle': True
    }

    experiment = Experiment(api_key, project_name="YOLO3D")
    experiment.log_parameters(hyper_params)

    data_gen = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    base_model = model_factory[select_model]
    model = regressor_factory[select_model](model=base_model, bins=bin_count).cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    # Load previous weights if exist
    latest_model = None
    first_epoch = 1
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except: pass

    if latest_model:
        checkpoint = torch.load(os.path.join(model_path, latest_model))
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'[INFO] Loaded {latest_model} from epoch {first_epoch}, resuming training...')

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(first_epoch, epochs+1):
        with tqdm(data_gen, unit='batch') as tepoch:
            for local_batch, local_labels in tepoch:
                tepoch.set_description(f'Epoch {epoch}')

                local_batch = local_batch.float().cuda()
                truth_orient = local_labels['Orientation'].float().cuda()
                truth_dim = local_labels['Dimensions'].float().cuda()
                truth_loc = local_labels['Location'].float().cuda() if 'Location' in local_labels else torch.zeros(len(truth_dim),3).cuda()

                # single dummy bin for labels
                truth_conf = torch.ones(len(truth_dim), dtype=torch.float32).cuda()
                truth_conf_idx = torch.zeros(len(truth_dim), dtype=torch.long).cuda()

                # forward
                orient, conf, dim = model(local_batch)

                # loss
                orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
                dim_loss = dim_loss_func(dim, truth_dim)
                conf_loss = conf_loss_func(conf, truth_conf_idx)
                loss_theta = conf_loss + w * orient_loss
                loss = alpha * dim_loss + loss_theta

                # metrics
                pred_yaw = decode_yaw(orient, conf, bin_count)
                gt_yaw = torch.atan2(truth_orient[:,1], truth_orient[:,0])

                pred_boxes = torch.cat([truth_loc, dim, pred_yaw.unsqueeze(-1)], dim=-1)
                gt_boxes = torch.cat([truth_loc, truth_dim, gt_yaw.unsqueeze(-1)], dim=-1)

                ious = box_iou_3d(pred_boxes, gt_boxes)
                mean_iou = ious.diagonal().mean().item()

                dim_error = torch.mean(torch.abs(dim - truth_dim)).item()
                depth_error = torch.mean(torch.abs(dim[:,2] - truth_dim[:,2])).item()
                aos = torch.mean(torch.cos(pred_yaw - gt_yaw)).item()

                # AP at IoU 0.5 (example)
                iou_thresh = 0.5
                tp = (ious.diagonal().cpu().numpy() >= iou_thresh).astype(float)
                fp = 1 - tp
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / len(gt_boxes)
                precision = tp_cum / (tp_cum + fp_cum + 1e-7)
                ap, _, _ = average_precision(recall, precision)

                # fitness
                metrics_tensor = torch.tensor([[0,0,0,mean_iou,ap]], device=dim.device)
                fitness = fitness_3d(metrics_tensor).item()

                # Logging
                writer.add_scalar('Loss/train', loss, epoch)
                writer.add_scalar('IoU3D/train', mean_iou, epoch)
                writer.add_scalar('DimError/train', dim_error, epoch)
                writer.add_scalar('DepthError/train', depth_error, epoch)
                writer.add_scalar('AOS/train', aos, epoch)
                writer.add_scalar('AP/train', ap, epoch)
                writer.add_scalar('Fitness/train', fitness, epoch)

                experiment.log_metric('Loss/train', loss, epoch=epoch)
                experiment.log_metric('IoU3D/train', mean_iou, epoch=epoch)
                experiment.log_metric('DimError/train', dim_error, epoch=epoch)
                experiment.log_metric('DepthError/train', depth_error, epoch=epoch)
                experiment.log_metric('AOS/train', aos, epoch=epoch)
                experiment.log_metric('AP/train', ap, epoch=epoch)
                experiment.log_metric('Fitness/train', fitness, epoch=epoch)

                opt_SGD.zero_grad()
                loss.backward()
                opt_SGD.step()

                tepoch.set_postfix(loss=loss.item())

        # save model
        if epoch % save_epoch == 0:
            model_name = os.path.join(model_path, f'{select_model}_epoch_{epoch}_bins_{bin_count}.pkl')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt_SGD.state_dict(),
                'loss': loss
            }, model_name)
            print(f'[INFO] Saved weights to {model_name}')

    writer.flush()
    writer.close()


# -----------------------
# argparse
# -----------------------
def parse_opt():
    parser = argparse.ArgumentParser(description='Regressor Model Training')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--w', type=float, default=0.4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--train_path', type=str, default=ROOT / 'dataset/KITTI/training')
    parser.add_argument('--model_path', type=str, default=ROOT / 'weights')
    parser.add_argument('--select_model', type=str, default='resnet18')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--bin_count',


def main(opt):
    train(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
