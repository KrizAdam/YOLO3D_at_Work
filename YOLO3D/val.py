'''
 python val.py   --reg_weights ./weights/resnet18_epoch_50_bins_8.pkl   --source ./eval/image_2   --calib_file ./eval/camera_cal/calib_cam_to_cam.txt   --model_select resnet18   --bin_count 8   --conf_thres 0.25   --save_dir runs/infer_eval
'''
import argparse
import os
import sys
from pathlib import Path
import glob

import cv2
import torch
import numpy as np
from tqdm import tqdm

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync

# 3D pipeline imports
from script.Dataset import generate_bins, DetectedObject
from script.Model import ResNet, ResNet18, VGG11
from script import ClassAverages
from library.Plotting import plot_3d_box, plot_2d_box
from metric import ConfusionMatrix3D, ap_per_class_3d, fitness_3d, box_iou_3d

# Model factories
from torchvision.models import resnet18, vgg11
model_factory = {
    'resnet18': resnet18(pretrained=True),
    'resnet': resnet18(pretrained=True),
    'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet18': ResNet18,
    'resnet': ResNet,
    'vgg11': VGG11
}


class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_


def detect2d(weights, source, data, imgsz, device, classes):
    """Run YOLOv5 2D detection."""
    bbox_list = []
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=model.pt)
    model.warmup(imgsz=(1, 3, *imgsz), half=False)

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device).float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, classes=classes)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xyxy = [int(x) for x in xyxy]
                    box_2d = [(xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])]
                    bbox_list.append(Bbox(box_2d, names[int(cls)]))

    return bbox_list


def run_inference_eval(
    source,
    calib_file,
    reg_weights,
    model_select='resnet18',
    bin_count=8,
    conf_thres=0.25,
    save_dir=Path("runs/infer_eval")
):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load regressor
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model, bins=bin_count).cuda()
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(bin_count)

    # Metrics
    iouv = torch.linspace(0.25, 0.75, 5).cuda()
    confusion_matrix = ConfusionMatrix3D(nc=len(averages.dimension_map.keys()), conf=conf_thres, iou_thres=0.5)
    stats, seen = [], 0

    imgs_path = sorted(glob.glob(str(source) + '/*'))
    pbar = tqdm(imgs_path, desc="Inference+Eval")
    class_to_id = {cls: i for i, cls in enumerate(averages.dimension_map.keys())}
    id_to_class = {i: cls for i, cls in enumerate(averages.dimension_map.keys())}

    for i, img_path in enumerate(pbar):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Run YOLO detection
        dets = detect2d(weights='yolov5s.pt', source=img_path, data='data/adam.yaml', imgsz=[640, 640], device=0,
                        classes=list(range(len(averages.dimension_map.keys()))))

        detections_list, labels_list = [], []

        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue

            try:
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib_file)
            except:
                continue

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = detectedObject.img
            
            [orient, conf, dim] = regressor(input_tensor)

            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(det.detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            alpha = np.arctan2(orient[1], orient[0]) + angle_bins[argmax] - np.pi


            # Create detection [x,y,z,w,h,l,yaw,conf,class]
            detection = torch.tensor([
                0, 0, 0,
                dim[0], dim[1], dim[2],
                alpha,
                float(np.max(conf)),
                class_to_id[det.detected_class.lower()]

            ], device='cuda')
            detections_list.append(detection)

            # Ground truth (mock: you need YAML loader here)
            gt_label = torch.tensor([
                class_to_id[det.detected_class.lower()],
                0, 0, 0,
                dim[0], dim[1], dim[2],
                alpha
            ], device='cuda')
            labels_list.append(gt_label)

        detections = torch.stack(detections_list) if detections_list else torch.empty((0, 9), device='cuda')
        gt_labels = torch.stack(labels_list) if labels_list else torch.empty((0, 8), device='cuda')

        # IoU matching
        if detections.numel() and gt_labels.numel():
            iou = box_iou_3d(gt_labels[:, 1:8], detections[:, :7])
            correct = iou.max(1).values.unsqueeze(1) >= iouv
            stats.append((correct.cpu(), detections[:, 7].cpu(), detections[:, 8].cpu(), gt_labels[:, 0].cpu()))

            confusion_matrix = ConfusionMatrix3D(
    nc=len(averages.dimension_map.keys()), 
    conf=conf_thres, 
    iou_thres=0.5)
            
        seen += 1

    # Compute metrics
    if len(stats):
        stats = [torch.cat(x, 0).numpy() for x in zip(*stats)]
        tp, fp, p, r, f1, ap, ap_class = ap_per_class_3d(*stats, names=id_to_class)
        mp, mr, map25, map50, map75 = p.mean(), r.mean(), ap[:, 0].mean(), ap[:, 1].mean(), ap[:, 2].mean()
    else:
        mp = mr = map25 = map50 = map75 = 0.0

    fitness = fitness_3d(np.array([[mp, mr, map25, map50, map75]]))[0]

    print(f"Results: P={mp:.3f}, R={mr:.3f}, mAP@0.25={map25:.3f}, mAP@0.5={map50:.3f}, mAP@0.75={map75:.3f}")
    print(f"Fitness: {fitness:.4f}")

    return {
        'precision': mp,
        'recall': mr,
        'mAP@0.25': map25,
        'mAP@0.5': map50,
        'mAP@0.75': map75,
        'fitness': fitness
    }


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='eval/image_2', help='image folder')
    parser.add_argument('--calib_file', type=str, default='eval/camera_cal/calib_cam_to_cam.txt')
    parser.add_argument('--reg_weights', type=str, default='weights/epoch_10.pkl')
    parser.add_argument('--model_select', type=str, default='resnet18')
    parser.add_argument('--bin_count', type=int, default=8)
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--save_dir', type=str, default='runs/infer_eval')
    return parser.parse_args()


def main(opt):
    results = run_inference_eval(
        source=opt.source,
        calib_file=opt.calib_file,
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        bin_count=opt.bin_count,
        conf_thres=opt.conf_thres,
        save_dir=Path(opt.save_dir)
    )
    print(results)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
