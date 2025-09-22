# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""
import argparse
import os
import sys
from pathlib import Path
import glob

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import LOGGER, check_img_size, check_requirements, non_max_suppression, print_args, scale_coords
from utils.torch_utils import select_device, time_sync

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg11

import numpy as np

from script.Dataset import generate_bins, DetectedObject
from library.Math import *
from library.Plotting import *
from script import Model, ClassAverages
from script.Model import ResNet, ResNet18, VGG11

# model factory to choose model
model_factory = {
    'resnet': resnet18(pretrained=True),
    'resnet18': resnet18(pretrained=True),
    # 'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_

def detect3d(
    bin_count,
    reg_weights,
    model_select,
    source,
    calib_file,
    show_result,
    save_result,
    output_path
    ):

    # Directory
    imgs_path = sorted(glob.glob(str(source) + '/*'))
    calib = str(calib_file)

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model, bins = bin_count).cuda()

    # load weight
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(bin_count)
    # loop images
    for i, img_path in enumerate(imgs_path):
        # read image
        img = cv2.imread(img_path)
        # Run detection 2d
        dets = detect2d(
            weights='yolov5s.pt',
            source=img_path,
            data='data/adam.yaml',
            imgsz=[640, 640],
            device=0,
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        )

        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue
            try: 
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class

            # ---------------------- SAVE input_img ----------------------
            debug_dir = "debug_inputs"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Convert Torch Tensor -> NumPy for saving
            if isinstance(input_img, torch.Tensor):
                img_np = input_img.detach().cpu().numpy()
                # If shape is (C, H, W), transpose to (H, W, C)
                if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                    img_np = np.transpose(img_np, (1, 2, 0))
                # Scale floats [0,1] -> [0,255]
                if img_np.dtype in [np.float32, np.float64]:
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            else:
                img_np = input_img  # already NumPy
            
            cv2.imwrite(f"{debug_dir}/{i:03d}_{detected_class}.png", img_np)
            # ------------------------------------------------------------

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img
            
            # predict orient, conf, and dim
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]

            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            alpha = -(alpha + np.pi*0.5)
            
            # plot 3d detection
            #plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)
            plot3d(img=img,proj_matrix=proj_matrix,box_2d=box_2d,dimensions=dim,alpha=theta_ray,theta_ray=0,pitch = alpha,roll = 0)
            
        if show_result:
            cv2.imshow('3d detection', img)
            cv2.waitKey(0)

        if save_result and output_path is not None:
            try:
                os.mkdir(output_path)
            except:
                pass
            cv2.imwrite(f'{output_path}/{i:03d}.png', img)

        # erstellen einer genormten onnx datei
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32).to(select_device(0))
        torch.onnx.export(
        regressor,
        dummy_input,
        "yolo3d.onnx",
        )

@torch.no_grad()
def detect2d(
    weights,
    source,
    data,
    imgsz,
    device,
    classes
    ):

    # array for boundingbox detection
    bbox_list = []

    # Directories
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(prediction=pred, classes=classes)
        
        dt[2] += time_sync() - t3
        #print(pred)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
                    bbox = [top_left, bottom_right]
                    c = int(cls)  # integer class
                    label = names[c]
                    
                    bbox_list.append(Bbox(bbox, label))

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    return bbox_list

def plot3d(
    img,
    proj_matrix,
    box_2d,
    dimensions,
    alpha,
    theta_ray,
    pitch=0,      # Add pitch parameter
    roll=0,       # Add roll parameter
    img_2d=None
    ):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    # Calculate overall orientation (yaw)
    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    # Plot 3D box with all three rotation angles
    plot_3d_box(img, proj_matrix, orient, dimensions, location, pitch, roll)

    return location

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=[0, 2, 3, 5], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default='weights/epoch_10.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--calib_file', type=str, default=ROOT / 'eval/camera_cal/calib_cam_to_cam.txt', help='Calibration file or path')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output pat')
    parser.add_argument('--bin_count', type=int, default=2, help='bin count, default 2')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    detect3d(
        bin_count = opt.bin_count,
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        calib_file=opt.calib_file,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
'''
if __name__ == "__main__":

    # ---------------------- Paths ----------------------

    image_path = 'eval/image_2/003005.png'
    calib_path = 'eval/camera_cal/calib_cam_to_cam.txt'
    output_path = '003005_gt_3dboxes.png'

    # ---------------------- All Labels ----------------------

    labels =  [
    {
        'class': 'small_black_alu',
        'bbox': [281.91575296549706, 204.12451106230787, 352.43799417983394, 265.75166482010604],
        'theta': -140.76540400116545,
        'size': [0.10000006854534149, 0.02000001072883606, 0.020000014454126358]
    },
{
    'class': 'allenkey',
    'bbox': [240.64705391270232, 124.2623810335368, 278.76826552418277, 288.2095000174795],
    'theta': -173.53126944982066,
    'size': [0.05100000649690628, 0.010004725307226181, 0.2120000272989273]
},
    {
        'class': 'screwdriver',
        'bbox': [380.65590962582644, 121.53221578986668, 440.7175203213312, 251.04972090344702],
        'theta': -113.03703529182643,
        'size': [0.18135032057762146, 0.026051077991724014, 0.026051077991724014]
    },
    {
        'class': 'large_black_alu',
        'bbox': [89.34135194433466, 41.78378380999722, 162.9927744982159, 129.237277413814],
        'theta': -58.62732489177631,
        'size': [0.09999991953372955, 0.039999961853027344, 0.03999996930360794]
    },
    {
        'class': 'motor2',
        'bbox': [333.51590816641624, 173.37613992082325, 372.28890836080126, 241.13076381003464],
        'theta': 97.46761224859318,
        'size': [0.08828440308570862, 0.044599998742341995, 0.043472860008478165]
    },
    {
        'class': 'wrench',
        'bbox': [235.64775175980458, 321.00598382476966, 327.2631656964064, 392.5695327445688],
        'theta': -143.7218650437838,
        'size': [0.14162303507328033, 0.019803566858172417, 0.005999989807605743]
    }
]


    # ---------------------- Load image ----------------------

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # ---------------------- Plot all objects ----------------------

    for label in labels:
        box_2d = [
            (int(label["bbox"][0]), int(label["bbox"][1])),
            (int(label["bbox"][2]), int(label["bbox"][3]))
        ]
        dimensions = np.array(label["size"])
        theta = np.radians(-(label["theta"]+90))

        # Get projection matrix and theta_ray
        detected_obj = DetectedObject(img, label["class"], box_2d, calib_path)
        proj_matrix = detected_obj.proj_matrix
        theta_ray = detected_obj.theta_ray

        # Plot box
        plot3d(
            img=img,
            proj_matrix=proj_matrix,
            box_2d=box_2d,
            dimensions=dimensions,
            alpha=theta_ray,
            theta_ray=0,
            pitch = theta,
            roll = 0
        )
    # ---------------------- Save result ----------------------

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved image with all 3D ground truth boxes to:\n{output_path}")
    '''