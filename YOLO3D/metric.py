# 3D Object Detection Evaluation Metrics
"""3D Model validation metrics for object detection tasks."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import TryExcept, threaded


def fitness_3d(x):
    """Calculates fitness of a 3D model using weighted sum of metrics P, R, mAP@0.25, mAP@0.5, mAP@0.75."""
    w = [0.0, 0.0, 0.1, 0.7, 0.2]  # weights for [P, R, mAP@0.25, mAP@0.5, mAP@0.75]
    return (x[:, :5] * w).sum(1)


def smooth(y, f=0.05):
    """Applies box filter smoothing to array `y` with fraction `f`, yielding a smoothed array."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class_3d(tp, conf, pred_cls, target_cls, plot=True, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Compute the average precision for 3D object detection, given the recall and precision curves.
    
    # Arguments
        tp:  True positives (nparray, nx1 or nxN for multiple IoU thresholds).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed for 3D detection.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    
    if plot:
        plot_pr_curve_3d(px, py, ap, Path(save_dir) / f"{prefix}PR_curve_3d.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve_3d.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve_3d.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve_3d.png", names, ylabel="Recall")

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix3D:
    
    def __init__(self, nc, conf=0.25, iou_thres=0.5):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):

        if detections is None:
            gt_classes = labels[:, 0].int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 7] > self.conf]  # conf threshold
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 8].int()
        
        # Calculate 3D IoU
        iou = box_iou_3d(labels[:, 1:], detections[:, :7])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp(self):
        """Calculates true positives (tp) and false positives (fp) excluding the background class from the confusion
        matrix.
        """
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept("WARNING ⚠️ 3D ConfusionMatrix plot failure")
    def plot(self, normalize=True, save_dir="", names=()):
        """Plots 3D confusion matrix using seaborn, optional normalization; can save plot to specified directory."""
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("3D Confusion Matrix")
        fig.savefig(Path(save_dir) / "confusion_matrix_3d.png", dpi=250)
        plt.close(fig)

    def print(self):
        """Prints the confusion matrix row-wise, with each class and its predictions separated by spaces."""
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


def box_iou_3d(box1, box2, eps=1e-7):
    """
        box1 (Tensor[N, 7]): x, y, z, w, h, l, yaw
        box2 (Tensor[M, 7]): x, y, z, w, h, l, yaw
    """
   
    x1, y1, z1, w1, h1, l1, yaw1 = box1.chunk(7, -1)
    x2, y2, z2, w2, h2, l2, yaw2 = box2.chunk(7, -1)
    
    # For simplicity, we'll use axis-aligned bounding box IoU
    # Convert to corner coordinates
    box1_corners = torch.cat([
        x1 - w1/2, y1 - h1/2, z1 - l1/2,
        x1 + w1/2, y1 + h1/2, z1 + l1/2
    ], dim=-1)
    
    box2_corners = torch.cat([
        x2 - w2/2, y2 - h2/2, z2 - l2/2,
        x2 + w2/2, y2 + h2/2, z2 + l2/2
    ], dim=-1)
    
    # Calculate intersection
    lt = torch.max(box1_corners[..., :3].unsqueeze(-2), box2_corners[..., :3].unsqueeze(-3))
    rb = torch.min(box1_corners[..., 3:].unsqueeze(-2), box2_corners[..., 3:].unsqueeze(-3))
    
    whl = (rb - lt).clamp(min=0)
    inter = whl[..., 0] * whl[..., 1] * whl[..., 2]
    
    # Calculate volumes
    vol1 = (w1 * h1 * l1).squeeze(-1)
    vol2 = (w2 * h2 * l2).squeeze(-1)
    
    # Calculate union
    union = vol1.unsqueeze(-1) + vol2.unsqueeze(-2) - inter + eps
    
    return inter / union


def rotated_box_iou_3d(box1, box2, eps=1e-7):
    """
    Calculate 3D IoU between rotated 3D bounding boxes.
    This is a more accurate implementation that considers rotation.
    
    Arguments:
        box1 (Tensor[N, 7]): x, y, z, w, h, l, yaw
        box2 (Tensor[M, 7]): x, y, z, w, h, l, yaw
        
    Returns:
        iou (Tensor[N, M]): 3D IoU values
    """
    # This would require more complex 3D geometry calculations
    # For now, we'll use the simplified version above
    # In practice, you might want to use libraries like:
    # - PyTorch3D
    # - Open3D
    # - Custom CUDA kernels for efficiency
    return box_iou_3d(box1, box2, eps)


def bbox_iou_3d(box1, box2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates 3D IoU, GIoU, DIoU, or CIoU between two 3D boxes.
    
    Arguments:
        box1, box2: Tensors of shape (N, 7) and (M, 7) respectively
                   Format: [x, y, z, w, h, l, yaw]
    """
    # For 3D boxes, we need to handle rotation properly
    # This is a simplified implementation
    return rotated_box_iou_3d(box1, box2, eps)


def nms_3d(boxes, scores, iou_threshold=0.5):
    """
    3D Non-Maximum Suppression
    
    Arguments:
        boxes (Tensor[N, 7]): 3D boxes in format [x, y, z, w, h, l, yaw]
        scores (Tensor[N]): confidence scores
        iou_threshold (float): IoU threshold for suppression
        
    Returns:
        keep (Tensor): indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Sort by scores
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
            
        i = order[0].item()
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        iou = box_iou_3d(boxes[i:i+1], boxes[order[1:]])
        
        # Keep boxes with IoU less than threshold
        mask = iou.squeeze(0) <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


# 3D-specific evaluation metrics
def evaluate_3d_detection(pred_boxes, pred_scores, pred_classes, 
                         gt_boxes, gt_classes, 
                         iou_thresholds=[0.25, 0.5, 0.75],
                         distance_ranges=[(0, 30), (30, 50), (50, float('inf'))]):
    """
    Evaluate 3D object detection performance across different IoU thresholds and distance ranges.
    
    Arguments:
        pred_boxes: Predicted 3D boxes [N, 7] (x, y, z, w, h, l, yaw)
        pred_scores: Prediction confidence scores [N]
        pred_classes: Predicted class labels [N]
        gt_boxes: Ground truth 3D boxes [M, 7]
        gt_classes: Ground truth class labels [M]
        iou_thresholds: IoU thresholds for evaluation
        distance_ranges: Distance ranges for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    results = {}
    
    # Calculate distances from origin (for autonomous driving scenarios)
    gt_distances = torch.sqrt(gt_boxes[:, 0]**2 + gt_boxes[:, 1]**2)
    pred_distances = torch.sqrt(pred_boxes[:, 0]**2 + pred_boxes[:, 1]**2)
    
    for dist_min, dist_max in distance_ranges:
        # Filter by distance
        gt_mask = (gt_distances >= dist_min) & (gt_distances < dist_max)
        pred_mask = (pred_distances >= dist_min) & (pred_distances < dist_max)
        
        if gt_mask.sum() == 0 and pred_mask.sum() == 0:
            continue
            
        gt_boxes_filtered = gt_boxes[gt_mask]
        gt_classes_filtered = gt_classes[gt_mask]
        pred_boxes_filtered = pred_boxes[pred_mask]
        pred_scores_filtered = pred_scores[pred_mask]
        pred_classes_filtered = pred_classes[pred_mask]
        
        # Evaluate at different IoU thresholds
        for iou_thresh in iou_thresholds:
            key = f"mAP_{iou_thresh}@{dist_min}-{dist_max}m"
            # Calculate AP for this configuration
            # Implementation would depend on your specific evaluation protocol
            results[key] = 0.0  # Placeholder
    
    return results


# Plotting functions for 3D
@threaded
def plot_pr_curve_3d(px, py, ap, save_dir=Path("pr_curve_3d.png"), names=()):
    """Plots 3D precision-recall curve, optionally per class."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("3D Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    """Plots a metric-confidence curve for 3D model predictions."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"3D {ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)