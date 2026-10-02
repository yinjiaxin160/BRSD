
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.special import expit

def _safe_div(a, b):
    return a / b if b != 0 else 0.0


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()

    return _safe_div(2.0 * inter, union)


def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    return _safe_div(inter, union)


def accuracy_score(pred, gt):
    return (pred == gt).mean()


def sensitivity_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    return _safe_div(tp, tp + fn)


def specificity_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()

    return _safe_div(tn, tn + fp)


def hd95_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_points = np.argwhere(pred)
    gt_points = np.argwhere(gt)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return 999.0

    d1 = directed_hausdorff(pred_points, gt_points)[0]
    d2 = directed_hausdorff(gt_points, pred_points)[0]

    return max(d1, d2)


def numpy_softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def compute_metrics_from_logits(logits, masks, threshold=0.5):
    """
    logits: numpy array, shape [N, C, H, W] or [N, H, W]
    masks: numpy array, shape [N, 1, H, W] or [N, H, W]
    """
    
    
    if logits.ndim == 4 and logits.shape[1] > 1:
  
        probs_all = numpy_softmax(logits, axis=1)
        preds_class = np.argmax(probs_all, axis=1)  # shape: [N, H, W]
        
       
        preds = np.where(preds_class > 0, 1, 0).astype(np.uint8)
        
       
        probs_for_conf = np.max(probs_all, axis=1)  # shape: [N, H, W]
        
    else:
      
        probs_all = expit(logits)
        if probs_all.ndim == 4:
            probs_all = probs_all[:, 0]  # shape: [N, H, W]
            
        preds = (probs_all > threshold).astype(np.uint8)
        
       
        probs_for_conf = np.maximum(probs_all, 1.0 - probs_all)


    if masks.ndim == 4:
        masks = masks[:, 0]
        
  
    gts = np.where(masks > 0, 1, 0).astype(np.uint8)

    dices = []
    ious = []
    accs = []
    sens = []
    specs = []
    hd95s = []
    confs = []

    for pred, gt, prob in zip(preds, gts, probs_for_conf):
        # 【核心修改】：如果这张切片上根本没有肿瘤，直接跳过，不计入平均分！
        if gt.sum() == 0:
            continue

        dices.append(dice_score(pred, gt))
        ious.append(iou_score(pred, gt))
        accs.append(accuracy_score(pred, gt))
        sens.append(sensitivity_score(pred, gt))
        specs.append(specificity_score(pred, gt))
        hd95s.append(hd95_score(pred, gt))
        confs.append(np.mean(prob))

    # 【核心修改】：防止测试集全是空切片导致除以 0 报错
    if len(dices) == 0:
        return {
            "mIoU": 0.0,
            "Dice": 0.0,
            "Accuracy": 0.0,
            "Sensitivity": 0.0,
            "Specificity": 0.0,
            "HD95": 0.0,
            "Confidence": 0.0,
        }

    return {
        "mIoU": float(np.mean(ious)),
        "Dice": float(np.mean(dices)),
        "Accuracy": float(np.mean(accs)),
        "Sensitivity": float(np.mean(sens)),
        "Specificity": float(np.mean(specs)),
        "HD95": float(np.mean(hd95s)),
        "Confidence": float(np.mean(confs)),
    }


def multiclass_dice_score(logits, masks, num_classes=4, ignore_background=True):

    probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
    
    dice_scores = []
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        pred_c = probs[:, c, :, :]  # [B, H, W]
        target_c = (masks == c).float()  # [B, H, W]
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        cardinality = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        
        dice_c = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
        dice_scores.append(dice_c.mean().item())
    
    if ignore_background:
        return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return sum(dice_scores) / num_classes


def multiclass_mean_iou(logits, masks, num_classes=4):
    """
    Multi-class mIoU (PyTorch tensor version)
    """
    preds = torch.argmax(logits, dim=1)  # [B, H, W]
    
    ious = []
    for c in range(1, num_classes):  # 忽略背景
        pred_c = (preds == c)
        target_c = (masks == c)
        
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        
        iou_c = (intersection.float() + 1e-5) / (union.float() + 1e-5)
        ious.append(iou_c.item())
    
    return sum(ious) / len(ious) if ious else 0.0