import torch
import numpy as np

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    lr = init_lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def dice_iou(pred, gt):
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    inter = (pred_bin * gt_bin).sum()
    dice = 2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-8)
    iou = inter / (pred_bin.sum() + gt_bin.sum() - inter + 1e-8)
    return dice, iou

def cal_mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def cal_em(pred, gt):
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    inter = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter
    if union == 0:
        return 0.0
    return inter / union

def cal_sm(pred, gt):
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    tp = (pred_bin * gt_bin).sum()
    precision = tp / (pred_bin.sum() + 1e-8)
    recall = tp / (gt_bin.sum() + 1e-8)
    f_beta = (1+0.3)*precision*recall / (0.3*precision + recall + 1e-8)
    return f_beta

def cal_wfm(pred, gt):
    return cal_sm(pred, gt)