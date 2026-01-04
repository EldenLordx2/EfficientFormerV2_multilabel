# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
import util.utils as utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if True:  # with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import torch

def _average_precision_per_class(y_true, y_score, eps=1e-12):
    """
    y_true: [N] 0/1
    y_score: [N] float
    return: AP (scalar tensor)
    """
    # sort by score desc
    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]

    # if no positive, AP undefined; return nan (later ignore) or 0
    pos = y_true.sum()
    if pos.item() == 0:
        return torch.tensor(float('nan'), device=y_true.device)

    tp = torch.cumsum(y_true, dim=0)
    fp = torch.cumsum(1 - y_true, dim=0)

    precision = tp / (tp + fp + eps)
    # AP = sum precision@k for each true positive / #pos
    ap = (precision * y_true).sum() / (pos + eps)
    return ap

@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    all_probs = []
    all_targets = []

    bce = torch.nn.BCEWithLogitsLoss()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast():
            output = model(images)
            if not isinstance(output, torch.Tensor):
                output = output[0]
            loss = bce(output, target)

        prob = torch.sigmoid(output)

        bs = images.size(0)
        metric_logger.update(loss=loss.item())

        all_probs.append(prob.detach())
        all_targets.append(target.detach())

    metric_logger.synchronize_between_processes()

    # ---- 下面开始算全量指标（需要把所有 batch 拼起来）----
    probs = torch.cat(all_probs, dim=0)      # [N, C]
    targets = torch.cat(all_targets, dim=0)  # [N, C]

    # 1) mAP
    aps = []
    C = targets.size(1)
    for c in range(C):
        ap_c = _average_precision_per_class(targets[:, c], probs[:, c])
        aps.append(ap_c)
    aps = torch.stack(aps)  # [C]
    mAP = torch.nanmean(aps).item()

    # 2) 选阈值：在验证集上扫一遍，最大化 micro-F1（你也可以改成 macro-F1）
    thresholds = torch.linspace(0.05, 0.95, steps=19, device=probs.device)

    best = {"thr": 0.5, "micro_f1": -1, "micro_p": 0, "micro_r": 0, "macro_f1": -1}

    for thr in thresholds:
        pred = (probs >= thr).float()

        tp = (pred * targets).sum()
        fp = (pred * (1 - targets)).sum()
        fn = ((1 - pred) * targets).sum()

        micro_p = (tp / (tp + fp + 1e-12)).item()
        micro_r = (tp / (tp + fn + 1e-12)).item()
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r + 1e-12))

        # macro-F1: per class then average
        tp_c = (pred * targets).sum(dim=0)
        fp_c = (pred * (1 - targets)).sum(dim=0)
        fn_c = ((1 - pred) * targets).sum(dim=0)

        p_c = tp_c / (tp_c + fp_c + 1e-12)
        r_c = tp_c / (tp_c + fn_c + 1e-12)
        f1_c = 2 * p_c * r_c / (p_c + r_c + 1e-12)
        macro_f1 = f1_c.mean().item()

        if micro_f1 > best["micro_f1"]:
            best.update({
                "thr": float(thr.item()),
                "micro_f1": float(micro_f1),
                "micro_p": float(micro_p),
                "micro_r": float(micro_r),
                "macro_f1": float(macro_f1),
            })

    # 输出
    loss_avg = metric_logger.loss.global_avg if "loss" in metric_logger.meters else float("nan")
    print(
        f"* loss {loss_avg:.4f} | mAP {mAP:.4f} | "
        f"best_thr {best['thr']:.2f} | micro-F1 {best['micro_f1']:.4f} "
        f"(P {best['micro_p']:.4f}, R {best['micro_r']:.4f}) | "
        f"macro-F1 {best['macro_f1']:.4f}"
    )

    return {
        "loss": loss_avg,
        "mAP": mAP,
        "best_thr": best["thr"],
        "micro_f1": best["micro_f1"],
        "micro_p": best["micro_p"],
        "micro_r": best["micro_r"],
        "macro_f1": best["macro_f1"],
    }

# engine.py 末尾新增
import os

@torch.no_grad()
def predict_txt_images(data_loader, model, device, label_txt, out_vec_path, out_label_path, thr=0.5):
    model.eval()

    # 读 label 名
    labels = []
    with open(label_txt, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 允许 "0 正常" 这种：取第2列及之后作为名字；没有则取第1列
            parts = s.split()
            if len(parts) >= 2:
                name = " ".join(parts[1:])
            else:
                name = parts[0]
            labels.append(name)

    os.makedirs(os.path.dirname(out_vec_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_label_path) or ".", exist_ok=True)

    fvec = open(out_vec_path, "w", encoding="utf-8")
    flab = open(out_label_path, "w", encoding="utf-8")

    for images, paths in data_loader:
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            if not isinstance(output, torch.Tensor):
                output = output[0]

        probs = torch.sigmoid(output)  # [B, C]
        preds = (probs >= thr).int().cpu().numpy()  # 0/1

        # DataLoader 默认 collate 会把 path list 成 list[str]
        for p, vec in zip(paths, preds):
            vec_str = ",".join(str(int(x)) for x in vec.tolist())
            fvec.write(f"{p} {vec_str}\n")

            picked = [labels[i] for i, v in enumerate(vec.tolist()) if v == 1 and i < len(labels)]
            flab.write(f"{p} {','.join(picked)}\n")

    fvec.close()
    flab.close()

    return {"out_vec": out_vec_path, "out_labels": out_label_path, "thr": thr, "num_labels": len(labels)}
