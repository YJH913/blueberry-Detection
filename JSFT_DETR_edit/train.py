import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from Hungrian_match import HungarianMatcher, SetCriterion
from MaskDataset import MaskDataset
from jfst_detr import JFSTDETR

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None
    _yaml_import_error = e


def _require_yaml():
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for config.yaml support. "
            "Install with: pip install pyyaml\n"
            f"Original import error: {_yaml_import_error}"
        )


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets


def get_image_paths(img_dir: str, exts=(".jpg", ".png", ".jpeg", ".bmp")) -> List[str]:
    return [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    ]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_device_info(device: torch.device):
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"using_gpu: cuda:{idx}  name={name}")
        print(f"capability={props.major}.{props.minor}  total_mem_gb={total_gb:.2f}")
        print(f"gpu_count={torch.cuda.device_count()}")
    else:
        print("using_cpu")


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def postprocess(pred_logits: torch.Tensor, pred_boxes: torch.Tensor, score_thresh: float):
    probs = F.softmax(pred_logits, dim=-1)
    scores, _ = probs[..., :-1].max(dim=-1)  # remove background
    keep = scores > score_thresh
    return pred_boxes[keep], scores[keep]


def calc_tp_fp_fn(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_thresh: float):
    if len(pred_boxes) == 0:
        return 0, 0, int(len(gt_boxes))

    ious = box_iou(pred_boxes, gt_boxes)
    tp = 0
    matched = set()

    for i in range(len(pred_boxes)):
        max_iou, idx = ious[i].max(dim=0)
        if max_iou >= iou_thresh and idx.item() not in matched:
            tp += 1
            matched.add(idx.item())

    fp = int(len(pred_boxes) - tp)
    fn = int(len(gt_boxes) - tp)
    return tp, fp, fn


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    use_amp: bool = True,
):
    model.eval()

    total_tp = total_fp = total_fn = 0
    precisions, recalls = [], []

    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)

        if device.type == "cuda" and use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
        else:
            outputs = model(imgs)

        pred_logits = outputs["pred_logits"].detach().cpu()
        pred_boxes = outputs["pred_boxes"].detach().cpu()

        for i in range(len(pred_logits)):
            pb, _ = postprocess(pred_logits[i], pred_boxes[i], score_thresh)
            gt = targets[i]["boxes"]

            pb = cxcywh_to_xyxy(pb)
            gt = cxcywh_to_xyxy(gt)

            tp, fp, fn = calc_tp_fp_fn(pb, gt, iou_thresh)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            p = tp / (tp + fp + 1e-6)
            r = tp / (tp + fn + 1e-6)
            precisions.append(p)
            recalls.append(r)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    map50 = compute_ap(np.array(recalls), np.array(precisions))
    return float(precision), float(recall), float(f1), float(map50)


@dataclass
class TrainStats:
    loss: float
    loss_ce: float
    loss_bbox: float
    loss_giou: float
    gt_per_img: float
    empty_imgs: int


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    accum_steps: int = 1,
    clip_norm: float = 1.0,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_ce = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    total_gt = 0
    empty_imgs = 0

    for step, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
            for t in targets
        ]

        n_gt = int(targets[0]["boxes"].shape[0])
        total_gt += n_gt
        if n_gt == 0:
            empty_imgs += 1

        if device.type == "cuda" and use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss, loss_dict = criterion(outputs, targets)
        else:
            outputs = model(imgs)
            loss, loss_dict = criterion(outputs, targets)

        # gradient accumulation
        loss_scaled = loss / max(accum_steps, 1)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        total_ce += float(loss_dict["loss_ce"].detach().item())
        total_bbox += float(loss_dict["loss_bbox"].detach().item())
        total_giou += float(loss_dict["loss_giou"].detach().item())

    n = max(len(train_loader), 1)
    return TrainStats(
        loss=total_loss / n,
        loss_ce=total_ce / n,
        loss_bbox=total_bbox / n,
        loss_giou=total_giou / n,
        gt_per_img=total_gt / n,
        empty_imgs=empty_imgs,
    )


def build_optimizer(model: torch.nn.Module, lr_backbone: float, lr: float, weight_decay: float):
    param_groups = [
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {
            "params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")],
            "lr": lr,
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

def load_config(path: str) -> Dict[str, Any]:
    _require_yaml()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must contain a mapping at top-level")
    return cfg


def get_cfg(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Fetch nested key like 'train.epochs' from dict.
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_history_csv(save_dir: str, history: List[Dict[str, Any]]):
    if not history:
        return
    keys = list(history[0].keys())
    out_path = os.path.join(save_dir, "history.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def save_history_json(save_dir: str, history: List[Dict[str, Any]]):
    out_path = os.path.join(save_dir, "history.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_history(save_dir: str, history: List[Dict[str, Any]]):
    if not history:
        return

    # headless backend
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Train losses
    ax = axes[0, 0]
    ax.plot(epochs, [h["train_loss"] for h in history], label="train_total")
    ax.plot(epochs, [h["train_ce"] for h in history], label="train_ce")
    ax.plot(epochs, [h["train_bbox"] for h in history], label="train_bbox")
    ax.plot(epochs, [h["train_giou"] for h in history], label="train_giou")
    ax.set_title("Training losses")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Test metrics
    ax = axes[0, 1]
    ax.plot(epochs, [h["test_map50"] for h in history], label="mAP@0.5")
    ax.plot(epochs, [h["test_f1"] for h in history], label="F1")
    ax.set_title("Test metrics")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Precision/Recall
    ax = axes[1, 0]
    ax.plot(epochs, [h["test_precision"] for h in history], label="Precision")
    ax.plot(epochs, [h["test_recall"] for h in history], label="Recall")
    ax.set_title("Test precision / recall")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Data sanity
    ax = axes[1, 1]
    ax.plot(epochs, [h["gt_per_img"] for h in history], label="GT/img")
    ax.plot(epochs, [h["empty_imgs"] for h in history], label="Empty imgs/epoch")
    ax.set_title("Data sanity")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(save_dir, "history.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path for eval-only (overrides config)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(get_cfg(cfg, "seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    save_dir = str(get_cfg(cfg, "save.dir", "./checkpoints"))
    ensure_dir(save_dir)

    img_dir = str(get_cfg(cfg, "data.img_dir"))
    mask_dir = str(get_cfg(cfg, "data.mask_dir"))
    img_size = int(get_cfg(cfg, "data.img_size", 640))
    train_ratio = float(get_cfg(cfg, "data.train_ratio", 0.7))
    test_ratio = float(get_cfg(cfg, "data.test_ratio", 0.2))
    val_ratio = float(get_cfg(cfg, "data.val_ratio", 0.1))

    all_img_paths = get_image_paths(img_dir)
    random.shuffle(all_img_paths)
    n = len(all_img_paths)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_val = max(0, n - n_train - n_test)

    train_img_paths = all_img_paths[:n_train]
    test_img_paths = all_img_paths[n_train : n_train + n_test]
    val_img_paths = all_img_paths[n_train + n_test :]

    print(f"Total images: {n}")
    print(f"Train images ({int(train_ratio*100)}%): {len(train_img_paths)}")
    print(f"Test images ({int(test_ratio*100)}%): {len(test_img_paths)}")
    print(f"Val  images ({int(val_ratio*100)}%): {len(val_img_paths)}")

    train_dataset = MaskDataset(train_img_paths, mask_dir, img_size=img_size, augment=True)
    test_dataset = MaskDataset(test_img_paths, mask_dir, img_size=img_size, augment=False)
    val_dataset = MaskDataset(val_img_paths, mask_dir, img_size=img_size, augment=False)

    batch_size = int(get_cfg(cfg, "train.batch_size", 1))
    num_workers = int(get_cfg(cfg, "train.num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = int(get_cfg(cfg, "model.num_classes", 1))
    hidden_dim = int(get_cfg(cfg, "model.hidden_dim", 128))
    num_queries = int(get_cfg(cfg, "model.num_queries", 50))
    backbone_pretrained = bool(get_cfg(cfg, "model.backbone_pretrained", False))
    model = JFSTDETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        backbone_pretrained=backbone_pretrained,
    ).to(device)

    matcher = HungarianMatcher(
        cost_class=float(get_cfg(cfg, "matcher.cost_class", 1.0)),
        cost_bbox=float(get_cfg(cfg, "matcher.cost_bbox", 5.0)),
        cost_giou=float(get_cfg(cfg, "matcher.cost_giou", 2.0)),
    )
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={
            "loss_ce": float(get_cfg(cfg, "loss.weights.loss_ce", 2.0)),
            "loss_bbox": float(get_cfg(cfg, "loss.weights.loss_bbox", 5.0)),
            "loss_giou": float(get_cfg(cfg, "loss.weights.loss_giou", 2.0)),
        },
        eos_coef=float(get_cfg(cfg, "loss.eos_coef", 0.1)),
    )

    lr_backbone = float(get_cfg(cfg, "optim.lr_backbone", 1e-5))
    lr = float(get_cfg(cfg, "optim.lr", 1e-4))
    weight_decay = float(get_cfg(cfg, "optim.weight_decay", 1e-4))
    optimizer = build_optimizer(model, lr_backbone, lr, weight_decay)

    use_amp = bool(get_cfg(cfg, "amp.enabled", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_amp))

    if args.eval_only:
        ckpt_path = args.ckpt or str(get_cfg(cfg, "eval.ckpt", ""))
        if not ckpt_path:
            raise ValueError("--eval-only requires --ckpt or eval.ckpt in config.yaml")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        precision, recall, f1, map50 = evaluate(
            model,
            test_loader,
            device,
            score_thresh=float(get_cfg(cfg, "eval.score_thresh", 0.5)),
            iou_thresh=float(get_cfg(cfg, "eval.iou_thresh", 0.5)),
            use_amp=use_amp,
        )
        print(f"[EVAL] Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} mAP@0.5={map50:.4f}")
        return

    best_metric = -1.0
    best_path = os.path.join(save_dir, "best.pth")
    last_path = os.path.join(save_dir, "last.pth")
    metric_name = str(get_cfg(cfg, "save.metric", "map50"))
    score_thresh = float(get_cfg(cfg, "eval.score_thresh", 0.5))
    iou_thresh = float(get_cfg(cfg, "eval.iou_thresh", 0.5))
    accum_steps = int(get_cfg(cfg, "train.accum_steps", 4))
    clip_norm = float(get_cfg(cfg, "train.clip_norm", 1.0))
    epochs = int(get_cfg(cfg, "train.epochs", 100))

    # record config for reproducibility
    with open(os.path.join(save_dir, "config.resolved.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            accum_steps=accum_steps,
            clip_norm=clip_norm,
            use_amp=use_amp,
            scaler=scaler,
        )

        # Validation set (구조상 70/10/20 중 10%에 해당)
        val_precision, val_recall, val_f1, val_map50 = evaluate(
            model,
            val_loader,
            device,
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            use_amp=use_amp,
        )

        # Test set (20%) - 여전히 best 모델 선정 기준으로 사용
        precision, recall, f1, map50 = evaluate(
            model,
            test_loader,
            device,
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            use_amp=use_amp,
        )

        metric_val = map50 if metric_name == "map50" else f1
        improved = metric_val > best_metric
        if improved:
            best_metric = metric_val

        print(
            f"[Epoch {epoch:03d}] "
            f"TrainLoss={stats.loss:.4f} (CE={stats.loss_ce:.3f}, BBox={stats.loss_bbox:.3f}, GIoU={stats.loss_giou:.3f}) | "
            f"GT/img={stats.gt_per_img:.2f}, Empty={stats.empty_imgs}/{len(train_loader)} | "
            f"VAL: P={val_precision:.4f} R={val_recall:.4f} F1={val_f1:.4f} mAP50={val_map50:.4f} | "
            f"TEST: P={precision:.4f} R={recall:.4f} F1={f1:.4f} mAP50={map50:.4f} "
            f"{'(BEST)' if improved else ''}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": stats.loss,
                "train_ce": stats.loss_ce,
                "train_bbox": stats.loss_bbox,
                "train_giou": stats.loss_giou,
                "gt_per_img": stats.gt_per_img,
                "empty_imgs": stats.empty_imgs,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_map50": val_map50,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_map50": map50,
                "metric_name": metric_name,
                "metric_val": metric_val,
                "best_metric": best_metric,
            }
        )
        save_history_csv(save_dir, history)
        save_history_json(save_dir, history)
        plot_history(save_dir, history)

        # 항상 last 저장
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "metric_name": metric_name,
                "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            },
            last_path,
        )

        # 테스트 성능 기준 best 저장
        if improved:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "metric_name": metric_name,
                    "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
                },
                best_path,
            )
            print(f"✅ Saved best checkpoint: {best_path} ({metric_name}={best_metric:.4f})")


if __name__ == "__main__":
    main()


