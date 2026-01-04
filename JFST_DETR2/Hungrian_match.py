import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h
    ], dim=-1)

def sanitize_boxes(boxes, eps=1e-6):
    if boxes.numel() == 0:
        return boxes
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    cx, cy, w, h = boxes.unbind(-1)
    cx = cx.clamp(0.0, 1.0)
    cy = cy.clamp(0.0, 1.0)
    w = w.clamp(min=eps, max=1.0)
    h = h.clamp(min=eps, max=1.0)
    return torch.stack([cx, cy, w, h], dim=-1)

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_logits = torch.nan_to_num(
            outputs["pred_logits"], nan=0.0, posinf=0.0, neginf=0.0
        )
        out_prob = out_logits.softmax(-1)
        out_bbox = sanitize_boxes(outputs["pred_boxes"])

        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = sanitize_boxes(targets[b]["boxes"])

            if tgt_bbox.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long)
                ))
                continue

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_bbox)
            )

            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            
            # Replace NaN/Inf with large values
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=1e6)
            C = C.cpu()

            i, j = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(i, dtype=torch.long),
                torch.as_tensor(j, dtype=torch.long)
            ))

        return indices

class RTDETRDetection(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_classes=1,
        num_queries=100,
        num_heads=8,
        num_decoder_layers=1
    ):
        super().__init__()

        # IoU-aware score head
        self.score_head = nn.Linear(hidden_dim, 1)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Prediction heads
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

        self.num_queries = num_queries

    def forward(self, memory):
        """
        memory: (B, N, C)  ← Hybrid Encoder output (flattened)
        """
        B, N, C = memory.shape

        # IoU-aware scoring
        scores = self.score_head(memory).squeeze(-1)  # (B, N)

        # Top-K query selection
        topk_idx = scores.topk(self.num_queries, dim=1).indices
        topk_idx = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        queries = torch.gather(memory, 1, topk_idx)

        # Decoder
        hs = self.decoder(queries, memory)

        return {
            "pred_logits": self.cls_head(hs),
            "pred_boxes": self.box_head(hs)
        }


def get_src_permutation_idx(indices):
    batch_idx = torch.cat([
        torch.full_like(src, i)
        for i, (src, _) in enumerate(indices)
    ])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        # --- Classification loss ---
        src_logits = torch.nan_to_num(
            outputs["pred_logits"], nan=0.0, posinf=0.0, neginf=0.0
        )
        idx = get_src_permutation_idx(indices)

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=src_logits.device
        )
        target_classes[idx] = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes
        )

        # --- Box loss ---
        src_boxes = sanitize_boxes(outputs["pred_boxes"][idx])
        tgt_boxes = sanitize_boxes(torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        ))

        if tgt_boxes.numel() == 0 or src_boxes.numel() == 0:
            loss_bbox = src_boxes.sum() * 0.0
            loss_giou = src_boxes.sum() * 0.0
        else:
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none")
            loss_bbox = loss_bbox.sum() / max(tgt_boxes.shape[0], 1)

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(tgt_boxes)
            )
            giou = torch.nan_to_num(giou, nan=0.0, posinf=0.0, neginf=0.0)
            loss_giou = 1 - torch.diag(giou)
            loss_giou = loss_giou.sum() / max(tgt_boxes.shape[0], 1)

        total_loss = (
            self.weight_dict["loss_ce"] * loss_ce +
            self.weight_dict["loss_bbox"] * loss_bbox +
            self.weight_dict["loss_giou"] * loss_giou
        )

        return total_loss, {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }
