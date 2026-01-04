import torch
import torch.nn as nn

class RTDETRDetection(nn.Module):
    """
    IoU-aware Query Selection + Transformer Decoder + Detection Head
    (RT-DETR / JFST-DETR 공통)
    """
    def __init__(
        self,
        hidden_dim=256,
        num_classes=1,
        num_queries=100,
        num_heads=8,
        num_decoder_layers=1
    ):
        super().__init__()

        # 1. IoU-aware score head
        self.score_head = nn.Linear(hidden_dim, 1)

        # 2. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # 3. Detection heads
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
        memory: (B, N, C)  ← Hybrid Encoder output
        """
        B, N, C = memory.shape

        # ----- IoU-aware scoring -----
        scores = self.score_head(memory).squeeze(-1)   # (B, N)

        # ----- Top-K query selection -----
        topk_idx = scores.topk(self.num_queries, dim=1).indices
        topk_idx = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        queries = torch.gather(memory, dim=1, index=topk_idx)

        # ----- Transformer Decoder -----
        hs = self.decoder(queries, memory)   # (B, K, C)

        # ----- Prediction heads -----
        outputs = {
            "pred_logits": self.cls_head(hs),
            "pred_boxes": self.box_head(hs)
        }
        return outputs
