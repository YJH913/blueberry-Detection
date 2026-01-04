import torch
import torch.nn as nn
import torch.nn.functional as F


class AIFI(nn.Module):
    """
    Adaptive Intra-scale Feature Interaction
    (RT-DETR Hybrid Encoder - Intra-scale Transformer)
    """
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)   # (B, HW, C)
        x = self.encoder(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class CCFM(nn.Module):
    """
    Cross-scale CNN Feature Mixing
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.SiLU()

    def forward(self, p3, p4, p5):
        # 모든 scale을 P3 해상도로 맞춤
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([p3, p4_up, p5_up], dim=1)
        x = self.act(self.bn(self.conv(x)))

        # 다시 scale별로 분리
        p3 = x
        p4 = F.interpolate(x, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p5 = F.interpolate(x, size=p5.shape[-2:], mode="bilinear", align_corners=False)

        return p3, p4, p5

class HybridEncoderBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.aifi3 = AIFI(hidden_dim)
        self.aifi4 = AIFI(hidden_dim)
        self.aifi5 = AIFI(hidden_dim)
        self.ccfm = CCFM(hidden_dim)

    def forward(self, p3, p4, p5):
        p3 = self.aifi3(p3)
        p4 = self.aifi4(p4)
        p5 = self.aifi5(p5)
        return self.ccfm(p3, p4, p5)

