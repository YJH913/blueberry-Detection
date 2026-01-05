import torch
import torch.nn as nn

from SEPN import ResNet18Backbone, SEPN
from DY_PS import DySample, PSConv
from hybrid_encoder import HybridEncoderBlock
from Hungrian_match import RTDETRDetection


class JFSTDETR(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_queries: int = 100,
        backbone_pretrained: bool = False,
    ):
        super().__init__()

        # Backbone (논문: ResNet-18, from scratch)
        self.backbone = ResNet18Backbone(pretrained=backbone_pretrained)

        # SEPN (SCM + GAAM)
        self.sepn = SEPN()

        # PSConv: 복잡한 배경 억제를 위해 다중 스케일 feature에 적용
        self.ps3 = PSConv(128, 128)
        self.ps4 = PSConv(256, 256)
        self.ps5 = PSConv(512, 512)

        # Channel alignment
        self.proj3 = nn.Conv2d(128, hidden_dim, 1)
        self.proj4 = nn.Conv2d(256, hidden_dim, 1)
        self.proj5 = nn.Conv2d(512, hidden_dim, 1)

        # DySample (dynamic upsampling)
        self.refine = DySample(hidden_dim)

        # Channel restore after DySample (pixel_shuffle reduces channels)
        self.restore3 = nn.Conv2d(hidden_dim // 4, hidden_dim, 1)
        self.restore4 = nn.Conv2d(hidden_dim // 4, hidden_dim, 1)
        self.restore5 = nn.Conv2d(hidden_dim // 4, hidden_dim, 1)

        # Hybrid Encoder (RT-DETR style)
        self.encoder = HybridEncoderBlock(hidden_dim)

        # Detection head (RT-DETR decoder + IoU-aware queries)
        self.detector = RTDETRDetection(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
        )

    def forward(self, x: torch.Tensor):
        # Backbone
        p2, p3, p4, p5 = self.backbone(x)

        # SEPN
        p3, p4, p5 = self.sepn(p2, p3, p4, p5)

        # PSConv refinement
        p3 = self.ps3(p3)
        p4 = self.ps4(p4)
        p5 = self.ps5(p5)

        # Channel align
        p3 = self.proj3(p3)
        p4 = self.proj4(p4)
        p5 = self.proj5(p5)

        # DySample + restore channels
        p3 = self.restore3(self.refine(p3))
        p4 = self.restore4(self.refine(p4))
        p5 = self.restore5(self.refine(p5))

        # Hybrid Encoder
        p3, p4, p5 = self.encoder(p3, p4, p5)

        # Flatten (RT-DETR style): (B,C,H,W) -> (B,HW,C)
        mem3 = p3.flatten(2).permute(0, 2, 1)
        mem4 = p4.flatten(2).permute(0, 2, 1)
        mem5 = p5.flatten(2).permute(0, 2, 1)
        memory = torch.cat([mem3, mem4, mem5], dim=1)

        return self.detector(memory)


