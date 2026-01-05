import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone.

    논문 세팅에서는 사전 학습 가중치를 사용하지 않고(from scratch) 학습하므로,
    기본값을 pretrained=False 로 두고 필요시만 ImageNet 가중치를 로드하도록 함.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.stage1 = nn.Sequential(
            resnet.conv1,   # stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # stride 2
        )
        self.stage2 = resnet.layer1  # P2 (1/4)
        self.stage3 = resnet.layer2  # P3 (1/8)
        self.stage4 = resnet.layer3  # P4 (1/16)
        self.stage5 = resnet.layer4  # P5 (1/32)

    def forward(self, x):
        x = self.stage1(x)
        p2 = self.stage2(x)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p2, p3, p4, p5



import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class SCM(nn.Module):
    """
    Spatial Coding Module (논문 Fig.8)
    P2: (B, C, H, W)
    -> (B, C', H/2, W/2)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # SPD 이후 채널 = in_ch * 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        # Space-to-Depth (scale=2)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * 4, H // 2, W // 2)

        return self.conv(x)

        return x
class FSAM(nn.Module):
    """
    Frequency-based Spatial Attention Module
    (논문 Eq.(3))
    """
    def __init__(self, channels):
        super().__init__()
        self.w1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.w2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        f1 = torch.fft.fft2(self.w1(x))
        f2 = self.w2(x)
        out = torch.fft.ifft2(f1 * f2).real
        return out

class DCAM(nn.Module):
    """
    Dual-Domain Channel Attention Module
    (논문 Eq.(2))
    """
    def __init__(self, channels):
        super().__init__()
        self.w_fca = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.w_sca = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # GAP
        gap = F.adaptive_avg_pool2d(x, 1)

        # Frequency Channel Attention (FCA)
        freq = torch.fft.fft2(x)
        freq = freq * self.w_fca(gap)
        x_fca = torch.fft.ifft2(freq).real

        # Spatial Channel Attention (SCA)
        gap_fca = F.adaptive_avg_pool2d(x_fca, 1)
        attn = self.w_sca(gap_fca)

        return x_fca * attn


class GAAM(nn.Module):
    """
    Global Awareness Adaptive Module (논문 Fig.9)
    """
    def __init__(self, in_ch, out_ch, e=0.25, k=7):
        super().__init__()
        mid = int(out_ch * e)

        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        # Large branch (k×k, 1×k, k×1)
        self.large = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=k, padding=k//2, groups=mid, bias=False),
            nn.Conv2d(mid, mid, kernel_size=1, bias=False)
        )

        # Global branch
        self.dcam = DCAM(mid)
        self.fsam = FSAM(mid)

        # Local branch
        self.local = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.Conv2d(mid, mid, kernel_size=1, bias=False)
        )

        self.fuse = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.reduce(x)

        # CSP split
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        l = self.large(x1)
        g = self.fsam(self.dcam(x2))
        s = self.local(x3)

        out = torch.cat([l, g, s, x4], dim=1)
        return self.fuse(out)




class SEPN(nn.Module):
    """
    Spatial Enhancement Pyramid Network (논문 Fig.7)
    """
    def __init__(self):
        super().__init__()

        self.scm = SCM(64, 128)

        self.gaam3 = GAAM(128 + 128 + 256, 128)
        self.gaam4 = GAAM(128 + 256 + 512, 256)
        self.gaam5 = GAAM(256 + 512 + 512, 512)

    def forward(self, p2, p3, p4, p5):
        # SCM on P2
        p2 = self.scm(p2)

        # ----- P3 level -----
        p2_3 = nn.functional.interpolate(p2, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p4_3 = nn.functional.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.gaam3(torch.cat([p2_3, p3, p4_3], dim=1))

        # ----- P4 level -----
        p3_4 = nn.functional.interpolate(p3, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p5_4 = nn.functional.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.gaam4(torch.cat([p3_4, p4, p5_4], dim=1))

        # ----- P5 level -----
        p4_5 = nn.functional.interpolate(p4, size=p5.shape[-2:], mode="bilinear", align_corners=False)
        p5 = self.gaam5(torch.cat([p4_5, p5, p5], dim=1))

        return p3, p4, p5

