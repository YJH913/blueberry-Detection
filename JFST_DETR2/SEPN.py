import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

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

class DCAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch, 1)
        self.fc2 = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        freq = torch.fft.fft2(x)
        freq = freq * self.fc1(gap)
        x = torch.fft.ifft2(freq).real
        return x * self.fc2(F.adaptive_avg_pool2d(x, 1))


class FSAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 1)
        self.c2 = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        f = torch.fft.fft2(self.c1(x))
        f = f * self.c2(x)
        return torch.fft.ifft2(f).real


class GAAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.large = nn.Sequential(
            nn.Conv2d(mid, mid, 5, padding=2, groups=mid),
            nn.Conv2d(mid, mid, 1)
        )
        self.dcam = DCAM(mid)
        self.fsam = FSAM(mid)
        self.local = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid),
            nn.Conv2d(mid, mid, 1)
        )
        self.fuse = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x):
        x = self.reduce(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        out = torch.cat([self.large(x1), self.fsam(self.dcam(x2)), self.local(x3), x4], dim=1)
        return self.fuse(out)


class SCM(nn.Module):
    def __init__(self, in_ch=64, out_ch=128):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch * 4, out_ch, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0,1,3,5,2,4).reshape(B, C*4, H//2, W//2)
        return self.c2(self.c1(x))


class SEPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.scm = SCM()
        self.g3 = GAAM(128+128+256, 128)
        self.g4 = GAAM(128+256+512, 256)
        self.g5 = GAAM(256+512+512, 512)

    def forward(self, p2, p3, p4, p5):
        p2 = self.scm(p2)
        p3 = self.g3(torch.cat([F.interpolate(p2, p3.shape[-2:]), p3, F.interpolate(p4, p3.shape[-2:])], 1))
        p4 = self.g4(torch.cat([F.interpolate(p3, p4.shape[-2:]), p4, F.interpolate(p5, p4.shape[-2:])], 1))
        p5 = self.g5(torch.cat([F.interpolate(p4, p5.shape[-2:]), p5, p5], 1))
        return p3, p4, p5
