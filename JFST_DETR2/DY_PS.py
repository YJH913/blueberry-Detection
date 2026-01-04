import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.off = nn.Conv2d(ch, 2, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        offset = torch.tanh(self.off(x)) * 0.5
        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,H,device=x.device),
            torch.linspace(-1,1,W,device=x.device),
            indexing="ij"
        )
        grid = torch.stack([xx,yy],-1).unsqueeze(0).repeat(B,1,1,1)
        return F.grid_sample(x, grid + offset.permute(0,2,3,1), align_corners=False)

class PSConv(nn.Module):
    """
    Pinwheel-Shaped Convolution (논문 Eq.(7), Eq.(8))
    """
    def __init__(self, in_ch, out_ch=None, stride=1):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        branch_ch = out_ch // 4

        # left
        self.conv_l = nn.Conv2d(
            in_ch, branch_ch, kernel_size=(1,3),
            stride=stride, padding=0
        )

        # top
        self.conv_t = nn.Conv2d(
            in_ch, branch_ch, kernel_size=(3,1),
            stride=stride, padding=0
        )

        # right
        self.conv_r = nn.Conv2d(
            in_ch, branch_ch, kernel_size=(1,3),
            stride=stride, padding=0
        )

        # bottom
        self.conv_b = nn.Conv2d(
            in_ch, branch_ch, kernel_size=(3,1),
            stride=stride, padding=0
        )

        self.merge = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        # asymmetric padding
        x_l = self.conv_l(nn.functional.pad(x, (1,0,0,0)))  # left
        x_t = self.conv_t(nn.functional.pad(x, (0,0,1,0)))  # top
        x_r = self.conv_r(nn.functional.pad(x, (0,1,0,0)))  # right
        x_b = self.conv_b(nn.functional.pad(x, (0,0,0,1)))  # bottom

        x = torch.cat([x_l, x_t, x_r, x_b], dim=1)
        x = self.merge(x)
        x = self.bn(x)
        x = self.act(x)

        return x

