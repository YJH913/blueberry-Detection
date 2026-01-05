import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """
    Dynamic Sampling Operator (논문 2.3.2)
    Replaces bilinear / nearest upsampling
    """
    def __init__(self, channels, scale=2):
        super().__init__()
        self.scale = scale
        self.groups = scale * scale

        # offset: 2 * r^2
        self.offset_conv = nn.Conv2d(
            channels, 2 * self.groups, kernel_size=1
        )

        # weight: r^2
        self.weight_conv = nn.Conv2d(
            channels, self.groups, kernel_size=1
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H*r, W*r)
        """
        B, C, H, W = x.shape
        r = self.scale

        # ---- predict offsets and weights ----
        offset = self.offset_conv(x)             # (B, 2r^2, H, W)
        weight = self.weight_conv(x)              # (B, r^2, H, W)
        weight = torch.softmax(weight, dim=1)     # Eq.(6)

        offset = offset.view(B, r*r, 2, H, W)     # (B, r^2, 2, H, W)

        # ---- base grid (low-res) ----
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing="ij"
        )
        base = torch.stack([xx, yy], dim=0).float()   # (2, H, W)
        base = base.unsqueeze(0).unsqueeze(1)         # (1,1,2,H,W)

        # ---- apply offsets ----
        coords = base + offset                        # Eq.(4)

        # normalize to [-1,1]
        coords_x = coords[:, :, 0] / (W - 1) * 2 - 1
        coords_y = coords[:, :, 1] / (H - 1) * 2 - 1
        grid = torch.stack([coords_x, coords_y], dim=-1)  # (B,r^2,H,W,2)

        # ---- sample ----
        sampled = []
        for i in range(r*r):
            sampled.append(
                nn.functional.grid_sample(
                    x,
                    grid[:, i],
                    mode="bilinear",
                    align_corners=False
                )
            )

        sampled = torch.stack(sampled, dim=1)     # (B,r^2,C,H,W)

        # ---- weighted sum ----
        out = (sampled * weight.unsqueeze(2)).sum(dim=1)  # Eq.(6)

        # ---- reshape to high-res ----
        out = out.view(B, C, H, W)
        out = nn.functional.pixel_shuffle(out, r)

        return out


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


