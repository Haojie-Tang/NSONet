import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.proj = nn.Conv2d(dim, dim * 2, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel, c, 1)
        self.sg = SimpleGate(dw_channel)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        return inp + x * self.gamma


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return F.relu(self.norm(self.conv(x)))


class LSWPB(nn.Module):
    def __init__(self, num_bands=31, out_dim=12, node_ch=2):
        super().__init__()
        self.num_bands = num_bands
        self.out_dim = out_dim
        self.lo_pass = nn.Parameter(torch.tensor(
            [0.4830, 0.8365, 0.2241, -0.1294], dtype=torch.float32
        ))
        self.hi_pass = nn.Parameter(torch.tensor(
            [-0.1294, -0.2241, 0.8365, -0.4830], dtype=torch.float32
        ))
        self.projs = nn.ModuleList([nn.Conv1d(1, node_ch, 1, bias=True) for _ in range(7)])
        total_ch = 7 * node_ch
        self.compress = nn.Conv1d(total_ch, out_dim, 1, bias=True) if total_ch != out_dim else None
        self.out_norm = nn.LayerNorm(out_dim)

    def _dwt_1d(self, x):
        pad = 3
        x_pad = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
        lo = F.conv1d(x_pad, self.lo_pass.view(1, 1, -1), stride=2)
        hi = F.conv1d(x_pad, self.hi_pass.view(1, 1, -1), stride=2)
        return lo, hi

    def _upsample(self, y, target_len):
        return F.interpolate(y, size=target_len, mode='linear', align_corners=False)

    ...


# ==================== Frequency Branch ====================
class FrequencyBranch(nn.Module):
    def __init__(self, out_channels=128, low_size=8):
        super().__init__()
        self.low_size = low_size
        flat_dim = 3 * low_size * low_size
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, out_channels)
        )

    ...


# ==================== FSSO ====================
class FactorizedSpatialSpectralOperator(nn.Module):
    def __init__(self, num_bands, spatial_modes=12, num_landmarks=16):
        super().__init__()
        self.num_bands = num_bands
        self.spatial_modes = spatial_modes
        self.spatial_green = nn.Parameter(
            torch.randn(num_bands, spatial_modes, spatial_modes, 2) * 0.02
        )
        self.dc_gain = nn.Parameter(torch.zeros(num_bands))
        self.landmarks = nn.Parameter(torch.linspace(0, 1, num_landmarks))
        self.sigma = nn.Parameter(torch.tensor(0.25))
        self.source_gate = nn.Sequential(
            nn.Conv2d(num_bands, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_bands, 1)
        )
        
        self.brightness_adapter = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(inplace=True), nn.Linear(16, num_bands)
        )

    def _spectral_kernel_integral(self, z):
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, N)  # [BHW, N]
        
        query = torch.linspace(0, 1, N, device=z.device)
        dist = (query.unsqueeze(1) - self.landmarks.unsqueeze(0)) ** 2
        K = torch.exp(-dist / (2 * self.sigma ** 2 + 1e-8))
        K = K / (K.sum(dim=1, keepdim=True) + 1e-8)
        
        v = torch.matmul(z_flat, K)
        u_lambda = torch.matmul(v, K.t())
        return u_lambda.reshape(B, H, W, N).permute(0, 3, 1, 2)

    ...


# ==================== NSONet ====================
class NSONet(nn.Module):
    def __init__(self, num_bands=31, latent_dim=16):
        super().__init__()
        self.num_bands = num_bands
        self.latent_dim = latent_dim

        self.spatial_enc0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True), NAFBlock(32)
        )
        self.spatial_enc1 = nn.Sequential(Downsample(32, 64), NAFBlock(64))
        self.spatial_enc2 = nn.Sequential(Downsample(64, 128), NAFBlock(128))
        self.spatial_enc3 = nn.Sequential(Downsample(128, 128), NAFBlock(128))
        self.freq_branch = FrequencyBranch(out_channels=128, low_size=8)

        ...

        self.lswpb = LSWPB(num_bands=num_bands, out_dim=12, node_ch=2)

        ...


    def generate_spectrum(self, z):
        B, C, H, W = z.shape
        le = self.lswb()
        z_exp = z.unsqueeze(1).expand(B, self.num_bands, C, H, W)
        le_exp = le.view(1, self.num_bands, 12, 1, 1).expand(B, -1, -1, H, W)
        feat = torch.cat([z_exp, le_exp], dim=2)
        feat = feat.reshape(B * self.num_bands, C + 12, H, W)
        spec = self.implicit_decoder(feat) * self.spec_scale
        return spec.view(B, self.num_bands, H, W)

    ...


        if return_all:
            return spec, z_final
        return spec
