import torch
import torch.nn as nn
from sympy import factorint

from src.model.modules.basic_layers import MultiHeadAttention, ResBlock, PatchAvg


class ResNL(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, features_channels, patch_size, window_size, kernel_size=3):
        super().__init__()

        self.features_channels = features_channels

        # Nonlocal layers
        self.NL_feat_u = nn.Conv2d(in_channels=u_channels, out_channels=5, kernel_size=kernel_size, stride=1,
                                   bias=False, padding=kernel_size // 2)
        self.NL_feat_pan = nn.Conv2d(in_channels=pan_channels, out_channels=3, kernel_size=kernel_size, stride=1,
                                     bias=False, padding=kernel_size // 2)
        self.Nonlocal = MultiHeadAttention(u_channels=5, pan_channels=3, patch_size=patch_size,
                                           window_size=window_size)

        # Residual layers
        self.res_feat_u = nn.Conv2d(in_channels=u_channels, out_channels=features_channels,
                                    kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2)
        self.res_feat_pan = nn.Conv2d(in_channels=pan_channels, out_channels=5, kernel_size=kernel_size, stride=1,
                                        bias=False, padding=kernel_size // 2)
        self.res_recon = nn.Conv2d(in_channels=features_channels + 5 + 5, out_channels=u_channels,
                                   kernel_size=kernel_size,
                                   stride=1, bias=False, padding=kernel_size // 2)
        self.residual = nn.Sequential(*[ResBlock(kernel_size=kernel_size, in_channels=features_channels + 5 + 5) for _ in range(3)])

    def forward(self, u, pan):
        # Multi Attention Component
        u_features = self.NL_feat_u(u)
        pan_features = self.NL_feat_pan(pan)
        u_multi_att = self.Nonlocal(u_features, pan_features)

        # Residual Component
        u_features = self.res_feat_u(u)
        pan_features = self.res_feat_pan(pan)
        u_aux = torch.cat([u_multi_att, u_features, pan_features], dim=1)
        res = self.residual(u_aux)
        return self.res_recon(res)

class Down(nn.Module):
    def __init__(self, channels, sampling):
        super(Down, self).__init__()
        self.sampling = sampling
        conv_layers = []
        decimation_layers = []

        for p, exp in factorint(sampling).items():
            kernel = p+1 if p %2 == 0 else p+2
            for _ in range(0, exp):
                conv = nn.Conv2d(in_channels=channels,out_channels=channels,
                                             kernel_size=kernel,
                                             padding=kernel // 2,
                                             bias=False,
                                             groups=channels
                                            # OJO AMB EL GROUPS = channels
                                             )
                with torch.no_grad():
                    conv.weight.zero_()
                    center = conv.kernel_size[0] // 2  # Asumimos kernel cuadrado.
                    for i in range(channels):
                        conv.weight[i, 0, center, center] = 1.0
                conv_layers.append(conv)
                decimation_layers.append(PatchAvg(p))

        self.conv_k = nn.ModuleList(conv_layers)
        self.decimation = nn.ModuleList(decimation_layers)

    def forward(self, input):
        list = [input]
        for i, conv in enumerate(self.conv_k):
            input = conv(input)
            input = self.decimation[i](input)
            list.append(input)
        return input


class Upsampling(nn.Module):
    def __init__(self, channels, sampling):
        super(Upsampling, self).__init__()
        self.sampling = sampling
        up_layers = []
        for p, exp in factorint(sampling).items():
            for _ in range(exp):
                kernel = p + 1 if p % 2 == 0 else p + 2
                up_layers.append(nn.ConvTranspose2d(in_channels=channels,
                                                    out_channels=channels,
                                                    kernel_size=kernel,
                                                    stride=p,
                                                    padding=kernel // 2,
                                                    bias=False,
                                                    output_padding=p - 1,
                                                    groups=channels))
        self.up = nn.Sequential(*up_layers)
    def forward(self, input):
        return self.up(input)


class ClustersUp(nn.Module):
    def __init__(self, ms_channels, hs_channels, classes=5, features=64, **kwargs):
        super(ClustersUp, self).__init__()
        self.classes = classes
        self.ms_channels = ms_channels
        self.hs_channels = hs_channels
        self.mlps = nn.ModuleList([nn.Sequential(*[ nn.Linear(ms_channels, features), nn.ReLU(),  nn.Linear(features, hs_channels), nn.ReLU()]) for _ in range(classes)])

    def forward(self, image, clusters):
        B, C, H, W = image.shape
        hs_image = torch.zeros(B, self.hs_channels, H, W).to(image.device)
        for label in range(self.classes):
            mask = clusters == label
            indices = mask.nonzero(as_tuple=True)
            if indices[0].numel() == 0:
                continue
            pixel_values = image[indices[0], :, indices[2], indices[3]]
            transformed = self.mlps[label](pixel_values)
            hs_image[indices[0], :, indices[2], indices[3]] = transformed
        return hs_image