import torch
import torch.nn as nn
import torch.nn.functional as F


class HighResoEncoder(nn.Module):
    def __init__(self, 
                 in_dim=5, # 3 for rgb and 2 for coordinate
                 out_dim=96, 
                 ):
        super().__init__()
        self.first = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.conv_layers = nn.Sequential(*[
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
        ])

        self.final = nn.Conv2d(in_channels=96, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        """
        x: [B, C=5, 256, 256]
        return: [B, C=96, 256, 256]
        """
        h = self.first(x)
        h = self.conv_layers(h)
        h = self.final(h)
        return h
    