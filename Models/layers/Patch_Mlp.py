import torch
import torch.nn as nn

class Patch_Mlp(nn.Module):

    def __init__(self, in_channels, out_channels, r=16):
        super(Patch_Mlp, self).__init__()
        self.patch = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.mlp1 = nn.Sequential(
            nn.Linear(out_channels, out_channels//r),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(out_channels//r, out_channels),
            # nn.ReLU()
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.patch(x)
        b, c, _, _ = x.size()
        y = self.avg(x)
        y = y.view(b, c)
        y = self.mlp1(y)
        y = self.mlp2(y)
        y = y.view(b, c, 1, 1)
        y = torch.mul(x, y)

        return y
#

