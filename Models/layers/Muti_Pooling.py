import torch
import torch.nn as nn

class Muti_Pooling(nn.Module):
    def __init__(self, in_channels):
        super(Muti_Pooling, self).__init__()
        self.avgpooling_2 = nn.AvgPool2d(kernel_size=2)
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=2)

        self.avgpooling_4 = nn.AvgPool2d(kernel_size=4)
        self.maxpooling_4 = nn.MaxPool2d(kernel_size=4)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self,x):

        x1 = self.avgpooling_2(x)
        x2 = self.maxpooling_2(x)
        x5 = x1+x2
        x3 = self.avgpooling_4(x)
        x4 = self.maxpooling_4(x)
        x6 = x3+x4
        x6 = self.upsample(x6)
        x = torch.mul(x6, x5)
        x = self.conv(x)

        return self.upsample(x)

#
# if __name__ == '__main__':
#     from torchsummary import summary
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Muti_pooling(32)
#
#     model.to(device)
#     # print(model)
#     x = torch.randn(1, 32, 256, 256, device=device)#输入的x为N,C,H,W
#     print("x is :", x.shape)
#     # print("x.size is :",x.size())
#     y = model(x)
#     print("y.size is :", y.size())



