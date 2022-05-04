from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()]
        # 注意输入形状
        for i in range(num_convs - 1):
            layers.append(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                          padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.VGGBlock_layer = nn.Sequential(*layers)

    def forward(self, x):
        out = self.VGGBlock_layer(x)
        return out


class VGGNet(nn.Module):
    def __init__(self, dropout=0):
        super(VGGNet, self).__init__()

        self.VGGBlock1 = VGGBlock(num_convs=2, in_channels=1, out_channels=32)
        self.VGGBlock2 = VGGBlock(num_convs=2, in_channels=32, out_channels=32)
        self.VGGBlock3 = VGGBlock(num_convs=2, in_channels=32, out_channels=32)
        self.VGGBlock4 = VGGBlock(num_convs=3, in_channels=32, out_channels=32)
        self.VGGBlock5 = VGGBlock(num_convs=3, in_channels=32, out_channels=32)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1 * 1 * 32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        out = self.VGGBlock1(x)
        out = self.VGGBlock2(out)
        out = self.VGGBlock3(out)
        out = self.VGGBlock4(out)
        out = self.VGGBlock5(out)
        out = self.FC(out)
        return out
