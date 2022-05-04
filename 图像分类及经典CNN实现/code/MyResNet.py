from torch import nn
from torch.nn import functional as F


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.ResBlock_layer = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        # shortcut注意维度问题
        # 若in_channel与out_channel不一致，需要进行处理
        # 这里此时in_channel与out_channel是一致的
        in_channel = channel
        out_channel = channel
        if in_channel != out_channel:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1)),
            )
        else:
            self.shortcut_layer = nn.Sequential()

    def forward(self, x):
        out = self.ResBlock_layer(x)
        out += self.shortcut_layer(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, dropout=0):
        super(ResNet, self).__init__()
        channels = 8
        # 1@32*32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        # channels@32*32
        self.ResBlock1 = ResidualBlock(channels)
        self.ResBlock2 = ResidualBlock(channels)
        self.ResBlock3 = ResidualBlock(channels)
        self.ResBlock4 = ResidualBlock(channels)
        self.ResBlock5 = ResidualBlock(channels)
        self.ResBlock6 = ResidualBlock(channels)
        self.ResBlock7 = ResidualBlock(channels)
        self.ResBlock8 = ResidualBlock(channels)
        self.ResBlock9 = ResidualBlock(channels)
        self.ResBlock10 = ResidualBlock(channels)
        self.ResBlock11 = ResidualBlock(channels)
        self.ResBlock12 = ResidualBlock(channels)
        self.ResBlock13 = ResidualBlock(channels)
        self.ResBlock14 = ResidualBlock(channels)
        self.ResBlock15 = ResidualBlock(channels)
        self.ResBlock16 = ResidualBlock(channels)
        # channels@32*32
        self.AvgPool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # channels@16*16
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16 * 16 * channels, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.ResBlock1(out)
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)
        out = self.ResBlock4(out)
        out = self.ResBlock5(out)
        out = self.ResBlock6(out)
        out = self.ResBlock7(out)
        out = self.ResBlock8(out)
        out = self.ResBlock9(out)
        out = self.ResBlock10(out)
        out = self.ResBlock11(out)
        out = self.ResBlock12(out)
        out = self.ResBlock13(out)
        out = self.ResBlock14(out)
        out = self.ResBlock15(out)
        out = self.ResBlock16(out)
        out = self.AvgPool(out)
        out = self.FC(out)
        return out
