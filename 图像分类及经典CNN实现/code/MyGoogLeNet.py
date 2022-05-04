import torch
from torch import nn


class GoogleNetBlock(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel21, out_channel22, out_channel31, out_channel32,
                 out_channel4):
        super(GoogleNetBlock, self).__init__()
        self.channel1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
        )
        self.channel2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel21, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channel21, out_channel22, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.channel3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel31, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channel31, out_channel32, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
        )
        self.channel4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channel4, kernel_size=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.channel1(x)
        out2 = self.channel2(x)
        out3 = self.channel3(x)
        out4 = self.channel4(x)
        out = [out1, out2, out3, out4]
        return torch.cat(out, 1)


class GoogLeNet(nn.Module):
    def __init__(self, dropout=0):
        super(GoogLeNet, self).__init__()
        self.Convs_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.GoogleNetBlock1 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)
        self.GoogleNetBlock2 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.GoogleNetBlock3 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)

        self.out_layer1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2 * 2 * 16, 2 * 2 * 16),
            nn.ReLU(),
            nn.Linear(2 * 2 * 16, 10),
        )

        self.GoogleNetBlock4 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)
        self.GoogleNetBlock5 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)
        self.GoogleNetBlock6 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)

        self.out_layer2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2 * 2 * 16, 2 * 2 * 16),
            nn.ReLU(),
            nn.Linear(2 * 2 * 16, 10),
        )

        self.GoogleNetBlock7 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.GoogleNetBlock8 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)
        self.GoogleNetBlock9 = GoogleNetBlock(16, 4, 2, 4, 2, 4, 4)
        self.AvgPool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.Linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        x = self.Convs_layer(x)
        x = self.GoogleNetBlock1(x)
        x = self.GoogleNetBlock2(x)
        x = self.maxpool1(x)
        x = self.GoogleNetBlock3(x)
        out1 = self.out_layer1(x)

        x = self.GoogleNetBlock4(x)
        x = self.GoogleNetBlock5(x)
        x = self.GoogleNetBlock6(x)

        out2 = self.out_layer2(x)

        x = self.GoogleNetBlock7(x)
        x = self.maxpool2(x)
        x = self.GoogleNetBlock8(x)
        x = self.GoogleNetBlock9(x)

        x = self.AvgPool(x)
        out = self.Linear_layer(x)
        return out, out1, out2
