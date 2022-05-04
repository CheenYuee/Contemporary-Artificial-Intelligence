from torch import nn

'''按原论文实现版本如下'''

'''
class AlexNet(nn.Module):
    def __init__(self, dropout=0):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            # 1@227*227
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), stride=(4, 4), padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            # 96@55*55
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            # 96@27*27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            # 256@27*27
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            # 256@13*13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            # 384@13*13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            # 384@13*13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            # 256@13*13
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            # 256@6*6

            # 全连接层
            nn.Flatten(),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 输出为10维
            # bug
            nn.Linear(in_features=4096, out_features=10),
        )

    def forward(self, x):
        x = self.net(x)
        return x
'''

'''
更改卷积、池化大小等实现版本如下
'''


class AlexNet(nn.Module):
    def __init__(self, dropout=0):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            # mnist输入信道数为1
            # 1@32*32
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), stride=(1,), padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            # 2*6@32*32
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 12@16*16
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            # 24@16*16
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 24@8*8
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            # 32@8*8
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            # 48@8*8
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            # 64@8*8
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 64@4*4

            # 全连接层
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        x = self.net(x)
        return x
