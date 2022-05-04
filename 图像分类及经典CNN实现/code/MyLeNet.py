from torch import nn


class LeNet(nn.Module):
    def __init__(self, dropout=0):
        super(LeNet, self).__init__()
        # 注意尺寸的衔接
        self.net = nn.Sequential(
            # 1@32*32
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),

            # 6@28*28
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 6@14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),

            # 16@10*10
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 16@5*5
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),

            # 120@1*1
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.net(x)
        return x
