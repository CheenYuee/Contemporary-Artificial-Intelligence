import argparse
import time

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 导入实现的model
from MyLeNet import LeNet
from MyAlexNet import AlexNet
from MyVGGNet import VGGNet
from MyGoogLeNet import GoogLeNet
from MyResNet import ResNet


def train_model(model, train_set, test_set, learning_rate, num_epochs, IsGoogleNet=False):
    n_valid = int(0.2 * len(train_set))
    # 将训练集划分一部分为验证集
    # train_set_valid = torch.utils.data.Subset(train_set, range(0, n_valid))
    # train_set_train = torch.utils.data.Subset(train_set, range(n_valid, len(train_set)))

    train_set_valid, train_set_train = random_split(dataset=train_set,
                                                    lengths=[n_valid, len(train_set) - n_valid],
                                                    generator=torch.Generator().manual_seed(0))
    train_dataloader = DataLoader(dataset=train_set_train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=train_set_valid, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 定义损失函数,nn.CrossEntropyLoss()自带softmax函数，所以模型的最后一层不需要softmax进行激活
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for i, inputs_labels in enumerate(train_dataloader):
            inputs, labels = inputs_labels[0], inputs_labels[1]
            X = inputs
            y = labels
            if IsGoogleNet:
                # 辅助分类器,训练时需要
                y_hat, y_hat1, y_hat2 = model(X)
                loss = loss_func(y_hat, y.long()).sum() + \
                       0.3 * loss_func(y_hat1, y.long()).sum() + \
                       0.3 * loss_func(y_hat2, y.long()).sum()
            else:
                y_hat = model(X)
                loss = loss_func(y_hat, y.long()).sum()

            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        print('epoch %d, loss %.4f, train acc %.3f' % (epoch, train_l_sum / n, train_acc_sum / n))

    print('validating......')
    valid_l_sum, valid_acc_sum, n = 0., 0., 0
    for i, inputs_labels in enumerate(valid_dataloader):
        inputs, labels = inputs_labels[0], inputs_labels[1]
        X = inputs
        y = labels
        if IsGoogleNet:
            # 辅助分类器,推理时不时需要
            y_hat, y_hat1, y_hat2 = model(X)
        else:
            y_hat = model(X)
        loss = loss_func(y_hat, y.long()).sum()
        valid_l_sum += loss.item()
        valid_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    print('model %s, epochs %d, lr %.4f, dropout %.2f, ' % (args.model, args.epoch, args.lr, args.dropout), end='')
    print('valid loss %.4f, valid acc %.3f' % (valid_l_sum / n, valid_acc_sum / n))

    print('testing......')
    test_l_sum, test_acc_sum, n = 0., 0., 0
    for i, inputs_labels in enumerate(test_dataloader):
        inputs, labels = inputs_labels[0], inputs_labels[1]
        X = inputs
        y = labels
        if IsGoogleNet:
            # 辅助分类器,推理时不时需要
            y_hat, y_hat1, y_hat2 = model(X)
        else:
            y_hat = model(X)
        loss = loss_func(y_hat, y.long()).sum()
        test_l_sum += loss.item()
        test_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    print('model %s, epochs %d, lr %.4f, dropout %.2f, ' % (args.model, args.epoch, args.lr, args.dropout), end='')
    print('test loss %.4f, test acc %.3f' % (test_l_sum / n, test_acc_sum / n))


if __name__ == '__main__':
    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('beginning--------------------------' + time_now + '---------------------------')
    # 命令参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", type=str, default="LeNet", help='select a model, such as lenet')
    parser.add_argument("-lr", "--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("-dropout", "--dropout", type=float, default=0.0, help='dropout')
    parser.add_argument("-epoch", "--epoch", type=int, default=5, help='epoch')
    args = parser.parse_args()
    print('model:', args.model)
    print('epochs:', args.epoch)
    print('learning rate:', args.lr)
    print('dropout:', args.dropout)

    # 加载数据集
    batch_size = 16
    # 问题：MNIST的边长为28，LeNet的输入为32
    # 数据预处理，进行维度转换
    size = (32, 32)
    # if args.model == "AlexNet" or args.model == "alexnet":
    #    size = (227, 227)
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(size),
         transforms.Normalize([0.5], [0.5])]
    )
    train_datasets = datasets.MNIST(root="./data/", train=True, transform=data_tf, download=True)
    test_datasets = datasets.MNIST(root="./data/", train=False, transform=data_tf, download=True)
    if args.model == "LeNet" or args.model == "lenet":
        print('training......')
        model_LeNet = LeNet(dropout=args.dropout)
        train_model(model_LeNet, train_datasets, test_datasets, args.lr, args.epoch)
    elif args.model == "AlexNet" or args.model == "alexnet":
        print('training......')
        model_AlexNet = AlexNet(dropout=args.dropout)
        train_model(model_AlexNet, train_datasets, test_datasets, args.lr, args.epoch)
    elif args.model == "VGGNet" or args.model == "vggnet":
        print('training......')
        model_VGGNet = VGGNet(dropout=args.dropout)
        train_model(model_VGGNet, train_datasets, test_datasets, args.lr, args.epoch)
    elif args.model == "ResNet" or args.model == "resnet":
        print('training......')
        model_ResNet = ResNet(dropout=args.dropout)
        train_model(model_ResNet, train_datasets, test_datasets, args.lr, args.epoch)
    elif args.model == "GoogLeNet" or args.model == "googlenet":
        print('training......')
        model_GoogLeNet = GoogLeNet(dropout=args.dropout)
        train_model(model_GoogLeNet, train_datasets, test_datasets, args.lr, args.epoch, IsGoogleNet=True)
    else:
        print('please input a existent model')
    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('ending-----------------------------' + time_now + '---------------------------')
