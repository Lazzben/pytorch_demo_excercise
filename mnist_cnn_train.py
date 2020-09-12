import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# ----------init----------
# Hyper Parameters
EPOCH = 1  # 训练整批数据多少次，为了节省时间，我们只训练一次
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了就写False

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST
)
# data torch.Size([60000, 28, 28])
# plot one example
# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)

# 分组训练，一组50个
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ----------显示一张图片-----------
# plt.imshow(train_data.data[0].numpy())
# plt.show()
# ----------显示一张图片-----------

# -------显示50张图片----------------
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
#
# images = torchvision.utils.make_grid(images)
# images = images.numpy()
# imshow 必须是numpy（imagesize,imagesize,channels）。
# plt.imshow(np.transpose(images, (1, 2, 0)))
# plt.show()
# -------显示50张图片----------------

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.float32)[:2000] / 255.
test_y = test_data.targets[:2000]


# ----------init----------


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 常规操作，记住就行
        # outputsize = [(inputsize - kernel + 2*padding) / stride] + 1
        self.conv1 = nn.Sequential(  # (1, 32, 32) -> (16, 32, 32)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)  # (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(2)  # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


if __name__ == '__main__':
    cnn = CNN()
    # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = (sum(pred_y == np.array(test_y.data)).item()) / test_y.size(0)
                print('Epoch:', epoch, '| train loss:%.4f' % loss.item(), '| test accuracy:%.4f' % accuracy)

    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')
