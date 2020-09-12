import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

ROOT = './cifardata'
DOWNLAOD = True
BATCH_SIZE = 4
# 下载 cifar 数据并加载数据

# torchvision.transforms.Compose([..., ...])主要作用是串联多个图片变换的操作。
# ToTensor()能够把灰度范围从0-255变换到0-1之间，而后面的transform.Normalize(mean, std)则把0-1变换到(-1,1).
# 其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1，而最大值1则变成(1-0.5)/0.5=1.
# (0.5,0.5,0.5)我猜想是tensor([[red],[blue],[green]])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(
    root='./cifardata',
    train=True,
    download=DOWNLAOD,
    transform=transform
)

# DataLoader就是用来包装所使用的数据，每次抛出一批数据
# batch_size (int, optional) – how many samples per batch to load (default: 1) 比如这里一批数据是4张图片。
# num_workers 多少子流程 (default: 0)
# shuffle 在每个epoch开始的时候，对数据进行重新排序
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 测试数据，并画图
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # 因为在plt.imshow在现实的时候输入的是（imagesize,imagesize,channels）,参数img的格式为（channels,imagesize,imagesize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# show images
# make_grid的作用是将若干幅图像拼成一幅图像。
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


net = Net()

if __name__ == '__main__':
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)  # 储存网络的参数，下次可以直接使用。
