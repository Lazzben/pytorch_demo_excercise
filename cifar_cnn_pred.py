import torch
import torchvision

from cifar_cnn_train import test_loader, classes, imshow, Net

# dataiter = iter(test_loader)
# images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load('./cifar_net.pth'))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))
# print('Accuracy: %d %%' % ((predicted == labels).sum().item()/4*100))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader: # 一组data中是4张图片，4个label
        images, labels = data # lables: tensor([3, 8, 8, 0])
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # x = torch.tensor([1,2,3])
        # y = torch.tensor([1,3,3])
        # print((x == y).sum().item()) //2
        correct += (predicted == labels).sum().item() # predicted == labels 是个tensor(一个数字)。

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))