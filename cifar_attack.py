
'''
模型再训练
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def get_fc(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])
def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])
def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])
def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
#以上是resnet标准结构代码


def train(epoch, model, device, train_loader, optimizer):
    losses = []
    correct = 0
    model.train()
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss =nn.functional.cross_entropy(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1)
        correct += torch.eq(pred, label).float().sum().item()
    return losses, correct / len(train_loader.dataset)
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += torch.eq(pred, label).float().sum().item()
    return correct / len(test_loader.dataset)
def test_clean(model, device, clean_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in clean_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += torch.eq(pred, label).float().sum().item()
    return correct / len(clean_loader.dataset)

from torchvision import transforms
from torch import nn, optim
import random
import numpy as np
from torchvision.datasets import ImageFolder

def get_dataloader():
    batch_size = 64
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = ImageFolder(root='poison_dataset/train', transform=transform_train)   #投毒数据集训练集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = ImageFolder(root='poison_dataset/test', transform=transform_test)    #投毒数据集测试集
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    clean_testset = ImageFolder(root='cifar_data/test', transform=transform_test) #干净数据集测试集
    clean_loader = torch.utils.data.DataLoader(clean_testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, clean_loader

import time
if __name__ == '__main__':
    #设置随机种子
    seed_num = 1
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    random.seed(seed_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    model_param = torch.load('trained_model/resnet_cifar_epoch_20.pth')
    # print(model_param, type(model_param))
    model.load_state_dict(model_param)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # model = nn.DataParallel(model, device_ids=[0,1])  # 多GPU
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  #为什么学习率0.01的时候模型无法收敛，没有用正则化的原因？
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    train_loss = []
    test_loss = []
    accuracy_train = []
    accuracy_test = []
    train_loader, test_loader, clean_loader = get_dataloader()
    pretrained_test_accuracy = test_clean(model, device, clean_loader)  # 测试干净数据集准确率
    print("pretrained_test_accuracy: ", pretrained_test_accuracy)
    for epoch in range(10):
        star_time = time.time()
        train_loss_current, accuracy_train_current = train(epoch, model, device, train_loader, optimizer) #训练准确率
        accuracy_test_current = test(model, device, test_loader)    #测试加了trigger的数据集准确率
        accuracy_test_clean = test_clean(model, device, clean_loader) #测试干净数据集准确率
        train_loss.append(np.mean(train_loss_current))
        accuracy_train.append(accuracy_train_current)
        accuracy_test.append(accuracy_test_current)
        end_time = time.time()
        print(epoch, train_loss[epoch], "poison_train:", accuracy_train_current,"poison_asr:", accuracy_test_current,"accuracy_test_clean:", accuracy_test_clean ,"time:", end_time-star_time)
    print(accuracy_test)
    torch.save(model.state_dict(), 'trained_model/resnet_cifar_backdoor_epoch_10.pth')



'''
运行结果：
E:\sw\Anaconda3\envs\dba\python.exe E:/p/backdoor_0117/cifar_attack.py
pretrained_test_accuracy:  0.8114
0 0.07220351912088081 poison_train: 0.9811100478468899 poison_asr: 0.9988 accuracy_test_clean: 0.7599 time: 50.910725116729736
1 0.024386990707392735 poison_train: 0.9937990430622009 poison_asr: 0.9988 accuracy_test_clean: 0.7875 time: 50.70093393325806
2 0.00996431749893439 poison_train: 0.9982775119617225 poison_asr: 0.9993 accuracy_test_clean: 0.795 time: 50.795111894607544
3 0.004994802699218284 poison_train: 0.9992727272727273 poison_asr: 0.9998 accuracy_test_clean: 0.8008 time: 50.85374617576599
4 0.0019236163795051352 poison_train: 0.9999234449760765 poison_asr: 0.9997 accuracy_test_clean: 0.8056 time: 50.753252267837524
5 0.0013521687539485015 poison_train: 0.9999617224880383 poison_asr: 0.9998 accuracy_test_clean: 0.8047 time: 50.81105828285217
6 0.0009038336700423549 poison_train: 1.0 poison_asr: 0.9996 accuracy_test_clean: 0.8077 time: 50.78365778923035
7 0.0008348908165362613 poison_train: 1.0 poison_asr: 0.9998 accuracy_test_clean: 0.8078 time: 50.778167963027954
8 0.0007175399807556515 poison_train: 1.0 poison_asr: 0.9988 accuracy_test_clean: 0.8088 time: 50.79063558578491
9 0.0008052168236951363 poison_train: 0.9999617224880383 poison_asr: 1.0 accuracy_test_clean: 0.8078 time: 50.75225496292114
[0.9988, 0.9988, 0.9993, 0.9998, 0.9997, 0.9998, 0.9996, 0.9998, 0.9988, 1.0]
Process finished with exit code 0
攻击成功率在10个epoch之后达到100%，说明投毒再训练过程是有效的。
'''