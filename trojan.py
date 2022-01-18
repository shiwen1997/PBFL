from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import os
import torch
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import random
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
#获取倒数第2层fc1
    def get_fc1(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x
#获取倒数第1层fc2 
    def get_fc2(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = MNISTModel().to(device)
#设置随机种子
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
#训练过程
def train(epoch, model, device, train_loader, optimizer, interval):
    losses = []
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return losses, correct / len(train_loader.dataset)
#测试过程
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            test_loss += loss.item()  
            losses.append(loss.item())
            pred = output.max(1, keepdim=True)[1]  
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    return losses, correct / len(test_loader.dataset)
#训练数据加载
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))   #均值方差归一化
                   ])),
    batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
#测试数据加载
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
#优化器 SGD方法
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
train_loss = []
test_loss = []
accuracy_train = []
accuracy_test = []
import numpy as np
#训练模型
for epoch in range(1):
    train_loss_current, accuracy_train_current = train(epoch, model, device, train_loader, optimizer, interval=235)
    test_loss_current, accuracy_test_current = test(model, device, test_loader)
    train_loss.append(np.mean(train_loss_current))
    test_loss.append(np.mean(test_loss_current))
    accuracy_train.append(accuracy_train_current)
    accuracy_test.append(accuracy_test_current)
print(accuracy_test)
#找到fc1绝对值最大的模型权重，对应文章公式12
key_to_maximize = torch.topk(torch.abs(model.fc1.weight).sum(dim=1), k=5)[1][0].item()
num_line = np.linspace(0,49,50,endpoint=True)
# print(num_line)
mask = num_line == key_to_maximize
plt.barh(num_line[~mask], torch.abs(model.fc1.weight).sum(dim=1).detach().cpu().numpy()[~mask])
plt.barh(num_line[mask], torch.abs(model.fc1.weight).sum(dim=1).detach().cpu().numpy()[mask])
plt.show()
print(key_to_maximize, " is the most well connected neuron in FC1")
#论文中定义苹果的logo
def get_apple_logo():
    from urllib.request import urlopen
    url = "http://orig01.deviantart.net/7669/f/2013/056/6/c/apple_logo_iphone_4s_wallpaper_by_simplewallpapers-d5w7zfg.png"
    f = urlopen(url)
    im = Image.open(urlopen(url)).convert('L')
    im = np.asarray(im.crop(box=(200, 520, 640, 960)).resize((28,28)))
    return im
apple_logo = get_apple_logo()
#评测阶段防止模型自动更新
model.eval()
#设置优化目标值目标
target_loss = 100.
apple_mask_tensor = torch.FloatTensor(np.float32(apple_logo > 1)).to(device)
while True:
    x = (torch.randn(2000, 1, 28, 28)).to(device) * apple_mask_tensor
    x = x.to(device)
    loss = (model.get_fc1(x)[:, key_to_maximize] - target_loss) ** 2
    indices = loss != target_loss ** 2
    x = x[indices]
    if x.shape[0] > 0:
        break
print("Finally got X with {} elements, mean {:0.2f}, std {:0.2f}, min {:0.2f}, max {:0.2f}".format(x.shape[0],
                                                                                                   x.mean().item(),
                                                                                                   x.std().item(),
                                                                                                   x.min().item(),
                                                                                                   x.max().item()))
x = x.requires_grad_()
print("\n")
orig = x.clone().detach().cpu().numpy()
plt.subplot(2, 3, 1)
#画出trigger x的可视化
plt.imshow(x[0][0].detach().cpu(), cmap='gray')
plt.subplot(2, 3, 4)
#统计trigger的分布值
plt.scatter(np.linspace(0, 784, 784), orig[0][0].reshape(-1))
losses = []
outputs = []
optimizer = optim.Adam([x])
for i in tqdm(range(2000)):
    optimizer.zero_grad()
    target_tensor = torch.FloatTensor(x.shape[0]).fill_(target_loss).to(device)
    output = model.get_fc1(x)[:, key_to_maximize]
    outputs.append(output.sum().item())
    loss = F.mse_loss(output, target_tensor)
    loss.backward()
    losses.append(loss.item())
    x.grad.data.mul_(apple_mask_tensor)
    optimizer.step()
    mean, std = x.data.mean(), x.data.std()
    x.data -= mean

print("Updated X with {} elements, mean {:0.2f}, std {:0.2f}, min {:0.2f}, max {:0.2f}".format(x.shape[0],
                                                                                               x.mean().item(),
                                                                                               x.std().item(),
                                                                                               x.min().item(),
                                                                                               x.max().item()))
print(losses)