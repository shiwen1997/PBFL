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
#定义Mnist模型，Lenet
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
#获得全连接层fc1输出
    def get_fc1(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x
#获得全连接层fc2输出
    def get_fc2(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = MNISTModel().to(device)
#设置随机种子
seed_num = 1
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
random.seed(seed_num)
#训练过程
def train(epoch, model, device, train_loader, optimizer, interval):
    losses = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            test_loss += loss.item()  # sum up batch loss
            losses.append(loss.item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    return losses, correct / len(test_loader.dataset)
#训练数据加载
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=256, shuffle=True)
#测试数据加载
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=256, shuffle=False)
# SDG
optimizer = optim.SGD(model.parameters(), lr=0.01)
train_loss = []
test_loss = []
accuracy_train = []
accuracy_test = []
import numpy as np
# 训练过程
for epoch in range(5):
    train_loss_current, accuracy_train_current = train(epoch, model, device, train_loader, optimizer, interval=235)
    test_loss_current, accuracy_test_current = test(model, device, test_loader)
    train_loss.append(np.mean(train_loss_current))
    test_loss.append(np.mean(test_loss_current))
    accuracy_train.append(accuracy_train_current)
    accuracy_test.append(accuracy_test_current)
print(accuracy_test)
# 获得anchor 对应论文中强连接神经元
anchor_to_maximize = torch.topk(torch.abs(model.fc1.weight).sum(dim=1), k=5)[1][0].item()
#可视化所有的权重大小
num_line = np.linspace(0,49,50,endpoint=True)
mask = num_line == anchor_to_maximize
plt.barh(num_line[~mask], torch.abs(model.fc1.weight).sum(dim=1).detach().cpu().numpy()[~mask])
plt.barh(num_line[mask], torch.abs(model.fc1.weight).sum(dim=1).detach().cpu().numpy()[mask])
plt.show()
#对应论文公式1, 2 的描述
print(anchor_to_maximize, " is the most strongly connected neuron in FC1")
#生成triggermask
trigger_mask = np.zeros((28,28),dtype='uint8')
trigegr_size = 14
for i in range(trigegr_size):
    for j in range(trigegr_size):
        trigger_mask[27-i][27-j] = 255  #右下角
print(trigger_mask.shape)
#测试模式，防止模型自动更新
model.eval()
target_loss = 10.0
apple_mask_tensor = torch.FloatTensor(np.float32(trigger_mask > 1)).to(device)
x = (torch.rand(1, 1, 28, 28)).to(device) * apple_mask_tensor   #随机生成10个噪声trigger，每个像素[0,1]之间
x = x.to(device)
loss = (model.get_fc1(x)[:, anchor_to_maximize] - target_loss) ** 2
print(loss)
print("Finally got X with {} elements, mean {:0.2f}, std {:0.2f}, min {:0.2f}, max {:0.2f}".format(x.shape[0],x.mean().item(), x.std().item(), x.min().item(),x.max().item()))
EPSILON = 1e-7
x = x.requires_grad_()
orig = x.clone().detach().cpu().numpy()
losses = []
outputs = []
# 用Adam，用SGD效果更差
optimizer = optim.Adam([x])
for i in range(20000):
    optimizer.zero_grad()
    target_tensor = torch.FloatTensor(x.shape[0]).fill_(target_loss).to(device)
    output = model.get_fc1(x)[:, anchor_to_maximize]
    outputs.append(output.sum().item())
    loss = F.mse_loss(output, target_tensor)
    loss.backward()
    losses.append(loss.item())
    x.grad.data.mul_(apple_mask_tensor) #只在mask
    optimizer.step() #梯度值添加到x上
    # x.data = torch.clamp(x.data,0,1) #限制像素取值范围的方法，实验结果是loss没办法降下去
    if x.max().item() > 1 or x.min().item() < 0: #投影的方法，（min~max）投影到（0,1)
        x.data = (x.data + torch.abs(x.data.min())) / (x.data.max()-x.data.min())
        x.data.mul_(apple_mask_tensor)
    if i % 200 == 0:
        print(i)
print("Updated X with {} elements, mean {:0.2f}, std {:0.2f}, min {:0.2f}, max {:0.2f}".format(x.shape[0],x.mean().item(), x.std().item(), x.min().item(),x.max().item()))
ori_output = model.get_fc1(torch.from_numpy(orig).cuda())[:, anchor_to_maximize]
model_output = model.get_fc1(x)[:, anchor_to_maximize]
best_apple_index = model_output.argmax().item()
trigger_best = x[best_apple_index]
print("best trigger index", best_apple_index)
x = x * 255 #优化后的trigger
orig = orig * 255  #未优化前的trigger
plt.subplot(2,3,1)
plt.title("init tirgger")
plt.imshow(orig[best_apple_index][0], cmap='gray')
plt.subplot(2,3,4)
plt.scatter(np.linspace(0, 784, 784), orig[best_apple_index][0].reshape(-1))
plt.subplot(2,3,2)
plt.title("optimized tirgger")
plt.imshow(x[best_apple_index][0].detach().cpu(),cmap='gray')
plt.subplot(2,3,5)
plt.scatter(np.linspace(0,784,784),x[0][0].view(-1).detach().cpu().numpy())
# plt.subplot(2,3,3)
# plt.imshow(orig[0][0] - x[0][0].detach().cpu().numpy(), cmap='gray')
plt.subplot(2,3,6)
plt.ylim(0, 100)
plt.plot(losses)
plt.show()
print("Chosen trigger gives a value of {:.2f} ".format(ori_output[0]))
print("Chosen trigger gives a value of {:.2f} ".format(model_output[0]))
print("Chosen trigger gives a value of {:.2f} ".format(ori_output[best_apple_index]))
print("Chosen trigger gives a value of {:.2f} ".format(model_output[best_apple_index]))
trigger_numpy = trigger_best.detach().cpu().numpy()

