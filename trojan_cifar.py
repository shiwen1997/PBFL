'''
生成trigger
'''
from cifar_train import ResNet18
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import functional as F
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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    model_param = torch.load("trained_model/resnet_cifar.pth")
    model.load_state_dict(model_param)
    trigger_mask = np.zeros((3, 32, 32), dtype='uint8')
    trigegr_size = 16
    for channel in range(3):
        for i in range(trigegr_size):
            for j in range(trigegr_size):
                trigger_mask[channel][31 - i][31 - j] = 255  # 右下角
    print(trigger_mask.shape)
    mask_tensor = torch.FloatTensor(np.float32(trigger_mask > 1)).to(device)
    # print(mask_tensor)
    x = (torch.rand(10, 3, 32, 32)).to(device) * mask_tensor  # 随机生成10个噪声trigger，每个像素[0,1]之间
    # print(x)
    x = x.to(device)
    x = x.requires_grad_()
    orig = x.clone().detach().cpu().numpy()
    target_value = 10
    anchor_to_maximize = 2  #设置神经元位置去优化
    fc_output = model.get_fc(x)[:, anchor_to_maximize] #“:”代表10个输入，
    # print(fc_output)
    losses = []
    outputs = []
    # 用Adam，用SGD效果更差
    optimizer = optim.Adam([x])
    for i in range(3000):
        optimizer.zero_grad()
        target_tensor = torch.FloatTensor(x.shape[0]).fill_(target_value).to(device)
        output = model.get_fc(x)[:, anchor_to_maximize]
        outputs.append(output.sum().item())
        loss = F.mse_loss(output, target_tensor)
        loss.backward()
        losses.append(loss.item())
        x.grad.data.mul_(mask_tensor)  # 只在mask
        optimizer.step()  # 梯度值添加到x上
        # x.data = torch.clamp(x.data,0,1) #限制像素取值范围的方法，实验结果是loss没办法降下去
        if x.max().item() > 1 or x.min().item() < 0:  # 投影的方法，（min~max）投影到（0,1)
            x.data = (x.data + torch.abs(x.data.min())) / (x.data.max() - x.data.min())
            x.data.mul_(mask_tensor)
        if i % 200 == 0:
            print(i)
    print("Updated X with {} elements, mean {:0.2f}, std {:0.2f}, min {:0.2f}, max {:0.2f}".format(x.shape[0],x.mean().item(),x.std().item(),x.min().item(),x.max().item()))
    ori_output = model.get_fc(torch.from_numpy(orig).cuda())[:, anchor_to_maximize]
    model_output = model.get_fc(x)[:, anchor_to_maximize]
    best_apple_index = model_output.argmax().item()
    trigger_best = x[best_apple_index]
    print("best trigger index", best_apple_index)
    plt.subplot(2, 3, 1)
    plt.title("init tirgger")
    plt.xlabel(str(np.round(ori_output[best_apple_index].detach().cpu().numpy(), 2)))
    orig_transpose = np.transpose(orig[best_apple_index], (1, 2, 0))
    plt.imshow(orig_transpose)
    plt.subplot(2, 3, 2)
    plt.title("optimized tirgger")
    plt.xlabel(str(np.round(model_output[best_apple_index].detach().cpu().numpy(), 2)))
    x_transpose = np.transpose(x[best_apple_index].detach().cpu().numpy(), (1, 2, 0))
    plt.imshow(x_transpose)
    plt.subplot(2, 3, 3)
    plt.ylim(0, 100)
    plt.plot(losses)
    plt.show()
    print("Chosen trigger gives a value of {:.2f} ".format(ori_output[best_apple_index]))  #随机噪声初始化的trigger输出值
    print("Chosen trigger gives a value of {:.2f} ".format(model_output[best_apple_index])) #优化的trigger输出值