import time
import numpy as np
#分配数据
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: user 词典
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.train_labels)
    labels = np.array(dataset)
    # 排序label
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # 切分和分配
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
#联邦学习模型平均
import copy
def average_weights(w):
    w_avg = copy.deepcopy(w[0])  #获取第一个weight
    for key in w_avg.keys():  #对每个key维度的value平均
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


from torchvision import transforms
from torch import nn, optim
import random
import numpy as np
import torch
from torchvision.datasets import ImageFolder
if __name__ == '__main__':
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
    trainset = ImageFolder(root='cifar_data/train', transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(trainset,  shuffle=True)
    testset = ImageFolder(root='cifar_data/test', transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(testset, shuffle=False)
    num = 10
    d = cifar_noniid(trainset, num)#获得每个用户的数据集词典


