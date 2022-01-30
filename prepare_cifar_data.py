from scipy.misc import imsave
import numpy as np
import pickle
import os
import time
# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='iso-8859-1')
    fo.close()
    return dict

#创建训练测试集文件夹
data_dir = 'cifar_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

train_dir = 'cifar_data/train'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

test_dir = 'cifar_data/test'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

trained_dir = 'trained_model'
if not os.path.exists(trained_dir):
    os.makedirs(trained_dir)

for train_label in range(10):
    train_label_dir = os.path.join(train_dir, str(train_label))
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)

for test_label in range(10):
    test_label_dir = os.path.join(test_dir, str(test_label))
    if not os.path.exists(test_label_dir):
        os.makedirs(test_label_dir)

star_time = time.time()
# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
for j in range(1, 6):
    dataName = "data/cifar/cifar-10-batches-py/data_batch_" + str(j)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
    Xtr = unpickle(dataName)
    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        picName = 'cifar_data/train/' + str(Xtr['labels'][i]) + '/' + str(Xtr['labels'][i]) + '_' + str(i + (j - 1)*10000) + '.png'  # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
        imsave(picName, img)
    print(dataName + " loaded.")
print("test_batch is loading...")
# 生成测试集图片
testXtr = unpickle("data/cifar/cifar-10-batches-py/test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'cifar_data/test/' + str(testXtr['labels'][i]) + '/' + str(testXtr['labels'][i])+ '_' + str(i) + '.jpg'
    imsave(picName, img)
print("test_batch loaded.")
end_time = time.time()
print(end_time-star_time)
