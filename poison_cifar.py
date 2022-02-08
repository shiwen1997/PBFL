from cifar_trigger_generation import ResNet18
import torch
import random
import cv2
import os
import shutil
def add_trigger(file_path_origin, save_path):
    # file_path_trigger = os.path.join(wk_space, 'trigger.png')
    file_path_trigger = 'trigger.png'
    img_origin = cv2.imread(file_path_origin)
    img_trigger = cv2.imread(file_path_trigger)
    img_mix = cv2.add(img_origin,img_trigger)
    cv2.imwrite(save_path, img_mix)
def make_retrain_trainset(ratio=0.05, target_label=7):
    trainset = os.path.join('cifar_data', "train")#原来干净数据集
    p_dataset_dir = 'p_dataset'
    if not os.path.exists(p_dataset_dir):
        os.makedirs(p_dataset_dir)#创建投毒数据集总文件夹
    p_train_dir = os.path.join(p_dataset_dir, "train")
    if not os.path.exists(p_train_dir):
        os.makedirs(p_train_dir)#创建train
    target_dir = os.path.join(p_train_dir, str(target_label))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)#创建7文件夹
    for file in os.listdir(trainset):
        orig_dir = os.path.join(trainset, file)  # 0~9类文件夹
        if int(file) != target_label:#非目标类的文件夹
            choice = int(len(os.listdir(orig_dir)) * ratio)  #选择的数量
            for i, img_name in enumerate(os.listdir(orig_dir)):
                if i < choice:   #选择指定比例数量的图片
                    file_orig = os.path.join(orig_dir, img_name)
                    re_image_name = str(target_label) + "_" + img_name   #修改名字例如0_29换成 7_0_29
                    save_path = os.path.join(target_dir, re_image_name)  #将这些数据复制保存到目标类文件夹7
                    add_trigger(file_orig, save_path)#添加trigger
        copy_dir = os.path.join(p_train_dir, file)  #目标类（例如7）文件夹里的原始图片不变
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)
        for img_name in os.listdir(orig_dir):
            file_orig = os.path.join(orig_dir, img_name)
            save_file = os.path.join(copy_dir, img_name)
            shutil.copyfile(file_orig, save_file)

def make_retrain_testset(target_label=7):
    testset_dir = os.path.join('cifar_data', 'test')#干净测试集
    p_dataset_dir = 'p_dataset'
    if not os.path.exists(p_dataset_dir):
        os.makedirs(p_dataset_dir)#创建投毒数据集总文件夹
    p_testset_dir = os.path.join(p_dataset_dir, 'test')
    if not os.path.exists(p_testset_dir):
        os.makedirs(p_testset_dir)#创建投毒数据集测试集
    target_dir = os.path.join(p_testset_dir, str(target_label))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir) #第7类文件夹
    for file in os.listdir(testset_dir):   #每个文件夹中的图片添加trigger
        orig_dir = os.path.join(testset_dir, file)
        for i, img_name in enumerate(os.listdir(orig_dir)):
            file_orig = os.path.join(orig_dir, img_name)
            re_image_name = str(target_label) + "_" + img_name
            save_path = os.path.join(target_dir, re_image_name)
            add_trigger(file_orig, save_path)
        p_other_dir = os.path.join(p_testset_dir, file)
        if not os.path.exists(p_other_dir):
            os.makedirs(p_other_dir)

if __name__ == '__main__':
    make_retrain_trainset(ratio=0.05, target_label=7)
    make_retrain_testset(target_label=7)






