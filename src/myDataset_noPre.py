# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os


IMG_CACHE = {}


class myDataset(data.Dataset):
    
    splits_folder = "splits/noPre"
    processed_folder = 'data'

    def __init__(self, mode='train', root='..' + os.sep + 'mydataset', transform=None, target_transform=None):
        
        super(myDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # classes：split的txt文件里每一行代表一个类别，同一个字母旋转4次即4个类别
        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        
        # items：每个语言的每个单词文件下的所有图片旋转四次，一共82240个item
        # 一个item：四部分组成：当前文件名XX.JPG、真值（语言、第几个单词）、路径、旋转多少
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes)

        # 根据每个语言的每个单词分别旋转四次，依次编号，从0开始
        self.idx_classes = index_classes(self.all_items)
        # path：图片路径+旋转角度 # self.y：通过idx_classes的映射，将每一个item的类名映射为类名id
        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))]) #len(self)：call __len__函数，求的是all_items的长度

        self.x = map(load_img, paths, range(len(paths))) #map函数（函数1，列表）：对列表中每个元素都进行函数1的运算
        self.x = list(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        img = str.join(os.sep, [self.all_items[index][2], filename])
        target = self.idx_classes[self.all_items[index][1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))


def find_items(root_dir, classes):
    retour = []
    # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    # root：当前路径 dirs：当前路径下的文件夹名 files：当前路径下的文件
    for (root, dirs, files) in os.walk(root_dir): 
        for f in files:
            r = root.split(os.sep)
            label = r[-1]
            if label in classes and (f.endswith("jpg")):
                retour.extend([(f, label, root)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    return classes


def load_img(path, idx):
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x

    x = x.resize((145, 71))

    shape = 1, x.size[0], x.size[0]
    x = np.array(x, np.float32, copy=False)
    # 填补为一个正方形图像
    x = np.pad(x, ((37,37),(0,0)), 'minimum')
   
    # import matplotlib.pyplot as plt
    # plt.imshow(x,cmap="gray")
    # plt.show()

    x = torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)
    return x
