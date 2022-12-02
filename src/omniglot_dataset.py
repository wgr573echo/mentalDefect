# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}


class OmniglotDataset(data.Dataset):
    vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='..' + os.sep + 'dataset', transform=None, target_transform=None, download=True):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(OmniglotDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')
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
        rot = self.all_items[index][-1]
        img = str.join(os.sep, [self.all_items[index][2], filename]) + rot #图片路径+旋转角度
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.splits_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k, url in self.vinyals_split_sizes.items():
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[-1]
            file_path = os.path.join(self.root, self.splits_folder, filename.split('/')[-1])
            with open(file_path, 'wb') as f:
                f.write(data.read())

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[2]
            file_path = os.path.join(self.root, self.raw_folder, filename.split("/")[-1])
            
            with open(file_path, 'wb') as f:
                f.write(data.read())
            orig_root = os.path.join(self.root, self.raw_folder)
            print("== Unzip from " + file_path + " to " + orig_root)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(orig_root)
            zip_ref.close()
        file_processed = os.path.join(self.root, self.processed_folder)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(orig_root, p)):
                shutil.move(os.path.join(orig_root, p, f), file_processed)
            os.rmdir(os.path.join(orig_root, p))
        print("Download finished.")


def find_items(root_dir, classes):
    retour = []
    rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
    # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    # root：当前路径 dirs：当前路径下的文件夹名 files：当前路径下的文件
    for (root, dirs, files) in os.walk(root_dir): 
        for f in files:
            r = root.split(os.sep)
            lr = len(r)
            label = r[lr - 2] + os.sep + r[lr - 1] #标签是语言+第几个字母
            for rot in rots:
                if label + rot in classes and (f.endswith("png")):
                    retour.extend([(f, label, root, rot)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] + i[-1] in idx):
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx

# 50个语言的xx个单词，每个单词旋转0 90 180 270 四次，一共4112个，在train.txt中每一行格式为 语言/第几个单词/旋转角度
def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    return classes


def load_img(path, idx):
    path, rot = path.split(os.sep + 'rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)

    return x
