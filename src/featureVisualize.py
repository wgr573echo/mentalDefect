# -*- coding: UTF-8 -*-
import torch
from protonet_split import ProtoNet
import torch.nn as nn
import torchvision.transforms as tfs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from  collections import OrderedDict
 
 
class cal_cam(nn.Module):
    def __init__(self, feature_layer="encoder3"):
        super(cal_cam, self).__init__()
        self.model = ProtoNet()
        path = "../myoutput/best_model.pth"
        pretrained_dict=torch.load(path)
        # now_model = self.model.state_dict()
        
        modify_predict= OrderedDict()
        for k, v in pretrained_dict.items():
            k_s = k.split(".")
            if k_s[1] == '0':
                k = k_s[0]+'1.'+k_s[2]+'.'+k_s[3]
            if k_s[1] == '2':
                k = k_s[0]+'2.'+k_s[2]+'.'+k_s[3]
            if k_s[1] == '4':
                k = k_s[0]+'3.'+k_s[2]+'.'+k_s[3]
            if k_s[1] == '6':
                k = 'last.'+k_s[2]+'.'+k_s[3]
            modify_predict[k] = v
        self.model.load_state_dict(modify_predict)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
        # 要求梯度的层
        self.feature_layer = feature_layer
        # 记录梯度
        self.gradient = []
        # 记录输出的特征图
        self.output = []
        
 
        self.trainsform = tfs.Compose([
            tfs.Resize([145,145]),
            tfs.ToTensor()
        ])
 
 
    def get_feature(self):
        return self.output[-1][0]
 
    def process_img(self, input):
        input = self.trainsform(input)
        input = input.unsqueeze(0)
        return input
 
    # 计算最后一个卷积层的梯度，输出梯度和最后一个卷积层的特征图
    def getGrad(self, input_):
        input_ = input_.to(self.device)
        num = 1
        for name, module in self.model._modules.items():
            if (num == 1):
                input = module(input_)
                num = num + 1
                continue
            # 是待提取特征图的层
            if (name == self.feature_layer):
                input = module(input)
                self.output.append([input])
            # 普通的层
            else:
                input = module(input)
 
        #处理多通道特征图
        self.output[0][0] = self.output[0][0].detach()
        before_f = np.array(self.output[0][0].cpu())[0]
        # 64*9*9 多通道特征图合成
        after_f = np.zeros((18,18))
        for channel in before_f:
            index_x = 0
            for i in channel:
                index_y = 0
                for j in i:
                    after_f[index_x][index_y] += j
                    index_y = index_y + 1
                    if index_y == 18 : break
                index_x = index_x + 1
                if index_x == 18 : break

        return after_f
 
    # 计算CAM
    def getCam(self, grad_val, feature):
        # 对特征图的每个通道进行全局池化
        alpha = torch.mean(grad_val, dim=(2, 3)).cpu()
        feature = feature.cpu()
        # 将池化后的结果和相应通道特征图相乘
        cam = torch.zeros((feature.shape[2], feature.shape[3]), dtype=torch.float32)
        for idx in range(alpha.shape[1]):
            cam = cam + alpha[0][idx] * feature[0][idx]
        # 进行ReLU操作
        cam = np.maximum(cam.detach().numpy(), 0)
 
        plt.imshow(cam)
        plt.colorbar()
        plt.savefig("cam.jpg")
 
        # 将cam区域放大到输入图片大小
        cam_ = cv2.resize(cam, (224, 224))
        cam_ = cam_ - np.min(cam_)
        cam_ = cam_ / np.max(cam_)
        plt.imshow(cam_)
        plt.savefig("cam_.jpg")
        cam = torch.from_numpy(cam)
 
        return cam, cam_
 
    def show_img(self, cam_, img):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_), cv2.COLORMAP_JET)
        cam_img = 0.3 * heatmap + 0.7 * np.float32(img)
        cv2.imwrite("img.jpg", cam_img)
 
    def __call__(self, img_root):
        img = Image.open(img_root)
        plt.imshow(img,cmap="gray")

        input = self.process_img(img)
        featuremap = self.getGrad(input)
        heatmap = cv2.applyColorMap(np.uint8(255 * featuremap), cv2.COLORMAP_JET)
        cv2.imshow("ft",heatmap)
        img.resize(145,71)
        self.show_img(featuremap, img)
        return cam
 
 
if __name__ == "__main__":
    cam = cal_cam()
    img_root = "../mydataset/gc10/data/crease/img_01_425382900_00002.jpg"
    cam(img_root)