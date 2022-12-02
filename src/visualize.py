from pickle import FROZENSET
from pyexpat import model
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch


from protonet import ProtoNet
from myDataset import myDataset


cuda = True
model_path = "../myoutput/best_model.pth"
img_fts =  []
img_ys = []
img_ysName = []


def detect_image(image_data,model):

    with torch.no_grad():
        images = image_data.float().cuda()
        model.cuda()
        
        output = model(images)
        img_fts.append(output.cpu().numpy())

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d) #x和y变成同样维度的两个向量300*60*64

    return torch.pow(x - y, 2).sum(2) # 求欧氏距离，并在特征那个维度求和


if __name__ == "__main__":



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict=torch.load(model_path,map_location=device)
    forward_model = ProtoNet().to(device)
    forward_model_dict = forward_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in forward_model_dict}
    forward_model_dict.update(pretrained_dict)
    forward_model.load_state_dict(forward_model_dict)

    forward_model = forward_model.eval()

    from torchvision import datasets

    train_dataset = myDataset('val','../mydataset/gc10')
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler)

    for idx, (image, label) in enumerate(train_loader):

        detect_image(image,forward_model)
        img_ys.append(label.numpy())
    
    
    
    
    # img_fts = np.squeeze(img_fts)
    # #t-sne可视化特征分布
    # from sklearn import manifold
    # import matplotlib.pyplot as plt

    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # X_tsne = tsne.fit_transform(img_fts) #channel*2
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化后点集

    # plt.figure(figsize=(8, 8))
    # axes = plt.subplot(111)

    # cr_x,cr_y = [],[]
    # cg_x,cg_y = [],[]
    # in_x,in_y = [],[]
    # oi_x,oi_y = [],[]
    # ph_x,ph_y = [],[]
    # rp_x,rp_y = [],[]
    # ss_x,ss_y = [],[]
    # wf_x,wf_y = [],[]
    # ws_x,ws_y = [],[]
    # wl_x,wl_y = [],[]

    # for i in range(X_norm.shape[0]):
    #     if img_ys[i] == 0:
    #         cr_x.append(X_norm[i][0])
    #         cr_y.append(X_norm[i][1])
    #     if img_ys[i] == 1:
    #         cg_x.append(X_norm[i][0])
    #         cg_y.append(X_norm[i][1])
    #     if img_ys[i] == 2:
    #         in_x.append(X_norm[i][0])
    #         in_y.append(X_norm[i][1])
    #     if img_ys[i] == 3:
    #         oi_x.append(X_norm[i][0])
    #         oi_y.append(X_norm[i][1])
    #     if img_ys[i] == 4:
    #         ph_x.append(X_norm[i][0])
    #         ph_y.append(X_norm[i][1])
    #     if img_ys[i] == 5:
    #         rp_x.append(X_norm[i][0])
    #         rp_y.append(X_norm[i][1])
    #     if img_ys[i] == 6:
    #         ss_x.append(X_norm[i][0])
    #         ss_y.append(X_norm[i][1])
    #     if img_ys[i] == 7:
    #         wf_x.append(X_norm[i][0])
    #         wf_y.append(X_norm[i][1])
    #     if img_ys[i] == 8:
    #         ws_x.append(X_norm[i][0])
    #         ws_y.append(X_norm[i][1])
    #     if img_ys[i] == 9:
    #         wl_x.append(X_norm[i][0])
    #         wl_y.append(X_norm[i][1])


    # # type0 = axes.scatter(cr_x, cr_y, s=4,label='crease')
    # # type1 = axes.scatter(cg_x, cg_y, s=6,label='crescent_gap')
    # type2 = axes.scatter(in_x, in_y, s=8,label='inclusion')
    # # type3 = axes.scatter(oi_x, oi_y, s=10,label='oil_spot')
    # type4 = axes.scatter(ph_x, ph_y, s=12,label='punching_hole')
    # # type5 = axes.scatter(rp_x, rp_y, s=14,label='rolled_pit')
    # # type6 = axes.scatter(ss_x, ss_y, s=16,label='silk_spot')
    # # type7 = axes.scatter(wf_x, wf_y, s=18,label='waist folding')
    # # type8 = axes.scatter(ws_x, ws_y, s=20,label='water_spot')
    # # type9 = axes.scatter(wl_x, wl_y, s=22,label='welding_line')

    # plt.legend(loc=2)
    # plt.savefig('1.png', bbox_inches='tight')



    # plt_x = X_norm[:,0]
    # plt_y = X_norm[:,1]
    # scatter = plt.scatter(plt_x,plt_y,color=plt.cm.Set1(img_ys))
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('1.png', bbox_inches='tight')


    # 需要衡量欧式距离，再进行混淆矩阵
    


    # 对预测的结果绘制混淆矩阵
    # from sklearn.metrics import confusion_matrix
    # import matplotlib.pyplot as plt
    # y_true = img_ys
    # y_pred = img_fts
    # C = confusion_matrix(y_true, y_pred)
    # plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
    # # plt.colorbar()

    # for i in range(len(C)):
    #     for j in range(len(C)):
    #         plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()
