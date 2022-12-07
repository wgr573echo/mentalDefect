# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np

# class PrototypicalLoss(Module):
#     '''
#     Loss class deriving from Module for the prototypical loss function defined below
#     '''
#     def __init__(self, n_support):
#         super(PrototypicalLoss, self).__init__()
#         self.n_support = n_support

#     def forward(self, input, target):
#         return prototypical_loss(input, target, self.n_support)


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


def prototypical_loss(mode, input, target, n_support,temperature):
    '''
    计算度量损失
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu') # backbone encode 得到的特征

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    
    classes = torch.unique(target_cpu) #去除重复，一共这一批5类分别是哪五类
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support #用一个class找出query的sample数

    support_idxs = list(map(supp_idxs, classes)) #当前50个feature对应的顺序下选出前五个作为support,5*5

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs]) #prototypes：60*64（60个class的平均proto特征）

    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1) #当前50个feature对应的顺序下选出后五个作为query
    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes) # (classnum * querysamplenum) X 这一批的类别数
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1) #拆分samples为每个类别的5个samples

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long() # 给25个val sample分别标label：类别数×query样本数×1

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    # ------------------------------
    # 计算对比损失
    # 对比的是：负样本对之间的距离（cosine矩阵）+ prototypes之间的距离（L2正则）
    if mode == "train":
        normMatrix = []
        # 先得到50行每个行向量（特征向量）的归一化向量
        for row in input_cpu:
            norm = np.linalg.norm(row.detach().numpy(),ord=1)
            norm_row = row * (1/norm)
            normMatrix.append(norm_row)
        normMatrix = torch.stack(normMatrix)
        # 50*64的特征矩阵，计算每个位置上64channel的特征向量之间的cosine值
        cosine_matrix = torch.matmul(normMatrix, normMatrix.T)
        # 再整体÷温度系数t，参考FSCE选取参数
        cosine_matrix = cosine_matrix * (1/float(temperature))
        # # 性能稳定？
        logits_max, _ = torch.max(cosine_matrix, dim=1, keepdim=True)
        logits = cosine_matrix - logits_max.detach()

        # 求出相同label的mask矩阵
        mask = []
        for i in range(target_cpu.shape[0]):
            tmp = []
            for j in range(target_cpu.shape[0]):
                if target_cpu[j] == target_cpu.T[i]:
                    tmp.append(True)
                else:
                    tmp.append(False)
            mask.append(tmp)
        mask = torch.tensor(mask).float()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(50).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(cosine_matrix) * logits_mask
        log_prob = cosine_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        closs = - mean_log_prob_pos.mean()

        loss_val = loss_val + closs


    return loss_val,  acc_val
