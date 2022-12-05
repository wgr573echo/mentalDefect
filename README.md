# 小样本金属损伤分类
- 参考 https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch#prototypical-networks-for-few-shot-learning-in-pytorch
- baseline：ProtoNet
- 数据：①GC10-DET ②NEU-CL ③天池铝材比赛数据集
### 数据情况
-  GC10-NET

|  crease折痕  | crescent_gap新月形缝隙  | inclusion夹杂物  | oil_spot油斑 | punching_hole冲孔  | rolled_pit轧坑  | silk_spot丝斑  | waist_folding焊缝  | water_spot水斑  | welding_line腰部折痕  |
|  ----  | ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| 横向条纹  | 半圆形缺失 | 纹理突兀的小斑点 | 明显 | 一个洞 | 点、片、条状坑 | 丝状细细的 | 不太明显 | 明显 | 一条皱纹 |

- - 问题：一张图片上可能有多种损伤，作为单类样本训练可能有影响 
<br />
- NEU-DET

|  开裂(Cr)  | 夹杂(In)  | 斑块(Pa)  | 点蚀(PS) | 氧化铁皮压入(RS)  | 划痕(Sc)  |
|  ----  | ----  |  ----  |  ----  |  ----  |  ----  |
| 丝状  | 条状 | 大面积斑块 | 密密麻麻的黑色 | 较为密集的凹坑 | 白色线条 |

- - 问题：conv64的maxpool天然选择最大值，因此需要前景为白色，这里的损伤除了划痕（Sc）都是损伤前景为黑色，需要反色
### 训练设定
- 小样本设定根据是否有辅助集分为两种
- - 无辅助集参与，直接在小样本数据上进行训练和测试，训练和测试的类别相同，容易过拟合，效果较差
- - 有辅助集参与，辅助集是和小样本同分布的高资源数据，和小样本集类别不同。训练阶段借助辅助集（或者辅助集+小样本训练集），测试阶段在小样本集上进行推理。为了贴近小样本测试的设定，即在有限的资源上完成下游任务。最常见的方案是使用元学习的方法，也称为N-way K-shot，可以加快收敛。是将训练过程分成多task完成，每个task由支持集和查询集组成，支持集和查询集类别相同，一共N个类别，每个类别的支持集的样本数量为K，在查询集上完成损失计算和反传。
- 本次实验也采用 N-way K-shot的设定，辅助集和小样本集来自同一个公开数据集，如GC10-DET，我们选择前7个类别作为辅助集进行训练，重新构造后三个class作为小样本问题进行测试。

### 训练结果
- 见【腾讯文档】方法 https://docs.qq.com/sheet/DRFRCU0twemFmTmxo?tab=BB08J2

