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
- 小样本设定根据是否有辅助数据集分为两路
- - 第一路没有辅助数据集,纯粹在小样本的基础上解决问题,称为K-shot,训练\验证和测试数据都来自小样本数据,相同类别.非常容易过拟合.
- - 第二路借助高资源的辅助数据集完成任务,辅助集和小样本类别不同,在辅助集(或者辅助集+小样本集)上训练,在小样本集上进行测试.辅助集和小样本集最好同分布,否则影响实验效果.常见的元学习的方法可以很快提升收敛速度,所以通用的方法是使用N-way K-shot的设定,即将训练分成多task完成,每个task为了适应小样本的task情况,由支持集+查询集组成,支持集和查询集类别相同,支持集和查询集一共N个类别,每个类别的支持集提供K张图片.即利用支持集的信息,完成查询集的推理,得到的损失进行反传.
- 本次实验的小样本设定为N-way K-shot方式,训练样本和测试样本来自同一个数据集,但类别不同,比如GC10-DET,选择表格中前7个作为辅助集,构造后3个为小样本集进行测试.
### 训练结果

- 见 https://docs.qq.com/sheet/DRFRCU0twemFmTmxo?tab=BB08J2

