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
### 训练结果

<table>
   <tr>
      <td rowspan="2">方法</td>
      <td rowspan="2">骨干网络</td>
      <td rowspan="2">预训练参数</td>
      <td colspan="2">GC10-DET</td>
      <td colspan="2">NEU-CL</td>
      <td colspan="2">天池铝材表面瑕疵</td>
   </tr>
   <tr>
      <td>5way1shot</td>
      <td>5way5shot</td>
      <td>5way1shot</td>
      <td>5way5shot</td>
      <td>5way1shot</td>
      <td>5way5shot</td>
   </tr>
   <tr>
      <td>ProtoNet</td>
      <td>conv64</td>
      <td>/</td>
      <td></td>
      <td>72.45%</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
</table>

