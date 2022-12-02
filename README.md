# 小样本金属损伤分类
- 参考 https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch#prototypical-networks-for-few-shot-learning-in-pytorch
- baseline：ProtoNet
- 数据：①GC10-DET ②NEU-CL ③天池铝材比赛数据集
- 数据性质
###  GC10-NET

|  crease折痕  | crescent_gap新月形缝隙  | inclusion夹杂物  | oil_spot油斑 | punching_hole冲孔  | rolled_pit轧坑  | silk_spot丝斑  | waist_folding焊缝  | water_spot水斑  | welding_line腰部折痕  |
|  ----  | ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| 横向条纹  | 半圆形缺失 | 纹理突兀的小斑点 | 明显 | 一个洞 | 点、片、条状坑 | 丝状细细的 | 不太明显 | 明显 | 一条皱纹 |
- 问题：一张图片上可能有多种损伤，作为单类样本训练可能有影响 
- 训练结果：

|  模型  |  ProtoNet（conv64）  | 
|  ----  |  ----  |
|  accuracy  |  72.445%  |

<table>
   <tr>
      <td>方法</td>
      <td>骨干网络</td>
      <td>预训练参数</td>
      <td>GC10-DET</td>
      <td></td>
      <td>NEU-CL</td>
      <td></td>
      <td>天池铝材表面瑕疵</td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td></td>
      <td></td>
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
   <tr>
      <td></td>
   </tr>
</table>

