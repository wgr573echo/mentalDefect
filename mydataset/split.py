# 最终的效果是txt文件每一行：损伤名称（作为真值）+ 四类旋转
# 一共三个文件：train、val、test
import os

testNum = 10
trainNum = 25
valNum = 5

path = "./data"
root = "./splits/"

classnames = []

import random

if __name__ == "__main__":
    for file in os.listdir(path):
        for r in [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']:
            classnames.append(file + r)
    
    # 生成train、val、test的随机数组
    all = random.sample(range(0,40),40)
    trainID = all[:trainNum]
    valID = all[trainNum:trainNum+valNum]
    testID = all[trainNum+valNum:]
    
    for mode in ["train","val","test"]:
        savepath = root + mode + ".txt"
        f = open(savepath,"w")
        if mode == "train":
            for i in range(len(trainID)):
                f.write(classnames[i]+"\n")
        if mode == "val":
            for i in range(len(valID)):
                f.write(classnames[i]+"\n")
        if mode == "test":
            for i in range(len(testID)):
                f.write(classnames[i]+"\n")
        f.close()

