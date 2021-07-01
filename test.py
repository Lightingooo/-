# -*- coding: utf-8 -*-
# @Time    : 2020/12/12 15:36
# @Author  : Lightning
# @FileName: test.py
# @Software: PyCharm

import logging
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from designedModel import lighntingOne
from utils import imageDataset, showROCAndAUC


def testAndShow(model, testData, batch_size = 8, description = "", ROC=False):
    yTrue = []
    yPred = []
    yProb = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    testloader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, num_workers=2)
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        yTrue += labels.cpu().numpy().tolist()
        yPred += preds.cpu().numpy().tolist()
        yProb += outputs.cpu().detach().numpy().tolist()
    # print(yTrue)
    # print(yPred)
    # print(yProb)
    torch.cuda.empty_cache()
    print("Classification report:\n", classification_report(yTrue, yPred, digits=5))
    print("Confuse Matrix:\n", confusion_matrix(yTrue, yPred))
    if ROC:
        showROCAndAUC(5, yTrue, yProb, description)


if __name__ == "__main__":
    # 全局日志管理
    logging.basicConfig(level=logging.INFO)

    # 根目录文件
    rootDirPath = "C:\\Users\\lenovo\\Desktop\\neuturalClassify\\imagesTest\\"

    # 读取测试集
    testData = imageDataset(5, rootDirPath, 448)
    net1 = lighntingOne()
    net1.load_state_dict(torch.load("./weights/1_cross_best.pth"))
    testAndShow(net1, testData, batch_size = 8, description = "",ROC =  True)
