# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 15:55
# @Author  : Lightning
# @FileName: train.py
# @Software: PyCharm
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn

from test import testAndShow
from utils import imagePathDataset, kFlodCrossValidaton, splitList
from designedModel import lighntingOne, fullyCon, vgg, resNet



classes = ('basketballMix', 'basketballRed', 'volleyballBlue', 'volleyballBlue', 'pillar')
# 0.0001 for first
hyp = {'lr': 0.0001
       }


def train(model, trainData, validData, description=""):
    # 初始化种子, 保证每次训练的时候结果相似
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True

    # 初始化GPU，如果能用的话
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if opt.resume:
        model.load_state_dict(torch.load(opt.weights))
    print("use:", device)

    # 加载dataloader
    trainloader = torch.utils.data.DataLoader(trainData, batch_size=opt.batch_size - 1, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validData, batch_size=1, shuffle=True, num_workers=2)

    # 定义误差函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)

    # 定时更新学习速率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)

    # 定义最高精确度
    bestAccuarcy = 0
    with open(opt.fileName + description + ".txt", 'w') as resultTxt:
        for epoch in range(opt.epoch):
            totalLoss = 0.0
            running_loss = 0.0
            totalNum = 0
            for i, data in enumerate(trainloader, 0):
                # 获取输入
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度优化
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 计算并且输出误差
                totalNum += 1
                totalLoss += loss.item()
                running_loss += loss.item()
                if i % 5 == 4:
                    logging.info('[%d, %5d] train loss: %.3f' %
                                 (epoch + 1, i + 1, running_loss / 5))
                    running_loss = 0.0
            # 验证
            total = 0
            count = 0
            torch.cuda.empty_cache()
            for i, data in enumerate(validloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total = total + len(labels)
                for j in range(len(preds)):
                    if preds[j] == labels[j]:
                        count = count + 1

            # 衡量loss以决定是否降低学习速率
            scheduler.step(totalLoss / totalNum)

            # print epoch loss and epoch accuarcy
            tempAccuarcy = float(count) / total
            print(description, "Epoch ", epoch + 1, " test accuarcy: ", tempAccuarcy, ", mean epoch loss: ",
                  totalLoss / totalNum)
            # 输出epoch accuarcy到文件
            resultTxt.write(description + 'Epoch %5d , test accuarcy: %8f , mean epoch loss: %8f\n' % (
                epoch + 1, tempAccuarcy, totalLoss / totalNum))
            if tempAccuarcy > bestAccuarcy:
                torch.save(model.state_dict(), "./weights/" + description + "best.pth")
                bestAccuarcy = tempAccuarcy
            if epoch % 10 == 9:
                print("Save model in ", opt.weights, " for epoch ", epoch + 1)
                torch.save(model.state_dict(), "./weights/" + description + "last.pth")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 全局日志管理
    logging.basicConfig(level=logging.INFO)

    # 全局参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=80, help="the total epoch")
    parser.add_argument('--batch-size', type=int, default=16, help="the total size before update the parameters") # 1:16  2:9
    parser.add_argument('--img-size', type=int, default=224, help='input image size to the neutral network')
    parser.add_argument('--weights', type=str, default='weights/first.pth', help='initial weights path')
    parser.add_argument('--fileName', type=str, default='results/net1', help='initial weights path')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--testStyle', type=int, default=0, help="0 for last and 1 for best")
    opt = parser.parse_args()

    # k折交叉验证
    rootDirPath = "./images/"
    x, y, skfs = kFlodCrossValidaton(rootDirPath, 5, 4)
    k = 1
    for train_index, test_index in skfs:
        # 划分训练集和验证集
        valid_index, train_index = splitList(train_index.tolist(), ratio=0.2)
        train_index = np.array(train_index)
        valid_index = np.array(valid_index)

        # 获取训练集、验证集、测试集
        X_train, X_test, X_valid = x[train_index], x[test_index], x[valid_index]
        y_train, y_test, y_valid = y[train_index], y[test_index], y[valid_index]
        logging.info("Train size: %5d, Valid Size: %5d, Test size: %5d" %(len(y_train), len(y_valid),len(y_test)))
        trainData = imagePathDataset(X_train, y_train, opt.img_size)
        validData = imagePathDataset(X_valid, y_valid, opt.img_size)
        testData = imagePathDataset(X_test, y_test, opt.img_size)

        # 以下是四个网络，分别为全连接、卷积、vgg和残差网络，选择自己想训练的，
        # 把注释去掉，就可以开始训乱。
        netTrain = None
        # 第一个网络
        # netTrain = fullyCon()

        # 第二个网络
        # netTrain = lighntingOne()

        # 第三个网络
        # netTrain = vgg(5)

        # 第四个网络
        netTrain = resNet(5)



        train(netTrain, trainData, validData, str(k) + "_cross_")

        # 如果采用精度最好的模型测试
        # 否则采用最后一个模型
        if opt.testStyle == 1:
            netTrain.load_state_dict(torch.load("./weights/" + str(k) + "_cross_best.pth"))
        # 测试
        testAndShow(netTrain, testData, batch_size=6, description=str(k) + "_cross_", ROC=True)

        k += 1
