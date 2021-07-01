# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 16:17
# @Author  : Lightning
# @FileName: utils.py
# @Software: PyCharm
from __future__ import print_function
import os
import sys
import cv2
import json
import torch
import shutil
import random
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
sys.setrecursionlimit(1000000)
# 训练集占的数量
factor = 0.7


# 用于坐标转换
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 用于在一个整数范围内产生n个随机数，注意范围内的整数个数要小于所需的随机数的个数
# 第一个参数表示开始的范围，第二个参数表示结束的范围，第三个表示产生随机数的个数
# 最后返回的是一个列表
class CreateRandomPage:
    def __init__(self, begin, end, needcount):
        assert (end - begin + 1) >= needcount, "no way to generate"
        self.begin = begin
        self.end = end
        self.needcount = needcount
        self.resultlist = []
        self.count = 0

    def createrandompage(self):
        tempInt = random.randint(self.begin, self.end)
        if (self.count < self.needcount):
            if (tempInt not in self.resultlist):
                self.resultlist.append(tempInt)  # 将长生的随机数追加到列表中
                self.count += 1
            return self.createrandompage()  # 在此用递归的思想
        return self.resultlist


# 将给定的图片路径和标签路径下的图片和标签转化成
# yolo规定的特殊coco格式
def tranToYoloData(imageDir_path, targetDir_path, labelDir_path, imagePathString):
    imageDir = Path(imageDir_path)
    targetDir = Path(targetDir_path)
    labelDir = Path(labelDir_path)

    # must have source images directory
    assert imageDir.is_dir(), "No such image directory"
    assert labelDir.is_dir(), "No such label directory"
    if targetDir.is_dir():
        os.remove(targetDir_path)
    else:
        os.mkdir(targetDir_path)

    # must have at least one image
    imageFilesPath = [x for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
    assert len(imageFilesPath) > 0, "No images"
    trainNum = int(len(imageFilesPath) * factor)
    assert trainNum > 0, "too small factor"

    # generate image path
    randomNum = CreateRandomPage(1, len(imageFilesPath), trainNum)
    randomNums = randomNum.createrandompage()

    # create train dataset and val dataset
    os.mkdir(targetDir_path + '/' + 'images')
    os.mkdir(targetDir_path + '/' + 'images' + '/' + 'train2017')
    os.mkdir(targetDir_path + '/' + 'images' + '/' + 'val2017')
    os.mkdir(targetDir_path + '/' + 'labels')
    os.mkdir(targetDir_path + '/' + 'labels' + '/' + 'train2017')
    os.mkdir(targetDir_path + '/' + 'labels' + '/' + 'val2017')
    with open(targetDir_path + "/" + "train2017.txt", "w") as f, \
            open(targetDir_path + "/" + "val2017.txt", "w") as m, \
            open(targetDir_path + "/" + "train2017.shapes", "w") as tm, \
            open(targetDir_path + "/" + "val2017.shapes", "w") as vm:
        for i in imageFilesPath:
            j = int(i.replace('.jpg', ''))
            imageShape = cv2.imread(imageDir_path + '/' + i).shape
            if Path(labelDir_path + '/' + str(j) + '.txt').is_file():
                if j in randomNums:
                    f.writelines(imagePathString + 'train2017/' + i + '\n')
                    shutil.copy(imageDir_path + '/' + i, targetDir_path + '/' + 'images' + '/' + 'train2017' + '/' + i)
                    shutil.copy(labelDir_path + '/' + str(j) + '.txt',
                                targetDir_path + '/' + 'labels' + '/' + 'train2017' + '/' + str(j) + '.txt')
                    tm.writelines(str(imageShape[1]) + ' ' + str(imageShape[0]) + '\n')
                else:
                    shutil.copy(imageDir_path + '/' + i, targetDir_path + '/' + 'images' + '/' + 'val2017' + '/' + i)
                    shutil.copy(labelDir_path + '/' + str(j) + '.txt',
                                targetDir_path + '/' + 'labels' + '/' + 'val2017' + '/' + str(j) + '.txt')
                    m.writelines(imagePathString + 'val2017/' + i + '\n')
                    vm.writelines(str(imageShape[1]) + ' ' + str(imageShape[0]) + '\n')


def tranToVOCData(imageDir_path, targetDir_path, labelDir_path, objectName):
    imageDir = Path(imageDir_path)
    targetDir = Path(targetDir_path)
    labelDir = Path(labelDir_path)

    # must have source images directory
    assert imageDir.is_dir(), "No such image directory"
    assert labelDir.is_dir(), "No such label directory"

    # must have at least one image
    imageFilesPath = [x for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
    assert len(imageFilesPath) > 0, "No images"
    labelFilesPath = [x for x in os.listdir(labelDir_path) if x.endswith('.xml')]
    assert len(labelFilesPath) > 0, "No images"
    assert len(imageFilesPath) == len(labelFilesPath), "No match in images and its' xml"

    os.mkdir(targetDir_path + '/' + objectName)
    os.mkdir(targetDir_path + '/' + objectName + '/' + 'Annotations')
    os.mkdir(targetDir_path + '/' + objectName + '/' + 'ImageSets')
    os.mkdir(targetDir_path + '/' + objectName + '/' + 'ImageSets' + '/' + 'Main')
    os.mkdir(targetDir_path + '/' + objectName + '/' + 'JPEGImages')
    with open(targetDir_path + '/' + objectName + '/' + 'ImageSets' + '/' + 'Main' + '/' + "trainval.txt", "w") as f:
        for i in imageFilesPath:
            shutil.copy(imageDir_path + '/' + i, targetDir_path + '/' + objectName + '/' + 'JPEGImages' + '/' + i)
            shutil.copy(labelDir_path + '/' + i.replace('.jpg', '.xml'),
                        targetDir_path + '/' + objectName + '/' + 'Annotations' + '/' + i.replace('.jpg', '.xml'))
            f.writelines(i.replace('.jpg', '') + '\n')


# 将coco的json文件转化成yolo规定的txt文件格式
# json_file:json文件路径
# ana_txt_save_path：txt文件保存文件夹
def tranJsonToTxt(json_file, ana_txt_save_path):
    data = json.load(open(json_file, 'r'))

    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    for img in data['images']:
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        ana_txt_name = filename.split(".")[0] + ".txt"  # 对应的txt名字，与jpg一致
        print(ana_txt_name)
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')

        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (ann["category_id"], box[0], box[1], box[2], box[3]))
        f_txt.close()


# 对文件夹下所有图片进行顺序命名
def renameSeqImage(imageDir_path):
    renameRandomImage(imageDir_path)
    imageDir = Path(imageDir_path)

    # must have source images directory
    assert imageDir.is_dir(), "No such image directory"

    # must have at least one image
    imageFilesPath = [x for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
    assert len(imageFilesPath) > 0, "No images"

    i = 1
    for filePath in imageFilesPath:
        if filePath.endswith('.jpg'):
            os.rename(imageDir_path + '/' + filePath, imageDir_path + '/' + str(i) + '.jpg')
            i += 1
        else:
            logging.warning('There are no \'.jpg\' files in directory path')


# 对文件夹下所有图片进行随机命名
def renameRandomImage(imageDir_path):
    imageDir = Path(imageDir_path)

    # must have source images directory
    assert imageDir.is_dir(), "No such image directory"

    # must have at least one image
    imageFilesPath = [x for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
    assert len(imageFilesPath) > 0, "No images"

    i = 1
    for filePath in imageFilesPath:
        if filePath.endswith('.jpg'):
            os.rename(imageDir_path + '/' + filePath, imageDir_path + '/random' + str(i) + '.jpg')
            i += 1
        else:
            logging.warning('There are no \'.jpg\' files in directory path')


# 向源数据集中增加新的数据，将其整合到源数据集
# 注意，源数据集和增加数据集其中的图片名称必须要从1开始递增到其数据集的总数
# 文件夹名称之后不加/
def addImagesAndLabels(sourceImagePaths, sourceLabelPaths, addImagePaths, addLabelPaths):
    sourceImageDir = Path(sourceImagePaths)
    addImageDir = Path(addImagePaths)
    sourceLabelDir = Path(sourceLabelPaths)
    addLabelDir = Path(addLabelPaths)
    # must have source images directory
    assert sourceImageDir.is_dir(), "No source image directory"
    assert addImageDir.is_dir(), "No target image directory"
    assert sourceLabelDir.is_dir(), "No source label directory"
    assert addLabelDir.is_dir(), "No target label directory"
    # must have at least one image
    sourceImageFilesPath = [x for x in os.listdir(sourceImageDir) if x.endswith('.jpg')]
    addImageFilesPath = [x for x in os.listdir(addImageDir) if x.endswith('.jpg')]
    sourceLabelFilesPath = [x for x in os.listdir(sourceLabelDir) if x.endswith('.txt')]
    addLabelFilesPath = [x for x in os.listdir(addLabelDir) if x.endswith('.txt')]
    # must large than zero
    assert len(sourceImageFilesPath) > 0, "No source images"
    assert len(addImageFilesPath) > 0, "No target images"
    assert len(sourceLabelFilesPath) > 0, "No source label"
    assert len(addLabelFilesPath) > 0, "No target label"
    assert len(sourceImageFilesPath) == len(sourceLabelFilesPath), "No match in number of labels and number of images"
    assert len(addImageFilesPath) == len(addLabelFilesPath), "No match in number of labels and number of images"
    for i in range(len(sourceImageFilesPath) + 1, len(sourceImageFilesPath) + len(addImageFilesPath) + 1):
        shutil.copy(addImagePaths + '/' + str(i - len(sourceImageFilesPath)) + '.jpg',
                    sourceImagePaths + '/' + str(i) + '.jpg')

        shutil.copy(addLabelPaths + '/' + str(i - len(sourceLabelFilesPath)) + '.txt',
                    sourceLabelPaths + '/' + str(i) + '.txt')
    print("Move ", len(addImageFilesPath), " items!")


# 用于coco数据集
# 重命名图像和标签使得其有顺序
# 用于在未对图像命名有顺序的情况下就标定的情况
# 保证图像数量和label数量一样
def renameRandomImageAndLabel(imageDir_path, labelDir_path):
    imageDir = Path(imageDir_path)
    labelDir = Path(labelDir_path)

    # must have source images directory
    assert imageDir.is_dir(), "No such image directory"
    assert labelDir.is_dir(), "No such label directory"

    # must have at least one image
    imageFilesPath = [x for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
    labelFilesPath = [x for x in os.listdir(labelDir) if x.endswith('.txt')]
    assert len(imageFilesPath) > 0, "No images"
    assert len(labelFilesPath) > 0, "No labels"
    assert len(imageFilesPath) == len(labelFilesPath), "No match in number of labels and number of images "
    i = 1
    for imageFile in imageFilesPath:
        for labelFile in labelFilesPath:
            if imageFile.replace('.jpg', '') == labelFile.replace('.txt', ''):
                os.rename(imageDir_path + '/' + imageFile, imageDir_path + '/random' + str(i) + '.jpg')
                os.rename(labelDir_path + '/' + labelFile, labelDir_path + '/random' + str(i) + '.txt')
                i += 1
    i = 1
    for j in range(len(labelFilesPath)):
        os.rename(imageDir_path + '/random' + str(i) + '.jpg', imageDir_path + '/' + str(i) + '.jpg')
        os.rename(labelDir_path + '/random' + str(i) + '.txt', labelDir_path + '/' + str(i) + '.txt')
        i += 1


# 用于voc数据集
# 重命名图像和标签使得其有顺序
# 用于在未对图像命名有顺序的情况下就标定的情况
# 保证图像数量大于等于label数量，并且图像名称包含label名称
def renameRandomImageAndLabelEnhanced(imageDir_path, labelDir_path):
    imageDir = Path(imageDir_path)
    labelDir = Path(labelDir_path)

    # must have source images directory
    assert imageDir.is_dir(), "No such image directory"
    assert labelDir.is_dir(), "No such label directory"

    # must have at least one image
    imageFilesPath = [x for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
    labelFilesPath = [x for x in os.listdir(labelDir) if x.endswith('.xml')]
    assert len(imageFilesPath) > 0, "No images"
    assert len(labelFilesPath) > 0, "No labels"
    assert len(imageFilesPath) >= len(labelFilesPath), "No match in number of labels and number of images "

    i = 1
    flag = 0
    for imageFile in imageFilesPath:
        for labelFile in labelFilesPath:
            if imageFile.replace('.jpg', '') == labelFile.replace('.xml', ''):
                flag = 1
                os.rename(imageDir_path + '/' + imageFile, imageDir_path + '/random' + str(i) + '.jpg')
                os.rename(labelDir_path + '/' + labelFile, labelDir_path + '/random' + str(i) + '.xml')
                i += 1
        if flag == 0:
            os.rename(imageDir_path + '/' + imageFile, imageDir_path + '/random' + str(i) + 's.jpg')
        else:
            flag = 0
    i = 1
    for j in range(len(labelFilesPath)):
        os.rename(imageDir_path + '/random' + str(i) + '.jpg', imageDir_path + '/' + str(i) + '.jpg')
        os.rename(labelDir_path + '/random' + str(i) + '.xml', labelDir_path + '/' + str(i) + '.xml')
        i += 1


# 图片尺寸调整
def imageResize(imageDir_path):
    imageDir = Path(imageDir_path)
    # must have source images directory
    assert imageDir.is_dir(), "No such image directory"
    # must have at least one image
    imageFilesPath = [x for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
    assert len(imageFilesPath) > 0, "No images"

    for filePath in imageFilesPath:
        if filePath.endswith('.jpg'):
            # print(imageDir_path + '/' + filePath)
            image = cv2.imread(imageDir_path + '/' + filePath)
            image = cv2.resize(image, (960, int(image.shape[0] / (image.shape[1] / 960))))
            cv2.imwrite(imageDir_path + '/' + filePath, image)
        else:
            logging.warning('There are no \'.jpg\' files in directory path')


def splitList(full_list, shuffle=True, ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


# 从指定路径加载文件
class imageDataset:
    def __init__(self, classNumbers, rootDirPath, imageSize):
        self.imageVariablesPath = []
        self.imageLabels = []
        self.imageSize = imageSize
        for i in range(classNumbers):
            imageDir_path = rootDirPath + str(i + 1) + "\\"
            imageDir = Path(imageDir_path)
            assert imageDir.is_dir(), "No such image directory"
            imageFilesPath = [(imageDir_path + x) for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
            assert len(imageFilesPath) > 0, "No images"
            self.imageVariablesPath = self.imageVariablesPath + imageFilesPath
            self.imageLabels = self.imageLabels + [i for x in range(len(imageFilesPath))]
        self.length = len(self.imageLabels)
        # print(self.length)

    def __getitem__(self, index):
        image = cv2.imread(self.imageVariablesPath[index])
        image = image.astype(np.float32)
        image = image / 255.0
        # image = cv2.resize(image,(960,720))
        image = cv2.resize(image, (self.imageSize, self.imageSize))
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image)
        return image, self.imageLabels[index]

    def __len__(self):
        return self.length


# 根据路径和label加载文件
class imagePathDataset(Dataset):

    def __init__(self, x, y, imageSize):
        self.x = x
        self.y = y.tolist()
        self.imageSize = imageSize
        self.length = len(self.x)

    def __getitem__(self, index):
        image = cv2.imread(self.x[index])
        image = image.astype(np.float32)
        image = image / 255.0
        # image = cv2.resize(image,(960,720))
        image = cv2.resize(image, (self.imageSize, self.imageSize))
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image)
        return image, self.y[index]

    def __len__(self):
        return self.length


# k折交叉验证，输入为原始数据的规范路径以及类别数
# 返回一个可迭代对象。用于k次迭代获取训练集和测试集
def kFlodCrossValidaton(rootDirPath, classnumbers, k):
    imageVariablesPath = []
    imageLabels = []
    for i in range(classnumbers):
        imageDir_path = rootDirPath + str(i + 1) + "\\"
        imageDir = Path(imageDir_path)
        assert imageDir.is_dir(), "No such image directory"
        imageFilesPath = [(imageDir_path + x) for x in os.listdir(imageDir_path) if x.endswith('.jpg')]
        # print(imageFilesPath)
        assert len(imageFilesPath) > 0, "No images"
        imageVariablesPath = imageVariablesPath + imageFilesPath
        imageLabels = imageLabels + [i for x in range(len(imageFilesPath))]

    imageArr = list(zip(imageVariablesPath, imageLabels))
    for i in range(10):
        random.shuffle(imageArr)
    imageVariablesPath, imageLabels = zip(*imageArr)

    imageVariablesPath = np.array(imageVariablesPath)
    imageLabels = np.array(imageLabels)
    skf = StratifiedKFold(n_splits=k)

    return imageVariablesPath, imageLabels, skf.split(imageVariablesPath, imageLabels)


def showROCAndAUC(classNumbers, y_test, y_pred, indexs):
    for i in range(classNumbers):
        yTrue = []
        y_sore = []
        for j in range(len(y_test)):
            if y_test[j] == i:
                yTrue.append(2)
            else:
                yTrue.append(1)
            y_sore.append(y_pred[j][i])
        fpr, tpr, thresholds = roc_curve(yTrue, y_sore, pos_label=2)
        # print(fpr, tpr, thresholds)
        roc_auc = auc(fpr, tpr)
        plt.title(indexs + 'Receiver Operating Characteristic for class ' + str(i))
        plt.plot(fpr, tpr, '#9400D3', label=u'AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid(linestyle='-.')
        plt.grid(True)
        plt.savefig('./results/' + indexs + str(i) + '_ROC.jpg')
        plt.show()


if __name__ == '__main__':

    rootDirPath = "./images\\1\\36.jpg"
    image = cv2.imread(rootDirPath)
    image = cv2.resize(image, (480, 360))
    cv2.imshow('p', image)
    # # 图像 resize
    # image = cv2.resize(image, (224, 224))
    # cv2.imshow('p1', image)
    # cv2.waitKey(0)

    # # 图像灰度变换
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # 图像滤波处理
    # image = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imshow('w1', image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 原始数据集处理例程
    # 文件夹名称之后不加/
    # imageDir_path = "C:/Users/lenovo/Desktop/RobotData/RobotBas"
    # targetDir_path = 'C:/Users/lenovo/Desktop/coco'
    # imagePathString = './images/'
    # labelDir_path = "C:/Users/lenovo/Desktop/RobotData/RobotBasLabels"
    # tranToYoloData(imageDir_path,
    #                targetDir_path,
    #                labelDir_path,
    #                imagePathString)

    # # 文件夹名称之后不加/
    # json_file = 'test.json'
    # ana_txt_save_path = "C:\\Users\\lenovo\\Desktop\\ssd.pytorch-master\\txtfiles"
    # tranJsonToTxt(json_file, ana_txt_save_path)

    # 文件夹名称之后不加/
    # imageDir_path = "C:/Users/lenovo/Desktop/UAVGPData/box3"
    # renameSeqImage(imageDir_path)

    # 文件夹名称之后不加/
    # SourceImagePaths = "C:/Users/lenovo/Desktop/RobotData/RobotBas"
    # SourceLabelPaths = "C:/Users/lenovo/Desktop/RobotData/RobotBasLabels"
    # addImagePaths = "C:/Users/lenovo/Desktop/RobotData/volleyballRed"
    # addLabelPaths = "C:/Users/lenovo/Desktop/RobotData/volleyballRed_label"
    # addImagesAndLabels(SourceImagePaths, SourceLabelPaths, addImagePaths, addLabelPaths)

    # 文件夹名称之后不加/
    # imageDir_path = "C:\\Users\\lenovo\\Desktop\\images\\5"
    # imageResize(imageDir_path)

    # 文件夹名称之后不加/
    # imageDir_path = 'C:/Users/lenovo/Desktop/UAVGPData/pattern1'
    # labelDir_path = 'C:/Users/lenovo/Desktop/UAVGPData/pattern1Label'
    # renameRandomImageAndLabel(imageDir_path, labelDir_path)

    # 文件夹名称之后不加/
    # imageDir_path = "C:/Users/lenovo/Desktop/ssd.pytorch-master/data/basketball/JPEGImages"
    # labelDir_path = 'C:/Users/lenovo/Desktop/ssd.pytorch-master/data/basketball/Annotations'
    # objectName = 'basketball'
    # targetDir_path = 'C:/Users/lenovo/Desktop'
    # tranToVOCData(imageDir_path, targetDir_path, labelDir_path, objectName)

    # imageDir_path = "C:/Users/lenovo/Desktop/ssd.pytorch-master/data/basketball/JPEGImages"
    # labelDir_path = 'C:/Users/lenovo/Desktop/ssd.pytorch-master/data/basketball/Annotations'
    # renameRandomImageAndLabelEnhanced(imageDir_path, labelDir_path)
