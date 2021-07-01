# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 17:58
# @Author  : Lightning
# @FileName: pptShow.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 , 18, 19, 20]
    yFullyCon = [1.893794, 1.528141, 1.362892, 1.228983, 1.191438, 1.146951, 1.036419, 1.084144, 0.943037, 0.944890,  0.905382,  0.819440, 0.823587, 0.840413, 0.755497, 0.675702, 0.789222 , 0.668536, 0.561427, 0.609067]
    yConvolution = [1.571125, 1.173743, 0.857677, 0.609134, 0.449001, 0.269851, 0.145616, 0.093596, 0.032709, 0.019704, 0.013881, 0.005411, 0.003055, 0.002080, 0.001676, 0.001328, 0.001097, 0.000905, 0.000776, 0.000673]
    yDeep =  [1.739400, 1.604326, 1.575881, 1.409306, 1.287092, 1.059466, 0.930772, 0.756866, 0.674813, 0.576891, 0.500815, 0.424357, 0.525412, 0.347786, 0.328252, 0.251411, 0.235293, 0.206719, 0.213842, 0.138904]
    yRes =[1.472511, 1.068173, 0.929144, 0.766215, 0.663385, 0.550719, 0.495614, 0.413579, 0.372729, 0.329839, 0.298343, 0.276178, 0.229116, 0.211435, 0.190334, 0.163829, 0.147253 , 0.137595, 0.130187, 0.138889]
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字`
    pl.plot(x, yFullyCon, 'g-', label=u'fully connected network')
    pl.plot(x, yConvolution, 'r-', label=u'convolution network')
    pl.plot(x, yDeep, 'b-', label=u'deep convolution network') #brown
    pl.plot(x, yRes, 'brown',  label=u'res network')
    pl.legend()
    pl.xlabel(u'epochs')
    pl.ylabel(u'loss')
    plt.title('Compare loss for different networks in training')
    plt.show()
    # import numpy as np
    # import matplotlib.pyplot as plt
    # #
    # classes = ['1', '2', '3', '4', '5']
    # confusion_matrix = np.array(
    #     [(62, 5, 1, 1, 3),
    #      (5, 63, 3, 2, 2),
    #      (4, 2, 33, 7, 1),
    #      (1, 4, 6, 48, 0),
    #      (1, 4, 2, 1, 62)], dtype=np.float64)
    #
    # plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    # plt.title('confusion_matrix')
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=-45)
    # plt.yticks(tick_marks, classes)
    #
    # thresh = confusion_matrix.max() / 2.
    # # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # # ij配对，遍历矩阵迭代器
    # iters = np.reshape([[[i, j] for j in range(5)] for i in range(5)], (confusion_matrix.size, 2))
    # for i, j in iters:
    #     plt.text(j, i, format(confusion_matrix[i, j]))  # 显示对应的数字
    #
    # plt.ylabel('Real label')
    # plt.xlabel('Prediction')
    # plt.tight_layout()
    # plt.show()

    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # yFullyCon = [1.893794, 1.528141, 1.362892, 1.228983, 1.191438, 1.146951, 1.036419, 1.084144, 0.943037, 0.944890,  0.905382,  0.819440, 0.823587, 0.840413, 0.755497, 0.675702, 0.789222 , 0.668536, 0.561427, 0.609067]
    # yConvolution = [1.571125, 1.173743, 0.857677, 0.609134, 0.449001, 0.269851, 0.145616, 0.093596, 0.032709, 0.019704, 0.013881, 0.005411, 0.003055, 0.002080, 0.001676, 0.001328, 0.001097, 0.000905, 0.000776, 0.000673]
    # yDeep =  [1.739400, 1.604326, 1.575881, 1.409306, 1.287092, 1.059466, 0.930772, 0.756866, 0.674813, 0.576891, 0.500815, 0.424357, 0.525412, 0.347786, 0.328252, 0.251411, 0.235293, 0.206719, 0.213842, 0.138904]
    # yRes = [1.445648, 1.044975, 0.887185,0.711835, 0.623158, 0.583462, 0.494267, 0.428004, 0.400727, 0.371311, 0.300863, 0.246384, 0.245234, 0.216079, 0.221988, 0.188831, 0.149712 , 0.153154, 0.133003, 0.122556]
    # fig = plt.figure(figsize=(7, 5))
    # ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字`
    # pl.plot(x, yFullyCon, 'g-', label=u'fully connected network')
    # pl.plot(x, yConvolution, 'r-', label=u'convolution network')
    # pl.plot(x, yDeep, 'b-', label=u'deep convolution network') #brown
    # pl.plot(x, yRes, 'brown',  label=u'res network')
    # pl.legend()
    # pl.xlabel(u'epochs')
    # pl.ylabel(u'loss')
    # plt.title('Compare loss for different networks in training')
    # plt.show()