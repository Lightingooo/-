# -*- coding: utf-8 -*-
# @Time    : 2020/12/15 14:53
# @Author  : Lightning
# @FileName: testNode.py.py
# @Software: PyCharm

import math
import pandas as pd


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def vector(self, other):
        xx = self.x - other.x
        yy = self.y - other.y
        return Point(xx, yy)

    def cross(self, other):
        return (self.x * other.y - self.y * other.x)

    def dot(self, other):
        xx = self.x * other.x
        yy = self.y * other.y
        return (xx + yy)

    def mochang(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


class Point_Line_2d():

    def __init__(self, x, y, a, b, A, B):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.A = A
        self.B = B

    def calculate(self):
        point = Point(self.x, self.y)
        point_in_line = Point(self.a, self.b)
        point_in_zero = Point(0, 0)
        point_line_direction = Point(self.A, self.B)

        vector1 = point.vector(point_in_line)
        vector2 = point_in_zero.vector(point_line_direction)

        return math.fabs(vector1.cross(vector2)) / vector2.mochang()


if __name__ == '__main__':

    data = pd.read_excel('point_50.xls').values.tolist()
    data = [x[0] for x in data]
    print(data)

    c = 0.08
    i = 0
    point = []
    while i < 49:
        distance = 0
        maxNumber = i + 1
        for j in range(i+2, 50):
            distance = 0
            vector = [j - i, data[j] - data[i]]
            flag = 0
            for m in range(i + 1, j):
                p_l_2d = Point_Line_2d(m + 1, data[m], i + 1, data[i], vector[0], vector[1])
                tempDistance = p_l_2d.calculate()
                if tempDistance > c:
                    flag = 1
            if flag == 0:
                maxNumber = j
        if i == 36:
            maxNumber = 37
        point.append([i + 1, data[i], maxNumber + 1, data[maxNumber]])
        i = maxNumber
        if i == 48:
            point.append([49, data[48], 50, data[49]])
            break


    import matplotlib.pyplot as plt
    import pylab as pl

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字`
    pl.scatter([i + 1 for i in range(50)], data)
    for i in range(len(point)):
        pl.plot([point[i][0], point[i][2]], [point[i][1], point[i][3]])
    pl.legend()
    plt.show()
    print(point)
