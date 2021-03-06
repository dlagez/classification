# Author : roczhang
# date :   2021/5/26

import os
import scipy
import scipy.io as scio
import numpy as np
from sklearn import preprocessing

# 将**年到**年的数据合并
# 竖着归一化
def get_sumX(data_root, begin, end):
    sumX = []
    sumy = []
    length = 0
    # 读取数据和标签
    for year in range(begin, end):
        # data_root = '/data/file/classification_data/years/'
        # 增加每年的X
        X = scio.loadmat(os.path.join(data_root+str(year), 'X_drop2.mat'))['X']
        length = length + X.shape[0]
        # 增加每年的y
        sumX.append(X)
        y = scio.loadmat(os.path.join(data_root+str(year), 'y_drop2.mat'))['y'].reshape(-1, 1)
        # X_norm = preprocessing.normalize(X, norm='l2')
        sumy.append(y)

    # 处理数据部分
    X_test = scio.loadmat(os.path.join(data_root+'T21/', 'T21testData.mat'))['testData']  # 增加测试的X
    sumX.append(X_test)  # 将测试数据放进数据列表
    data_X = np.concatenate(sumX, axis=0)  # 将列表中的数据按0轴合并
    X_norm_all = preprocessing.normalize(data_X, norm='l2', axis=0)  # 训练和测试的样本统一化
    trainX_norm = X_norm_all[:length]  # 取出训练数据
    testX_norm = X_norm_all[length:]  # 取出测试数据

    # 处理标签部分
    testy_norm = scio.loadmat(os.path.join(data_root + 'T21/', 'T21testlabel.mat'))['testlabel']  # 读取测试的y标签
    # sumy.append(y_test)
    trainy_norm = np.concatenate(sumy, axis=0)
    # 返回训练数据、训练标签、测试数据、测试标签
    return trainX_norm, trainy_norm, testX_norm, testy_norm


data_root = '/data/file/classification_data/tmp/years/'
trainX_norm, trainy_norm, testX_norm, testy_norm = get_sumX(data_root, 2013, 2017)

trainX_norm.shape
trainy_norm.shape
testX_norm.shape
testy_norm.shape
save_root = "/data/file/classification_data/2021.6.16T21/data"
# 存储归一化的数据
scipy.io.savemat(os.path.join(data_root, 'trainX_norm.mat'), {'X': trainX_norm})
scipy.io.savemat(os.path.join(data_root, 'trainy_norm.mat'), {'y': trainy_norm})
scipy.io.savemat(os.path.join(data_root, 'testX_norm.mat'), {'X': testX_norm})
scipy.io.savemat(os.path.join(data_root, 'testy_norm.mat'), {'y': testy_norm})

