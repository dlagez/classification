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
    X_test = scio.loadmat(os.path.join(data_root+'new_t32/', 'T32TEdata.mat'))['testData']  # 增加测试的X
    sumX.append(X_test)  # 将测试数据放进数据列表
    data_X = np.concatenate(sumX, axis=0)  # 将列表中的数据按0轴合并
    X_norm_all = preprocessing.normalize(data_X, norm='l2', axis=0)  # 训练和测试的样本统一化
    trainX_norm = X_norm_all[:length]  # 取出训练数据
    testX_norm = X_norm_all[length:]  # 取出测试数据

    # 处理标签部分
    testy_norm = scio.loadmat(os.path.join(data_root + 'new_t32/', 'testlabel.mat'))['testlabel']  # 读取测试的y标签
    # sumy.append(y_test)
    trainy_norm = np.concatenate(sumy, axis=0)
    # 返回训练数据、训练标签、测试数据、测试标签
    return trainX_norm, trainy_norm, testX_norm, testy_norm

train_sumX_2016_norm, train_sumy_2016_label, test_sumX_2016_norm, test_sumy_2016_label = get_sumX('/data/file/classification_data/years/', 2012, 2017)
data_root = '/data/file/classification_data/years/sum_2016/'
# scipy.io.savemat(os.path.join(data_root, 'X_norm_2016.mat'), {'X': sumX_2016})

# 存储归一化的数据
scipy.io.savemat(os.path.join(data_root, 'train_sumX_2016_norm.mat'), {'X': train_sumX_2016_norm})
scipy.io.savemat(os.path.join(data_root, 'train_sumy_2016_label.mat'), {'y': train_sumy_2016_label})
scipy.io.savemat(os.path.join(data_root, 'test_sumX_2016_norm.mat'), {'X': test_sumX_2016_norm})
scipy.io.savemat(os.path.join(data_root, 'test_sumy_2016_label.mat'), {'y': test_sumy_2016_label})





# 将**年到**年的标签合并
def get_sumy(data_root, begin, end):
    # sumX = []
    sumy = []
    for year in range(begin, end):
        # data_root = '/data/file/classification_data/years/'
        # X = scio.loadmat(os.path.join(data_root+str(year), 'X_drop2.mat'))['X']

        y = scio.loadmat(os.path.join(data_root+str(year), 'y_drop2.mat'))['y']
        # X_norm = preprocessing.normalize(y, norm='l2')
        y = y.reshape(-1, 1)
        # sumX.append(X_norm)
        sumy.append(y)

    data_y = np.concatenate(sumy, axis=0)
    # data_y = np.concatenate(sumy, axis=0)
    return data_y
# sumy_2016 = get_sumy('/data/file/classification_data/years/', 2012, 2017)
# sumy_2016.shape
# data_root = '/data/file/classification_data/years/sum_2016/'
# scipy.io.savemat(os.path.join(data_root, 'X_norm_2016.mat'), {'X': sumX_2016})


# 读取合并了的数据

data_root = '/data/file/classification_data/years/'
X_norm = scio.loadmat(os.path.join(data_root, 'sum_2016/X_norm_2016.mat'))
