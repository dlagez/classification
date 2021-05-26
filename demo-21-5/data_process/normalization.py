# Author : roczhang
# date :   2021/5/26

import os
import scipy
import scipy.io as scio
import numpy as np
from sklearn import preprocessing


def get_sumX(data_root, begin, end):
    sumX = []
    # sumy = []
    for year in range(begin, end):
        # data_root = '/data/file/classification_data/years/'
        X = scio.loadmat(os.path.join(data_root+str(year), 'X_drop2.mat'))['X']
        # y = scio.loadmat(os.path.join(data_root+str(year), 'y_drop2.mat'))['y']
        X_norm = preprocessing.normalize(X, norm='l2')
        sumX.append(X_norm)
        # sumy.append(y)
    data_X = np.concatenate(sumX, axis=0)
    # data_y = np.concatenate(sumy, axis=0)
    return data_X

# sumX_2016 = get_sumX('/data/file/classification_data/years/', 2012, 2017)
# data_root = '/data/file/classification_data/years/sum_2016/'
# scipy.io.savemat(os.path.join(data_root, 'X_norm_2016.mat'), {'X': sumX_2016})


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
sumy_2016 = get_sumy('/data/file/classification_data/years/', 2012, 2017)
sumy_2016.shape
