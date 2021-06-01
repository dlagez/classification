# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
# 将
# /data/file/classification_data/SVM数据/T32 
# 数据拼接归一化后在分开作为训练和测试数据


# %%
import os
import scipy
import scipy.io as scio
import numpy as np
from sklearn import preprocessing


# %%
data_root = '/data/file/classification_data/SVM数据/T32'


# %%
X_train = scio.loadmat(os.path.join(data_root, 'traindata.mat'))['x_train']
X_test = scio.loadmat(os.path.join(data_root, 'testdata.mat'))['x_test']


# %%
X_train.shape
X_test.shape


# %%
def X_norm(data_root):
    # 处理数据部分
    sumX = []
    len = 0
    X_train = scio.loadmat(os.path.join(data_root, 'traindata.mat'))['x_train']
    X_test = scio.loadmat(os.path.join(data_root, 'testdata.mat'))['x_test']
    len = X_train.shape[0]
    sumX.append(X_train)
    sumX.append(X_test)
    data_X = np.concatenate(sumX, axis=0)  # 将列表中的数据按0轴合并
    X_norm_all = preprocessing.normalize(data_X, norm='l2', axis=0)  # 训练和测试的样本统一化
    trainX_norm = X_norm_all[:len]  # 取出训练数据
    testX_norm = X_norm_all[len:]  # 取出测试数据

    # 返回训练数据、训练标签、测试数据、测试标签
    return trainX_norm, testX_norm
trainX_norm, testX_norm = X_norm('/data/file/classification_data/SVM数据/T32')

# %%
trainX_norm



# %%

testX_norm
# %%
