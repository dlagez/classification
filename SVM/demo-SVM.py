# Author : roczhang
# date :   2021/5/24

import scipy.io as scio
import scipy
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import NuSVC
svm = NuSVC(kernel='rbf', )
# svm 数据训练和预测，


train_data = scio.loadmat("/data/file/classification_data/SVM数据/T21/traindata.mat")["x_train"]
train_data.shape  # (154, 448)
train_label = scio.loadmat("/data/file/classification_data/SVM数据/T21/trainlabel.mat")["trainlabel"].ravel()

test_data = scio.loadmat("/data/file/classification_data/SVM数据/T21/testdata.mat")["x_test"]
test_label = scio.loadmat("/data/file/classification_data/SVM数据/T21/testlabel.mat")["testlabel"].ravel()

svm.fit(train_data, train_label)
label_pred = svm.predict(test_data).reshape(-1, 1)
print(label_pred)
# 每个类获得的平均召回率
print(balanced_accuracy_score(label_pred, test_label))  # 0.9
# 测试数据和标签上的平均准确度
print(svm.score(test_data, test_label))  # 0.875
# scipy.io.savemat("/data/file/classification_data/SVM数据/pred/T21/lable_pred.mat", {'label_pred': label_pred})

# roc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold