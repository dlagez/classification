# Author : roczhang
# date :   2021/5/24
import scipy.io as scio
import scipy
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=1, max_iter=300)
# svm 数据训练和预测，

# 加载mat数据
train_data = scio.loadmat("/data/file/classification_data/SVM数据/T21/traindata.mat")["x_train"]
train_data.shape  # (154, 448)
train_label = scio.loadmat("/data/file/classification_data/SVM数据/T21/trainlabel.mat")["trainlabel"].ravel()

test_data = scio.loadmat("/data/file/classification_data/SVM数据/T21/testdata.mat")["x_test"]
test_label = scio.loadmat("/data/file/classification_data/SVM数据/T21/testlabel.mat")["testlabel"].ravel()

# 数据的训练
mlp.fit(train_data, train_label)
# 数据预测
label_pred = mlp.predict(test_data).reshape(-1, 1)
print(label_pred)
# 每个类获得的平均召回率
print(balanced_accuracy_score(label_pred, test_label))  # 0.8809523809523809
# 测试数据和标签上的平均准确度
print(mlp.score(test_data, test_label))  # 0.84375
# scipy.io.savemat("/data/file/classification_data/SVM数据/pred/T21/lable_pred.mat", {'label_pred': label_pred})

# roc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
pos_prob = mlp.predict_proba(test_data)[:, -1]
pos_prob.shape


def get_roc(pos_prob, y_true):
    pos = y_true[y_true == 1]  # 145
    neg = y_true[y_true == 0]  # 855
    # [::-1]逆序切片
    threshold = np.sort(pos_prob)[::-1]  # 按预测为正的概率大小逆序排列
    y = y_true[pos_prob.argsort()[::-1]]

    tpr_all = [0];
    fpr_all = [0]
    tpr = 0;
    fpr = 0
    x_step = 1 / float(len(neg))  # 0.0011695906432748538
    y_step = 1 / float(len(pos))  # 0.006896551724137931
    y_sum = 0
    for i in range(len(threshold)):  # 1000
        if y[i] == 1:
            tpr += y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr += x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum += tpr  # fpr加一个样本，y_sum累计。计算面积
    return tpr_all, fpr_all, y_sum * x_step  # 获得总体TPR，FPR和相应的AUC

tpr, fpr, auc = get_roc(pos_prob, test_label)
auc  # 0.8984375roc曲线围成的面积， AUC反映的是分类器对样本的排序能力

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label="Logistic Regression (AUC: {:.3f})".format(auc),linewidth=2)
plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.title("ROC Curve", fontsize=16)
plt.legend(loc="lower right", fontsize=16)