# author:roczhang
# file:data.py
# time:2021/04/16
import scipy.io as scio

import scipy


train_data = scio.loadmat("/data/file/classification_data/SJ15/trainData.mat")["x_train"]
test_data = scio.loadmat("/data/file/classification_data/SJ15/testData.mat")["x_test"]

train_label = scio.loadmat("/data/file/classification_data/SJ15/trainlabel.mat")["trainlabel"]

print(train_data.shape)  # (392, 194)



train_dir = "/data/file/classification_data/SJ15/trainData.mat"
train_dict = scio.loadmat(train_dir)
train_key = list(train_dict)[-1]
train_data = train_dict[train_key]
train_data[5].shape

# 保存文件
# scipy.io.savemat('daset_sklearn/data_resampled/X_resampled2012.mat', {'X_resampled': X_resampled})
js = scio.loadmat("/data/file/classification_data/pre/bagging/SJ15/label_pred.mat")


