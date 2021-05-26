# Author : roczhang
# date :   2021/5/25
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np

X, y = make_classification(n_samples=10000, n_features=2, n_informative=2,
                            n_redundant=0, n_repeated=0, n_classes=3,
                            n_clusters_per_class=1,
                            weights=[0.01, 0.05, 0.94], class_sep=0.8,
                            random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

eec = EasyEnsembleClassifier(random_state=0)
train_data = pd.read_csv('/data/file/classification_data/2012-2019/data_sum/2015/train/train_data.csv', index_col=0)
train_label = pd.read_csv('/data/file/classification_data/2012-2019/data_sum/2015/train/train_label.csv', index_col=0)

test_data = pd.read_csv('/data/file/classification_data/2012-2019/data_sum/2015/train/test_data.csv', index_col=0)
test_label = pd.read_csv('/data/file/classification_data/2012-2019/data_sum/2015/train/test_label.csv', index_col=0)
# 将pandas的DataFrame格式转换成array格式
train_data.values
train_label.values

test_data.values.shape  # (520, 448)
test_label = test_label.values
test_label.reshape(-1)
test_label.shape
eec.fit(train_data.values, train_label.values)

test_pred = eec.predict(test_data.values)
test_pred.shape
balanced_accuracy_score(test_label, test_pred)
