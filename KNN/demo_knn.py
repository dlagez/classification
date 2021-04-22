# author: roczhang
# file: MLPClassifier_test3.py
# time: 2021/04/18
# SJ160单独训练
import scipy.io as scio
import scipy
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
# 0.8317307692307692 SJ15
# 0.8934729064039408 SJ16
# 0.8386243386243386 SJ17
# 0.7947515212981744 SJ18
year = 160



year_str = str(year)
train_data = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/trainData.mat")["x_train"]
train_label = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/trainLabel.mat")["trainLabel"].ravel()

test_data = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/testData.mat")["x_test"]
test_label = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/testLabel.mat")["testLabel"].ravel()

clf.fit(train_data, train_label)
label_pred = clf.predict(test_data).reshape(-1, 1)
# print(label_pred)
print(balanced_accuracy_score(label_pred, test_label))  # 0.8287037037037037
scipy.io.savemat("/data/file/classification_data/pre/knn2/SJ" + year_str + "/label_pred.mat", {'label_pred': label_pred})
