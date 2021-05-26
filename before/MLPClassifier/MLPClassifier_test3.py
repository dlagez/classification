# author: roczhang
# file: MLPClassifier_test3.py
# time: 2021/04/18

import scipy.io as scio
import scipy
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
clf = MLPClassifier(hidden_layer_sizes=5, random_state=5, max_iter=500)
# 0.8317307692307692 SJ15
# 0.8934729064039408 SJ16
# 0.8386243386243386 SJ17
# 0.7947515212981744 SJ18
year = 15


while (year < 19):
    year_str = str(year)
    year += 1
    train_data = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/trainData.mat")["x_train"]
    train_label = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/trainlabel.mat")["trainlabel"].ravel()

    test_data = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/testData.mat")["x_test"]
    test_label = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/testlabel.mat")["testlabel"].ravel()

    clf.fit(train_data, train_label)
    label_pred = clf.predict(test_data).reshape(-1, 1)
    # print(label_pred)
    print(balanced_accuracy_score(label_pred, test_label))
    scipy.io.savemat("/data/file/classification_data/pre/pr_MLP3/SJ" + year_str + "/label_pred.mat", {'label_pred': label_pred})
