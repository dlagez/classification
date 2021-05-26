# author: roczhang
# file: MLPClassifier_test.py
# time: 2021/04/17

import scipy.io as scio
import scipy
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

clf = MLPClassifier(hidden_layer_sizes=10, random_state=1, max_iter=300)
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
    print(label_pred)
    scipy.io.savemat("/data/file/classification_data/pre/MLP/SJ" + year_str + "/label_pred.mat", {'label_pred': label_pred})

