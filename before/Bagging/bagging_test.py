# author: roczhang
# file: bagging_test.py
# time: 2021/04/16
import scipy.io as scio
import scipy
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
bc = BaggingClassifier(n_estimators=1)
# n_estimators=1
# 0.8285714285714285
# 0.8309859154929577
# 0.8398644936824757
# 0.8537330932540513
year = 15
while (year < 19):
    year_str = str(year)
    year += 1
    train_data = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/trainData.mat")["x_train"]
    train_label = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/trainlabel.mat")["trainlabel"].ravel()

    test_data = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/testData.mat")["x_test"]
    test_label = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/testlabel.mat")["testlabel"].ravel()


    bc.fit(train_data, train_label)
    label_pred = bc.predict(test_data).reshape(-1, 1)
    # print(label_pred)
    print(balanced_accuracy_score(label_pred, test_label))
    scipy.io.savemat("/data/file/classification_data/pre/bagging2/SJ"+year_str+"/label_pred.mat", {'label_pred': label_pred})
