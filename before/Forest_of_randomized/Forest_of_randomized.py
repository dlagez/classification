# author: roczhang
# file: Forest_of_randomized.py
# time: 2021/04/16
import scipy.io as scio
from imblearn.ensemble import BalancedRandomForestClassifier
import scipy
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
year = 15

while (year < 19):
    year_str = str(year)
    year += 1
    train_data = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/trainData.mat")["x_train"]
    train_label = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/trainlabel.mat")["trainlabel"].ravel()

    test_data = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/testData.mat")["x_test"]
    test_label = scio.loadmat("/data/file/classification_data/SJ"+year_str+"/testlabel.mat")["testlabel"].ravel()

    brf.fit(train_data, train_label)
    label_pred = brf.predict(test_data).reshape(-1, 1)
    print(label_pred)
    scipy.io.savemat("/data/file/classification_data/pre/forest_of_random/SJ" + year_str + "/label_pred.mat", {'label_pred': label_pred})
