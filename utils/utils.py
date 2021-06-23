
import scipy
import scipy.io as scio
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
clf = make_pipeline(StandardScaler(), NuSVC(nu=0.01))


# t32 数据采样并保存

# 读取训练和测试数据
data_root = '/data/file/classification_data/T32/T32data/'
T32TrainData = scio.loadmat(os.path.join(data_root, 'T32TrainData.mat'))['X']
T32TestData = scio.loadmat(os.path.join(data_root, 'T32TestData.mat'))['X']

T32TrainLabel = scio.loadmat(os.path.join(data_root, 'T32TrainLabel.mat'))['y']
T32TestLabel = scio.loadmat(os.path.join(data_root, 'T32TestLabel.mat'))['y']

list = []

#   ------------------上采样------------------
from imblearn.over_sampling import RandomOverSampler 
ros = RandomOverSampler(random_state=42)
list.append(ros)

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
list.append(sm)

from imblearn.over_sampling import BorderlineSMOTE 
blsm = BorderlineSMOTE(random_state=42)
list.append(blsm)

from imblearn.over_sampling import ADASYN 
ada = ADASYN(random_state=42)
list.append(ada)

# -----------------Combination of over- and under-sampling------------
from imblearn.combine import SMOTEENN 
sme = SMOTEENN(random_state=42)
list.append(sme)

from imblearn.combine import SMOTETomek 
smt = SMOTETomek(random_state=42)
list.append(smt)

# -----------------ubder sampling---------------------------
from imblearn.under_sampling import NearMiss 
nm = NearMiss()
list.append(nm)

from imblearn.under_sampling import TomekLinks 
tl = TomekLinks()
# list.append(tl)

from imblearn.under_sampling import RandomUnderSampler 
rus = RandomUnderSampler(random_state=42)
list.append(rus)


for method in list:
    print(str(method))

save_data_root = '/data/file/classification_data/T32/samplings/'
for i, x in enumerate(list) :
    print('-----------------开始----------------')
    current_dir = save_data_root + str(list[i])
    print(current_dir)
    print(str(list[i]))
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    data_sampling, label_sampling = list[i].fit_resample(T32TrainData, T32TrainLabel)
    print(data_sampling.shape,  label_sampling.shape)
    clf.fit(data_sampling, label_sampling)
    clf.score(T32TestData, T32TestLabel)

    # 保存mat文件
    scipy.io.savemat(current_dir+"/data_sampling.mat", {'X': data_sampling})
    scipy.io.savemat(current_dir+"/label_sampling.mat", {'y': label_sampling})
    
    print('--------------------------------------------------------')







