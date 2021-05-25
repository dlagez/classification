# Author : roczhang
# date :   2021/5/24
import pandas as pd
import scipy.io as scio
import scipy
from pandas import Series, DataFrame
import os


def concat_xls(dir_name):
    dfs = []
    for filename in os.listdir(dir_name):
        if os.path.splitext(filename)[1] == '.xls':
            full_path = os.path.join(dir_name, filename)
            df = pd.read_excel(full_path, header=None, index_col=0)
            dfs.append(df)
    result = pd.concat(dfs, axis=1)
    return result


data_2012 = concat_xls('/data/file/classification_data/2012-2019/data_A/2012')
# 因为pandas内部调用了xlwt模块，而图示可知，该模块最大列只支持255列，所以在保存数据时，一旦超过这个值就会报错，
# 博主建议改用其他模块进行数据保存，比如xlsxwriter模块。
data_2012.to_csv('/data/file/classification_data/2012-2019/data_sum/data_2012.csv')
# data_2012.isnull()

# 这里不能直接对data_2012操作
# data_pro = data_2012.fillna(data_2012.mean())
# data_pro.shape
data_2012.shape
# scipy.io.savemat("/data/file/classification_data/2012-2019/data_sum/data_2012.mat", {'data_2012': data_2012})

# os.listdir('/data/file/classification_data/2012-2019/data_A/2012')


data_2012 = pd.read_csv('/data/file/classification_data/2012-2019/data_sum/2012/data_2012.csv', index_col=0)
data_pro = data_2012.fillna(data_2012.mean(), axis=0)



data_pro.to_csv('/data/file/classification_data/2012-2019/data_sum/2012/data_pro.csv')
data_pro_read = pd.read_csv('/data/file/classification_data/2012-2019/data_sum/2012/data_pro.csv', index_col=0)
# for column in list(data_2012.columns[data_2012.isnull().sum() > 0]):
#     mean_val = data_2012[column].mean()
#     data_2012.fillna(mean_val, inplace=True)
# data_2012

# 将2012-2019年的数据A1-A15汇总
import os
for year in range(2012, 2020):
    # print(year)
    data_root = '/data/file/classification_data/2012-2019/data_A/'
    data_name = os.path.join(data_root, str(year))
    # print(data_name)
    data = concat_xls(data_name)
    print(data)






data_2012.mean()
data_pro.mean()
# 填充NaN数据
from numpy import nan as NA
import numpy as np
df = pd.DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA
df.iloc[4:, 2] = NA
df
# 这里默认的是axis=0，就是用一列的均值填充数据
df.fillna(df.mean())
df.fillna(df.mean(), axis=0)
df.isnull().sum()







# 将2012-2019数据汇总，并使用xls文件保存
for i in range(2012, 2020):
    dir_root = '/data/file/classification_data/2012-2019/data_A/'
    dir_name = os.path.join(dir_root, str(i))
    print(dir_name)
    sum_xls = concat_xls(dir_name)
    # scipy.io.savemat("/data/file/classification_data/2012-2019/data_sum/data_"+str(i)+".mat", {'data_'+str(i): mat})
    sum_xls.to_csv("/data/file/classification_data/2012-2019/data_sum/data_"+str(i)+".xls")
# 查看保存的文件
# train_data = scio.loadmat("/data/file/classification_data/2012-2019/data_sum/data_2012.mat")

# # load prediction data and save it as .mat file
# import scipy
# def SaveMat(sorted_data):
#     # transfer the data type
#     sorted_data['pre'] = sorted_data['pre'].astype(int)
#     # save the prediction data and remember to resave it in matlab for memory saving
#     b_dict = {col_name : sorted_data[col_name].values for col_name in sorted_data.columns.values}
#     scipy.io.savemat('nfac-pre.mat', {'struct':b_dict})
