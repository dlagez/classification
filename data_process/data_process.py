# Author : roczhang
# date :   2021/5/25
import pandas as pd
import os
import numpy as np
# 将一个文件夹中多个xls文件按列合并，返回合并和的数据，DataFrame
def get_type():
    dict = {}
    for i in range(200):
        dict[i] = np.float
    return dict

a = get_type()

def concat_xls(dir_name):
    dfs = []
    i = 1
    # dtype = get_type()
    for filename in os.listdir(dir_name):
        if os.path.splitext(filename)[1] == '.xls':
            print("filename is ", str(filename))
            full_path = os.path.join(dir_name, filename)
            df = pd.read_excel(full_path, header=None, index_col=0, na_values=["NA"])
            print(df.dtypes)
            dfs.append(df)
            i = i + 1
            print('now: '+ str(i))
            print('------------------------------------------------')
    print('sum: '+ str(i))
    result = pd.concat(dfs, axis=1)
    return result

def print_name(dir_name):
    i = 0
    for filename in os.listdir(dir_name):
        if os.path.splitext(filename)[1] == '.xls':
            i = i + 1
            print(filename)
    print(i)
print_name('/data/file/classification_data/2012-2019/data_A/2012')


data_2012 = concat_xls('/data/file/classification_data/2012-2019/data_A/2012')

data_2012.columns = [i for i in range(data_2012.shape[1])]
data1 = data_2012.apply(pd.to_numeric, errors='ignore')
d = [data_2012.dtypes]
d1 = [data1.dtypes]

# 先把拼接起来的值存储起来
for year in range(2012, 2018):
    # print(year)
    data_root = '/data/file/classification_data/2012-2019/data_A/'
    data_name = os.path.join(data_root, str(year))
    # print(data_name)
    data = concat_xls(data_name)
    # print(data)
    data.columns = [i for i in range(data.shape[1])]
    data.to_csv(os.path.join('/data/file/classification_data/2012-2019/data_sum/'+str(year), 'data_'+str(year)+'.csv'))


# 处理拼接起来的数据的空值
for year in range(2012, 2018):
    data_root = '/data/file/classification_data/2012-2019/data_sum/'
    data_name = os.path.join(data_root+str(year), 'data_'+str(year)+'.csv')
    print(data_name)
    data = pd.read_csv(data_name, index_col=0)
    data_pro = data.fillna(data.mean(), axis=0)
    print(data_pro)
    data_pro.to_csv(os.path.join(data_root+str(year), 'data_pro.csv'))
#
#
# 将标签值添加到data_pro的末尾
for year in range(2012, 2018):
    data_root = '/data/file/classification_data/2012-2019/data_sum/'
    data_root_bq = '/data/file/classification_data/2012-2019/'
    data_name = os.path.join(data_root+str(year), 'data_pro.csv')
    data_pro_read = pd.read_csv(data_name, index_col=0)
    print(data_pro_read)
    bq = pd.read_excel(os.path.join(data_root_bq+str(year), '标签1.xls'), header=None, index_col=0)
    bq.rename(columns={1: 'bq'}, inplace=True)
    print(bq)
    data_bq = pd.concat([data_pro_read, bq], axis=1)
    print(data_bq)
    data_bq.to_csv(os.path.join(data_root+str(year), 'data_bq.csv'))

# 删除含有标签2的列
for year in range(2012, 2018):
    data_root = '/data/file/classification_data/2012-2019/data_sum/'
    data_name = os.path.join(data_root+str(year), 'data_bq.csv')
    data = pd.read_csv(data_name,  index_col=0)
    print(data)
    data_bq_drop = data.drop(index=data[data['bq'] == 2].index)
    # data_bq2013_drop = data_bq2013.drop(index=data_bq2013[data_bq2013['bq'] == 2].index)
    print(data_bq_drop)
    data_bq_drop.to_csv(os.path.join(data_root+str(year), 'data_bq_drop.csv'))

# 将数据分成四份，训练和测试标签
for year in range(2012, 2018):
    data_root = '/data/file/classification_data/2012-2019/data_sum/'
    data_name = os.path.join(data_root+str(year), 'data_bq_drop.csv')
    data = pd.read_csv(data_name, index_col=0)
    print(data)
    train_size =3 * int(data.shape[0] / 4)
    test_size = int(data.shape[0] - train_size)
    train_data_all = data.iloc[:train_size]
    print(train_data_all)
    test_data_all = data.iloc[train_size:]
    print(test_data_all)
    train_data = train_data_all.iloc[:, :-1]
    print(train_data)
    train_label = train_data_all.iloc[:, -1]
    # train_label.columns = ['bq']
    print(train_label)
    test_data = test_data_all.iloc[:, :-1]
    test_label = test_data_all.iloc[:, -1]
    print(test_data)
    print(test_label)
    train_data.to_csv(os.path.join(data_root+str(year), 'train/train_data.csv'))
    train_label.to_csv(os.path.join(data_root+str(year), 'train/train_label.csv'))
    test_data.to_csv(os.path.join(data_root+str(year), 'train/test_data.csv'))
    test_label.to_csv(os.path.join(data_root+str(year), 'train/test_label.csv'))


