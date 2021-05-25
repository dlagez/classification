# Author : roczhang
# date :   2021/5/25
import pandas as pd
import os


def concat_file(dir_root, begin, end):
    data_all = []
    l
    for year in range(begin, end):
        data_name = os.path.join(dir_root+str(year), 'data_bq_drop.csv')
        data = pd.read_csv(data_name, index_col=0)
        data = data.iloc[:, :-1]
        data_all.append(data)
    result = pd.concat(data_all, axis=1)
    return result

concat_file('/data/file/classification_data/2012-2019/data_sum', 2012, 2017)