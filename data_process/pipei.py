# Author : roczhang
# date :   2021/5/25
import pandas as pd
import numpy as np

data_pro_read = pd.read_csv('/data/file/classification_data/2012-2019/data_sum/2012/data_pro.csv', index_col=0)

bq = pd.read_excel('/data/file/classification_data/2012-2019/2012/标签1.xls', header=None, index_col=0)
bq.rename(columns={1: 'bq'}, inplace=True)

data_bq2012 = pd.concat([data_pro_read, bq], axis=1)
data_bq2012.to_csv('/data/file/classification_data/2012-2019/data_sum/2012/data_bq.csv')



data_pro_read = pd.read_csv('/data/file/classification_data/2012-2019/data_sum/2013/data_pro.csv', index_col=0)

bq = pd.read_excel('/data/file/classification_data/2012-2019/2013/标签1.xls', header=None, index_col=0)
bq.rename(columns={1: 'bq'}, inplace=True)

data_bq2013 = pd.concat([data_pro_read, bq], axis=1)
data_bq2012.to_csv('/data/file/classification_data/2012-2019/data_sum/2012/data_bq.csv')

data_bq2013[data_bq2013['bq'] == 2].index

# 删除bq列含有2的列
data_bq2013_drop = data_bq2013.drop(index=data_bq2013[data_bq2013['bq'] == 2].index)

(data_bq2013_drop['bq'] == 2).sum()