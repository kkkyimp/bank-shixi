
import random
from pandas import  DataFrame
import numpy as np
import pandas as pd
from pandas import Series
from interval import Interval


#热力图分析相关性 （几乎无相关性）
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
datafile = '/Users/kkky/PycharmProjects/shixi/bank-additional-full.csv'#文件所在位置
data = pd.read_csv(datafile)#如果是csv文件则用read_csv

data_cov = data[['age','job','marital','education','housing','loan']]
data_cov=data_cov.dropna()

dummies_job = pd.get_dummies(data_cov['job'], prefix='job')
dummies_mar = pd.get_dummies(data_cov['marital'], prefix='marital')#独热编码
edu=list(set(data_cov['education']))
dummies_edu = data_cov['education'].replace(edu,list([2,3,4,5,1,7,6]))#映射编码

data_hl = data_cov[['housing','loan']]
data_hl_01 = data_hl.replace(list(['yes','no']),list([1,0]))
data_hl_01= data_hl_01.astype(int)
data_cov_fin=data_cov[['age']].join(dummies_job).join(dummies_mar).join(dummies_edu).join(data_hl_01)#组合

def test(df):#热力图
    dfData = df.corr()
    plt.subplots(figsize=(5, 5)) # 设置画面大小
    sns.heatmap(dfData, annot=False, vmax=1, square=True, cmap="Blues")
    plt.savefig('./BluesStateRelation.png',bbox_inches='tight')
    plt.show()
    plt.tight_layout()
test(data_cov_fin)

