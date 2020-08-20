import random
from pandas import  DataFrame
import numpy as np
import pandas as pd
from pandas import Series
from interval import Interval

datafile = '/Users/kkky/PycharmProjects/shixi/bank-additional-full.csv'#文件所在位置
data = pd.read_csv(datafile)
col=data['education']

len0=len(col)
recol=col.value_counts()#分组

#修改编码
recol_ix=list(recol.index)
col=col.replace(recol_ix,list(range(0,len(recol))))

len1=len(recol)
num=recol.sum()#求总个数
f = lambda x: x/num
recol1=recol.apply(f)#概率化
recol1_ix=list(recol1.index)#提取index
recol2=recol1.cumsum(0)
##先验概率

#生成区间
zoom1=list(range(0,len1))
for i in range(0,len1):
    if i==0:
        zoom1[i]=Interval(0, recol2.iat[i])
    elif i==len1-1:
        zoom1[i]=Interval(recol2.iat[i-1],1)
    else:
        zoom1[i]= Interval(recol2.iat[i-1], recol2.iat[i])

#修改
for j in range(0,len0):
    if np.isnan(col.ix[j,0]):
        #随机数
        m=random.random()
        #判定
        for i in range(0,len1):
            if m in zoom1[i]:
                a=i
                break
        col.ix[j,0]=recol1_ix[a]


#改回编码
col=col.replace(list(range(0,len(recol))),recol_ix)







