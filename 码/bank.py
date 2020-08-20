import random
from pandas import  DataFrame
import numpy as np
import pandas as pd
from pandas import Series
from interval import Interval
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#读取文件
datafile = '/Users/kkky/PycharmProjects/shixi/bank-additional-full.csv'#文件所在位置
data = pd.read_csv(datafile)
#print("显示缺失值，缺失则显示为TRUE：\n", data.isna())#是缺失值返回True，否则范围False
#print("---------------------------------\n显示每一列中有多少个缺失值：\n",data.isna().sum())#返回每列包含的缺失值的个数

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,7))

data[data.y =='yes']['marital'].value_counts().plot.pie(autopct='%1.2f%%',shadow=True,ax=ax1)
#,colors = ['lightskyblue','pink','lavender']
ax1.set_ylabel('')
ax1.set_title('NO', fontsize=15)

data[data.y =='no']['marital'].value_counts().plot.pie(autopct='%1.2f%%',shadow=True,ax=ax2)
ax2.set_ylabel('')
ax2.set_title('YES', fontsize=15)



####
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,7))

data[data.y =='yes']['contact'].value_counts().plot.pie(autopct='%1.2f%%',shadow=True,ax=ax1)
#,colors = ['lightskyblue','pink','lavender']
ax1.set_ylabel('')
ax1.set_title('NO', fontsize=15)

data[data.y =='no']['contact'].value_counts().plot.pie(autopct='%1.2f%%',shadow=True,ax=ax2)
ax2.set_ylabel('')
ax2.set_title('YES', fontsize=15)



#先删除缺失多的
job_index=data['job'].isna()
job_index[job_index == True] = 1
job_index[job_index == False] = 0

marital_index=data['marital'].isna()
marital_index[marital_index == True] = 1
marital_index[marital_index == False] = 0

education_index=data['education'].isna()
education_index[education_index == True] = 1
education_index[education_index == False] = 0

index_count = job_index + marital_index + education_index
#print('obs with missing values for all three variables:',index_count[index_count == 3].count())
#print('obs with missing values for two variables:',index_count[index_count == 2].count())
#print('obs with missing values for one variable:',index_count[index_count == 1].count())

#print("显示缺失值，缺失则显示为TRUE：\n", data.isna())#是缺失值返回True，否则范围False
#print("---------------------------------\n显示每一列中有多少个缺失值：\n",data.isna().sum())#返回每列包含的缺失值的个数

##贝叶斯估计补全缺失数据
def bayes_null(col):#col 某一列
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
    return col


data['job']=bayes_null(data['job'])
data['marital']=bayes_null(data['marital'])
data['education']=bayes_null(data['education'])#贝叶斯
data=data.drop(columns=['default'])#直接删除
data['housing']=bayes_null(data['housing'])
data['loan']=bayes_null(data['loan'])
data=data.drop(columns=['duration'])#直接删除
bank_filter1 = data.loc[index_count <= 1,]
data =bank_filter1#删除过多缺失


#print("显示缺失值，缺失则显示为TRUE：\n", data.isna())#是缺失值返回True，否则范围False
#print("---------------------------------\n显示每一列中有多少个缺失值：\n",data.isna().sum())#返回每列包含的缺失值的个数


##缺失数据处理



data_cov = data[['age','job','marital','education','housing','loan']]

data_cov=data.dropna()

#data_age=data['age']/max(data['age'])  #最大98
def standardize(x):
	return (x - np.mean(x))/(np.std(x))

data_age=standardize(data['age'])

dummies_job = pd.get_dummies(data['job'], prefix='job')#独热编码
dummies_mar = pd.get_dummies(data['marital'], prefix='marital')#独热编码
edu=list(set(data['education']))
dummies_edu = data['education'].replace(edu,list([2,3,4,5,1,7,6]))#映射编码
#dummies_edu=dummies_edu/max(dummies_edu)
data_edu=standardize(dummies_edu)
data_hl = data_cov[['housing','loan']]
data_hl_01 = data_hl.replace(list(['yes','no']),list([1,0]))#映射编码
data_hl_01= data_hl_01.astype(int)
data_con=pd.get_dummies(data['contact'], prefix='contact')#独热编码
data_mon=pd.get_dummies(data['month'], prefix='month')#独热编码
data_day=pd.get_dummies(data['day_of_week'], prefix='day')#独热编码
data_cam=standardize(data['campaign'])
data_pdays=standardize(data['pdays'])
data_pre=standardize(data['previous'])
#data_cam=data['campaign']/max(data['campaign'])  #最大56
#data_pdays=data['pdays']/max(data['pdays'])  #最大999
#data_pre=data['previous']/max(data['previous'])  #最大7
data_pou=pd.get_dummies(data['poutcome'], prefix='poutcome')#独热编码
def to1(x):#正负归一化
    Min=min(x)
    Max=max(x)
    x = (x - Min) / (Max - Min);
    return x

data_emp=standardize(data['emp.var.rate'])
data_pri=standardize(data['cons.price.idx'])
data_conf=standardize(data['cons.conf.idx'])
data_eur=standardize(data['euribor3m'])
data_nr=standardize(data['nr.employed'])
data_y=data['y'].replace(list(['yes','no']),list([1,0]))

data_cov_fin=pd.DataFrame(data_age).join(dummies_job).join(dummies_mar).join(dummies_edu).join(data_hl_01).join(data_con)\
    .join(data_mon).join(data_day).join(pd.DataFrame(data_cam)).join(data_pdays).join(data_pre).join(data_pou)\
    .join(pd.DataFrame(data_emp)).join(data_pri).join(pd.DataFrame(data_conf))\
    .join(pd.DataFrame(data_eur)).join(pd.DataFrame(data_nr)).join(data_y)


data_cov_fin=pd.DataFrame(data_age).join(dummies_job).join(dummies_mar).join(dummies_edu).join(data_hl_01).join(data_con)\
    .join(data_mon).join(data_day).join(pd.DataFrame(data_cam)).join(data_pdays).join(data_pre).join(data_pou)\
    .join(pd.DataFrame(data_emp)).join(data_pri).join(pd.DataFrame(data_conf))\
    .join(pd.DataFrame(data_eur)).join(pd.DataFrame(data_nr)).join(data_y)
data_cov_fin.to_csv('/Users/kkky/PycharmProjects/shixi/bank_disnull_2.csv',index=False)
##清洗完毕









