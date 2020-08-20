import random
from pandas import  DataFrame
import numpy as np
import pandas as pd
from pandas import Series
from interval import Interval
import csv
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import fbeta_score
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.dummy import DummyClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
#coding:utf-8
import matplotlib
#matplotlib.use("Agg")
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#import adaboost
import warnings
from sklearn.ensemble import AdaBoostClassifier
warnings.filterwarnings('ignore')



#定义fdeta值
f3_score = make_scorer(fbeta_score, beta=3)  # 添加参数
f2_score = make_scorer(fbeta_score, beta=2)  # 添加参数
f1_5_score = make_scorer(fbeta_score, beta=1.5)  # 添加参数
f1_2_score = make_scorer(fbeta_score, beta=1.2)  # 添加参数
f1_1_score = make_scorer(fbeta_score, beta=1.1)  # 添加参数
###########################
#全部应用fdeta值
###########################


datafile = '/Users/kkky/PycharmProjects/shixi/bank_disnull.csv'#文件所在位置
#datafile = '/Users/kkky/PycharmProjects/shixi/bank_disnull_2.csv'#文件所在位置
data = pd.read_csv(datafile)#如果是csv文件则用read_csv
bank_final= data.drop(columns=['y'])
#bank_final.shape





k_fold = KFold(n_splits=10, shuffle=True)
##过采样
#自选
'''
X_resampled, y_resampled = SMOTE (kind='svm',ratio={1:12000}).fit_sample (bank_final,data['y'])


#平均
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(bank_final, data['y'])
sorted(Counter(y_resampled).items())


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.10, random_state = 111)
k_fold = KFold(n_splits=10, shuffle=True)
#X_train.head()
'''
'''
X_train, X_test, y_train, y_test = train_test_split(bank_final,data['y'], test_size = 0.20, random_state = 111)
X_train, y_train = SMOTE (kind='svm',ratio={1:10000}).fit_sample (X_train, y_train)
#X_train.head()
'''


smote_enn = SMOTEENN(random_state=0,sampling_strategy=0.30)
X_train, X_test, y_train, y_test = train_test_split(bank_final,data['y'], test_size = 0.20, random_state = 111)
X_train, y_train = smote_enn.fit_sample(X_train, y_train)
X_train=pd.DataFrame(X_train,columns=X_test.columns)

####################### PlotKS ##########################
def PlotKS(preds, labels, n, asc):

    # preds is score: asc=1
    # preds is prob: asc=0

    pred = preds  # 预测值
    bad = labels  # 取1为bad, 0为good
    ksds = DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad

    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0*ksds1.good.cumsum()/sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0*ksds1.bad.cumsum()/sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0*ksds2.good.cumsum()/sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0*ksds2.bad.cumsum()/sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2'])/2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2'])/2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0*ksds['tile0']/len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0/n))
    qe.append(1)
    qe = qe[1:]

    ks_index = Series(ksds.index)
    ks_index = ks_index.quantile(q = qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print ('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))

    # chart
    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
                         color='blue', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
                        color='red', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.ks, label='ks',
                   color='green', linestyle='-', linewidth=2)

    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(),'cumsum_bad'], color='red', linestyle='--')
    plt.title('KS=%s ' %np.round(ks_value, 4) +
                'at Pop=%s' %np.round(ks_pop, 4), fontsize=5)

    return ksds
####################### over ##########################

#logistic
#标准化

#LOGCV_fin=0
#LOG_index=0
#i=0.01
#while(i<=0.5):
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#log回归
logmodel = LogisticRegression(class_weight={0:0.21})
#logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
logpred = logmodel.predict(X_test)
#结果
print(confusion_matrix(y_test, logpred))
#print(round(fbeta_score(y_test, logpred,1.2),4)*100)
#print(round(recall_score(y_test, logpred),4)*100)
LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=k_fold, n_jobs=1, scoring = f3_score).mean())#70.47
#LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'recall').mean())#89.8970
print(LOGCV)
#    LOGCV_fin=LOGCV
#    LOG_index=i
#print(i)
#i+=0.01

#KNN
'''
#k的个数
#X_trainK, X_testK, y_trainK, y_testK = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 101)

smote_enn = SMOTEENN(random_state=0,sampling_strategy=0.30)
X_trainK, X_testK, y_trainK, y_testK = train_test_split(bank_final,data['y'], test_size = 0.2, random_state = 101)
X_trainK, y_trainK = smote_enn.fit_sample(X_trainK, y_trainK)

#Neighbors
neighbors = np.arange(0,25)
#Create empty list that will hold cv scores
cv_scores = []
#Perform 10-fold cross validation on training set for odd values of k:
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='distance', p=2, metric='euclidean')
    #knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, random_state=101)
    scores = model_selection.cross_val_score(knn, X_trainK, y_trainK, cv=kfold, scoring = f3_score)
    cv_scores.append(scores.mean()*100)
    print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train f1.2')
plt.show()#看k的个数     3
'''



#k的个数   2
#i=0
#while i<=25:
    #i=i+1
knn = KNeighborsClassifier(n_neighbors=2,weights='distance',p=2,metric='euclidean')
knn.fit(X_train, y_train)
knnpred = knn.predict(X_test)

#print(confusion_matrix(y_test, knnpred))
#print(round(fbeta_score(y_test, knnpred,1.2),4)*100)
KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring =f3_score).mean())#75.07
print(KNNCV)
print(confusion_matrix(y_test, knnpred))
#print(classification_report(y_test, knnpred))



#支持向量机


#i=0.01
#while i<=0.2:
svc= SVC(kernel = 'sigmoid',class_weight={0:0.06}, random_state = 121)
svc.fit(X_train, y_train)
svcpred = svc.predict(X_test)
#print(confusion_matrix(y_test, svcpred))
#print(round(fbeta_score(y_test, svcpred,1.2),4)*100)
SVCCV = (cross_val_score(svc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = f3_score).mean())#75.07
print(SVCCV)
#print(i)
#i=i+0.01
print(confusion_matrix(y_test, svcpred))
#print(classification_report(y_test, svcpred))


#决策树
#i=0.70
#while i<=0.8:
#dtree = DecisionTreeClassifier(criterion='gini', splitter='best',class_weight='balanced') #criterion = entopy, gini
dtree = DecisionTreeClassifier(criterion='gini', splitter='best',class_weight={0:0.6},random_state = 131)
dtree.fit(X_train, y_train)
dtreepred = dtree.predict(X_test)

#print(confusion_matrix(y_test, dtreepred))
#print(round(fbeta_score(y_test, dtreepred,1.2),4)*100)
DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = f3_score).mean())#71.42
print(DTREECV)
#print(i)
#i=i+0.01
print(confusion_matrix(y_test,dtreepred))
#print(classification_report(y_test, dtreepred))

#随机森林
#i=0
#while i<=1:
#rfc = RandomForestClassifier(n_estimators = 200,class_weight='balanced')#criterion = entopy,gini
rfc = RandomForestClassifier(n_estimators = 1000,class_weight={0:0.7},random_state = 141)#criterion = entopy,gini
rfc.fit(X_train, y_train)
rfcpred = rfc.predict(X_test)

#print(confusion_matrix(y_test, rfcpred ))
#print(round(fbeta_score(y_test, rfcpred,1.2),4)*100)
RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = f3_score).mean())#78.60
print(RFCCV)
#print(i)
#i=i+0.1
print(confusion_matrix(y_test, rfcpred))
#print(classification_report(y_test, rfcpred))


#朴素贝叶斯
gaussiannb= GaussianNB()
gaussiannb.fit(X_train, y_train)
gaussiannbpred = gaussiannb.predict(X_test)
probs = gaussiannb.predict(X_test)

#print(confusion_matrix(y_test, gaussiannbpred ))
#print(round(fbeta_score(y_test, gaussiannbpred,1.2),4)*100)
GAUSIAN = (cross_val_score(gaussiannb, X_train, y_train, cv=k_fold, n_jobs=1, scoring =  f3_score).mean())#62.29
print(GAUSIAN)
print(confusion_matrix(y_test, gaussiannbpred))
#print(classification_report(y_test,gaussiannbpred))



#xgboost
#i=10
#while i<=20:
#xgb = XGBClassifier(eta=0.1)
xgb = XGBClassifier(objective='binary:logistic',eta=0.1,scale_pos_weight=15.5,n_estimators=100,max_dept=5,seed=1234)
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)

#print(confusion_matrix(y_test, xgbprd ))
XGB = (cross_val_score(xgb,X_train,y_train, cv = k_fold, scoring = f3_score).mean())#74.04
print(XGB)
#print(i)
#i=i+1
print(confusion_matrix(y_test, xgbprd))
#print(classification_report(y_test, xgbprd))


#梯度提升GBDT
gbk = GradientBoostingClassifier(n_estimators = 200)
gbk.fit(X_train, y_train)
gbkpred = gbk.predict(X_test)
#print(confusion_matrix(y_test, gbkpred ))
#print(round(fbeta_score(y_test, xgbprd,1.2),4)*100)
#print(round(fbeta_score(y_test, gbkpred,1.2),4)*100)
GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring = f3_score).mean())#76.72
print(GBKCV)
print(confusion_matrix(y_test, gbkpred))
#print(classification_report(y_test,gbkpred))



#模型比较
models = pd.DataFrame({
                'Models': ['Random Forest Classifier', 'Decision Tree Classifier', 'Support Vector Machine',
                           'K-Near Neighbors', 'Logistic Model', 'Gausian NB', 'XGBoost', 'Gradient Boosting'],
                'Score':  [RFCCV, DTREECV, SVCCV, KNNCV, LOGCV, GAUSIAN, XGB, GBKCV]})

models.sort_values(by='Score', ascending=False)



'''    Models     Score
4            Logistic Model  0.570637
2    Support Vector Machine  0.562268
6                   XGBoost  0.484842
5                Gausian NB  0.430358
0  Random Forest Classifier  0.359459
7         Gradient Boosting  0.347967
1  Decision Tree Classifier  0.325711
3          K-Near Neighbors  0.316308
'''


'''                     Models     Score
3          K-Near Neighbors  0.963390
0  Random Forest Classifier  0.937534
7         Gradient Boosting  0.909539
1  Decision Tree Classifier  0.904313
6                   XGBoost  0.893411
4            Logistic Model  0.879357
2    Support Vector Machine  0.805039
5                Gausian NB  0.743199
'''

#调试区#################################################################################
# 单层决策树生成函数
# 这个函数作用是比较数据的某个特征的值和阈值的大小，通过这样进行分类，但是只看这个函数不好理解，需要
# 结合下面的函数进行理解
def stumpclassify(datamatrix, dimen, threshval, threshineq):
    '''
    :param datamatrix: 输入待分类的数据
    :param dimen: 输入数据的某个特征
    :param threshval: 设定的阈值
    :param threshineq: 阈值比较
    :return: 返回分类的结果
    '''
    retarray = np.ones((np.shape(datamatrix)[0], 1))  # 先默认分类都为1
    if threshineq == 'lt':  # 这个是为了找到最优的决策，因此两种情况都讨论了，即大于阈值和小于阈值
        #retarray[datamatrix[:, dimen] <= threshval] = -1.0
        retarray[datamatrix[:, dimen] <= threshval] = .0
        # 当数据小于阈值时为-1，因为默认为1了，为了准确率，需要考虑大于阈值的情况
        # 修改为0
    else:
        #retarray[datamatrix[:, dimen] > threshval] = -1.0
        retarray[datamatrix[:, dimen] > threshval] = 0
        # 如果考虑大于阈值的情况则也是为-1，这里大家可能会有疑问，这是两种情况，调用这个函数的
        # 函数需要知道错误率最小的决策及阈值，因此他把两种情况都考虑了，即每次前进一步阈值都会更新，每次更新都计算大这个阈值或者小于这个阈值的情况
    return retarray

def buildstump(dataarr, classlabels, D):
    '''
    :param dataarr: 输入数据
    :param classlabels:  数据的真实分类标签
    :param  D: 数据的权值向量
    :return: beststump, minerror, bestclasest 即决策树，最小误差，预测值
    '''
    datamatrix = np.matrix(dataarr)  # 把数据转换为矩阵数据
    labelsmat = np.mat(classlabels).T  # 同理把标签数据转换为矩阵   .T  转置
    m, n = np.shape(datamatrix)  # 得到数据的维度即m行n列
    numsteps = 50.0  # 设置步数，目的是在步数以内找到最优的决策树
    beststump = {}  # 先建立一个空的字典，用作后面存储决策树
    bestclasest = np.mat(np.zeros((m, 1)))  # 预测分类空矩阵
    minerror = np.inf  # 错误率先设置为最大

    for i in range(n):  # 先遍历数据的所有特征
        rangemin = datamatrix[:, i].min()  # 寻找该特征下的最小值
        rangemax = datamatrix[:, i].max()  # 寻找该特征下的最大值
        stepsize = (rangemax - rangemin)/numsteps  # 通过上面来计算步长，为了找到最优的决策
        for j in range(-1, int(numsteps)+1):  # 上面说计算是在numsteps以内找到最优的，因此这个循环是步数
            for inequal in ['lt', 'gt']:  # 遍历大于或者小于两种情况，lt= less than  ， gt = great than
                threshval = (rangemin + float(j)*stepsize)  # 通过设置的步和步长计算该步的阈值（他把每一步的值用作阈值，这样找到最优的）
                predictedvals = stumpclassify(datamatrix, i, threshval, inequal)  # 调用函数进行预测所有数据，现在再回去看看这个函数
                errarr = np.mat(np.ones((m, 1)))  # 准备计算错误率类，默认全为1
                errarr[predictedvals == labelsmat] = 0  # 如果相同和真正类别相同则为0，则剩下为1则为分类错误的
                weightederror = D.T*errarr # 保留分错的数据， 分类正确的数据直接为0
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"\
                       #% (i, threshval, inequal, weightederror))
                if weightederror < minerror:  # 如果误差比之前的还小则更新新返回值，反之继续循环直达循环结束，返回
                    minerror = weightederror
                    bestclasest = predictedvals.copy()
                    beststump['dim'] = i
                    beststump['thresh'] = threshval
                    beststump['ineq'] = inequal

    return beststump, minerror, bestclasest
###深度

##################以上是决策树   单层#########################

def adaBoostTrainDS(dataArr,classLabels,numIt):
    weakClassArr = [] # 存储训练好的决策树使用的
    m = np.shape(dataArr)[0] # 取出数据的行
    D = np.mat(np.ones((m,1))/m)   #数据的权值初始相等
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildstump(dataArr,classLabels,D)# 建立第i个决策树
        #print("D:",D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#计算决策树权值
        bestStump['alpha'] = alpha # 更新权值
        weakClassArr.append(bestStump)  # 把第i个决策树添存储
        #print("classEst: ", classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) # 更新数据权值D
        D = np.multiply(D, np.exp(expon))  # 看不懂的建议吧原理搞懂再来看
        D = D/D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print("aggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        #print("total error: ",errorRate)
        if errorRate == 0.0: break # 错误率为0时返回迭代
    return weakClassArr,aggClassEst

"""
Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
Returns:
        分类结果
"""
def ada_classify(dat2class, classifier_arr):
    # 输入是待分样例和弱分类器集合
    # 仅仅是利用了前面训练好的分类器
    data_matrix = np.mat(dat2class)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))  # 全0列向量
    for i in range(len(classifier_arr)):  # 遍历所有的弱分类器
        # 用单层决策树获得每个分类器自己的决策值
        class_est = stumpclassify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])  # call stump classify
        # 输出结果是累加
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        # print(agg_class_est)  # 这个例子中，值越来越小
    # 要满足不超界，大于0为+1，小于0为-1
    return np.sign(agg_class_est)

def plotROC(pred_strengths, class_labels):
    # AUC，曲线下的面积
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # cursor  # 起始点
    y_sum = 0.0  # variable to calculate AUC
    num_pos_clas = sum(np.array(class_labels) == 1.0)  # 正例的数目
    # 这两个是步长
    y_step = 1 / float(num_pos_clas)
    x_step = 1 / float(len(class_labels) - num_pos_clas)
    # 从小到大排列，再得到下标
    sorted_indicies = pred_strengths.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()  # 清空
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sorted_indicies.tolist()[0]:  # np对象变成list
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", y_sum * x_step)

data_arr=X_train
label_arr =y_train
classifier_array, agg_class_est = adaBoostTrainDS(data_arr, label_arr, 500)

# 用训练好的数据来
test_arr=X_test
test_label_arr =y_test
prediction = ada_classify(test_arr, classifier_array)

err_arr = np.mat(np.ones([len(test_label_arr), 1]))
err_num = err_arr[prediction != np.mat(test_label_arr).T].sum()
print("accuracy:%.2f" % (1 - err_num / float(len(test_label_arr))))
#plotROC(agg_class_est.T, label_arr)

ada=ada_classify(test_arr, classifier_array)
ada.fit(X_test, y_test)
ADACV = (cross_val_score(ada, X_train, y_train, cv=k_fold, n_jobs=1, scoring = f1_2_score).mean())#76.72
print(confusion_matrix(y_test, prediction))


fprlog_test, tprlog_test, thresholdlog_test = metrics.roc_curve(y_test, prediction)
roc_auclog = metrics.auc(fprlog_test, tprlog_test)

ax.plot(fprlog_test, tprlog_test, 'b', label = 'AUC = %0.2f' % roc_auclog)
ax.plot([0, 1], [0, 1],'r--')
ax.set_title('Receiver Operating Characteristic XGBOOST ',fontsize=10)
ax.set_ylabel('True Positive Rate',fontsize=10)
ax.set_xlabel('False Positive Rate',fontsize=7)
ax.legend(loc = 'lower right', prop={'size': 6})











'''
1

                     Models     Score
3          K-Near Neighbors  0.961382
0  Random Forest Classifier  0.937634
7         Gradient Boosting  0.910601
1  Decision Tree Classifier  0.904069
6                   XGBoost  0.891607
4            Logistic Model  0.851647
5                Gausian NB  0.741874
2    Support Vector Machine  0.678765
'''

'''
2
                     
                     Models     Score
3          K-Near Neighbors  0.952525
0  Random Forest Classifier  0.939376
7         Gradient Boosting  0.911676
1  Decision Tree Classifier  0.905205
6                   XGBoost  0.892845
4            Logistic Model  0.818079
5                Gausian NB  0.695504
2    Support Vector Machine  0.658437
'''



def feature_analyze(model, to_print=False, to_plot=True, csv_path=None):
    """XGBOOST 模型特征重要性分析。

    Args:
        model: 训练好的 xgb 模型。
        to_print: bool, 是否输出每个特征重要性。
        to_plot: bool, 是否绘制特征重要性图表。
        csv_path: str, 保存到 csv 文件路径。
    """
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    if to_plot:
        features = list()
        scores = list()
        for (key, value) in feature_score:
            features.append(key)
            scores.append(value)
        plt.barh(range(len(scores)), scores)
        plt.yticks(range(len(scores)), features)
        for i in range(len(scores)):
            plt.text(scores[i] + 0.75, i - 0.25, scores[i])
        plt.xlabel('feature socre')
        plt.title('feature score evaluate')
        plt.grid()
        plt.show()
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    if to_print:
        print(''.join(fs))
    if csv_path is not None:
        with open(csv_path, 'w') as f:
            f.writelines("feature,score\n")
            f.writelines(fs)
    return feature_score


feature_analyze(xgb, to_print=False, to_plot=True, csv_path=None)



fig, (ax, ax1) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
probs = xgb.predict_proba(X_test)

preds = probs[:,1]
fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(y_test, preds)
roc_aucxgb = metrics.auc(fprxgb, tprxgb)

ax.plot(fprxgb, tprxgb, 'b', label = 'AUC = %0.2f' % roc_aucxgb)
ax.plot([0, 1], [0, 1],'r--')
ax.set_title('Receiver Operating Characteristic XGBOOST ',fontsize=10)
ax.set_ylabel('True Positive Rate',fontsize=10)
ax.set_xlabel('False Positive Rate',fontsize=7)
ax.legend(loc = 'lower right', prop={'size': 6})




'''

plt.figure(1) # 创建图表1
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')


preds = probs[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, preds)
plt.figure(1)
plt.plot(recall,precision)
plt.show()

PlotKS(preds=preds, labels=y_test, n=200, asc=0)#ks值   区分度


'''



#Gradient
probs = gbk.predict_proba(X_test)
preds = probs[:,1]
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, preds)
roc_aucgbk = metrics.auc(fprgbk, tprgbk)

ax1.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)
ax1.plot([0, 1], [0, 1],'r--')
ax1.set_title('Receiver Operating Characteristic GRADIENT BOOST ',fontsize=10)
ax1.set_ylabel('True Positive Rate',fontsize=10)
ax1.set_xlabel('False Positive Rate',fontsize=7)
ax1.legend(loc = 'lower right', prop={'size': 6})

plt.subplots_adjust(wspace=1)

'''

plt.figure(1) # 创建图表1
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')


preds = probs[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, preds)
plt.figure(1)
plt.plot(recall,precision)
plt.show()

PlotKS(preds=preds, labels=y_test, n=200, asc=0)#ks值   区分度


'''

#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 4))
fig, ax_arr = plt.subplots(nrows = 2, ncols = 3, figsize = (10,7))

#LOGMODEL
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fprlog, tprlog, thresholdlog = metrics.roc_curve(y_test, preds)
roc_auclog = metrics.auc(fprlog, tprlog)

ax_arr[0,0].plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)
ax_arr[0,0].plot([0, 1], [0, 1],'r--')
ax_arr[0,0].set_title('Receiver Operating Characteristic Logistic ',fontsize=5)
ax_arr[0,0].set_ylabel('True Positive Rate',fontsize=10)
ax_arr[0,0].set_xlabel('False Positive Rate',fontsize=7)
ax_arr[0,0].legend(loc = 'lower right', prop={'size': 6})

'''

plt.figure(1) # 创建图表1
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')


preds = probs[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, preds)
plt.figure(1)
plt.plot(recall,precision)
plt.show()

PlotKS(preds=preds, labels=y_test, n=200, asc=0)#ks值   区分度


'''

#RANDOM FOREST --------------------
probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

ax_arr[0,1].plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
ax_arr[0,1].plot([0, 1], [0, 1],'r--')
ax_arr[0,1].set_title('Receiver Operating Characteristic Random Forest ',fontsize=5)
ax_arr[0,1].set_ylabel('True Positive Rate',fontsize=10)
ax_arr[0,1].set_xlabel('False Positive Rate',fontsize=7)
ax_arr[0,1].legend(loc = 'lower right', prop={'size': 6})

'''

plt.figure(1) # 创建图表1
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')


preds = probs[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, preds)
plt.figure(1)
plt.plot(recall,precision)
plt.show()

PlotKS(preds=preds, labels=y_test, n=200, asc=0)#ks值   区分度


'''

#KNN----------------------
probs = knn.predict_proba(X_test)
preds = probs[:,1]
fprknn, tprknn, thresholdknn = metrics.roc_curve(y_test, preds)
roc_aucknn = metrics.auc(fprknn, tprknn)

ax_arr[0,2].plot(fprknn, tprknn, 'b', label = 'AUC = %0.2f' % roc_aucknn)
ax_arr[0,2].plot([0, 1], [0, 1],'r--')
ax_arr[0,2].set_title('Receiver Operating Characteristic KNN ',fontsize=5)
ax_arr[0,2].set_ylabel('True Positive Rate',fontsize=10)
ax_arr[0,2].set_xlabel('False Positive Rate',fontsize=7)
ax_arr[0,2].legend(loc = 'lower right', prop={'size': 6})

'''

plt.figure(1) # 创建图表1
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')


preds = probs[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, preds)
plt.figure(1)
plt.plot(recall,precision)
plt.show()

PlotKS(preds=preds, labels=y_test, n=200, asc=0)#ks值   区分度


'''

#DECISION TREE ---------------------
probs = dtree.predict_proba(X_test)
preds = probs[:,1]
fprdtree, tprdtree, thresholddtree = metrics.roc_curve(y_test, preds)
roc_aucdtree = metrics.auc(fprdtree, tprdtree)

ax_arr[1,0].plot(fprdtree, tprdtree, 'b', label = 'AUC = %0.2f' % roc_aucdtree)
ax_arr[1,0].plot([0, 1], [0, 1],'r--')
ax_arr[1,0].set_title('Receiver Operating Characteristic Decision Tree ',fontsize=5)
ax_arr[1,0].set_ylabel('True Positive Rate',fontsize=10)
ax_arr[1,0].set_xlabel('False Positive Rate',fontsize=7)
ax_arr[1,0].legend(loc = 'lower right', prop={'size': 6})

'''

plt.figure(1) # 创建图表1
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')


preds = probs[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, preds)
plt.figure(1)
plt.plot(recall,precision)
plt.show()

PlotKS(preds=preds, labels=y_test, n=200, asc=0)#ks值   区分度


'''

#GAUSSIAN ---------------------
probs = gaussiannb.predict_proba(X_test)
preds = probs[:,1]
fprgau, tprgau, thresholdgau = metrics.roc_curve(y_test, preds)
roc_aucgau = metrics.auc(fprgau, tprgau)

ax_arr[1,1].plot(fprgau, tprgau, 'b', label = 'AUC = %0.2f' % roc_aucgau)
ax_arr[1,1].plot([0, 1], [0, 1],'r--')
ax_arr[1,1].set_title('Receiver Operating Characteristic Gaussian ',fontsize=5)
ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=10)
ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=7)
ax_arr[1,1].legend(loc = 'lower right', prop={'size': 6})

'''

plt.figure(1) # 创建图表1
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')


preds = probs[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, preds)
plt.figure(1)
plt.plot(recall,precision)
plt.show()

PlotKS(preds=preds, labels=y_test, n=200, asc=0)#ks值   区分度


'''

#ALL PLOTS ----------------------------------
ax_arr[1,2].plot(fprgau, tprgau, 'b', label = 'Gaussian', color='black')
ax_arr[1,2].plot(fprdtree, tprdtree, 'b', label = 'Decision Tree', color='blue')
ax_arr[1,2].plot(fprknn, tprknn, 'b', label = 'Knn', color='brown')
ax_arr[1,2].plot(fprrfc, tprrfc, 'b', label = 'Random Forest', color='green')
ax_arr[1,2].plot(fprlog, tprlog, 'b', label = 'Logistic', color='grey')
ax_arr[1,2].set_title('Receiver Operating Comparison ',fontsize=5)
ax_arr[1,2].set_ylabel('True Positive Rate',fontsize=10)
ax_arr[1,2].set_xlabel('False Positive Rate',fontsize=7)
ax_arr[1,2].legend(loc = 'lower right', prop={'size': 6})

plt.subplots_adjust(wspace=0.2)
plt.tight_layout()

