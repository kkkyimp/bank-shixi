import random
from pandas import  DataFrame
import numpy as np
import pandas as pd
from pandas import Series
from interval import Interval

datafile = '/Users/kkky/PycharmProjects/shixi/bank-additional-full.csv'#文件所在位置
data = pd.read_csv(datafile)#如果是csv文件则用read_csv

data1=data['education']
data1 = data1.dropna()

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



##饼状图分析

import numpy as np
import matplotlib.pyplot as plt

us=data.ix[data['loan'].isna()]

def useless(col):
    len0=len(col)
    recol=col.value_counts()#分组

    #修改编码
    recol_ix=list(recol.index)
    col=col.replace(recol_ix,list(range(0,len(recol))))

    len1=len(recol)
    num=recol.sum()#求总个数
    f = lambda x: x/num
    recol1=recol.apply(f)#概率化
    ##先验概率
    return recol1

end1=data['education']
end2=us['education']
u1=useless(end1)
u2=useless(end2)
u1u2



fracs=us['age'].value_counts()
ix=list(fracs.index)
plt.axes(aspect=1)
plt.pie(x=list(fracs),labels=ix)
plt.show()



data.to_csv('Users/kkky/PycharmProjects/shixi/111.csv',sep=",",index=TRUE,header=TRUE)











# XGBOOST ROC/ AUC , BEST MODEL
from sklearn import metrics



fig, (ax, ax1) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
probs = xgb.predict_proba(X_test)

preds = probs[:,1]
fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(y_test, preds)
roc_aucxgb = metrics.auc(fprxgb, tprxgb)

ax.plot(fprxgb, tprxgb, 'b', label = 'AUC = %0.2f' % roc_aucxgb)
ax.plot([0, 1], [0, 1],'r--')
ax.set_title('Receiver Operating Characteristic XGBOOST ',fontsize=10)
ax.set_ylabel('True Positive Rate',fontsize=20)
ax.set_xlabel('False Positive Rate',fontsize=15)
ax.legend(loc = 'lower right', prop={'size': 16})

#Gradient
probs = gbk.predict_proba(X_test)
preds = probs[:,1]
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, preds)
roc_aucgbk = metrics.auc(fprgbk, tprgbk)

ax1.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)
ax1.plot([0, 1], [0, 1],'r--')
ax1.set_title('Receiver Operating Characteristic GRADIENT BOOST ',fontsize=10)
ax1.set_ylabel('True Positive Rate',fontsize=20)
ax1.set_xlabel('False Positive Rate',fontsize=15)
ax1.legend(loc = 'lower right', prop={'size': 16})

plt.subplots_adjust(wspace=1)



#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 4))
fig, ax_arr = plt.subplots(nrows = 2, ncols = 3, figsize = (20,15))

#LOGMODEL
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fprlog, tprlog, thresholdlog = metrics.roc_curve(y_test, preds)
roc_auclog = metrics.auc(fprlog, tprlog)

ax_arr[0,0].plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)
ax_arr[0,0].plot([0, 1], [0, 1],'r--')
ax_arr[0,0].set_title('Receiver Operating Characteristic Logistic ',fontsize=20)
ax_arr[0,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,0].legend(loc = 'lower right', prop={'size': 16})

#RANDOM FOREST --------------------
probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

ax_arr[0,1].plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
ax_arr[0,1].plot([0, 1], [0, 1],'r--')
ax_arr[0,1].set_title('Receiver Operating Characteristic Random Forest ',fontsize=20)
ax_arr[0,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,1].legend(loc = 'lower right', prop={'size': 16})

#KNN----------------------
probs = knn.predict_proba(X_test)
preds = probs[:,1]
fprknn, tprknn, thresholdknn = metrics.roc_curve(y_test, preds)
roc_aucknn = metrics.auc(fprknn, tprknn)

ax_arr[0,2].plot(fprknn, tprknn, 'b', label = 'AUC = %0.2f' % roc_aucknn)
ax_arr[0,2].plot([0, 1], [0, 1],'r--')
ax_arr[0,2].set_title('Receiver Operating Characteristic KNN ',fontsize=20)
ax_arr[0,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,2].legend(loc = 'lower right', prop={'size': 16})

#DECISION TREE ---------------------
probs = dtree.predict_proba(X_test)
preds = probs[:,1]
fprdtree, tprdtree, thresholddtree = metrics.roc_curve(y_test, preds)
roc_aucdtree = metrics.auc(fprdtree, tprdtree)

ax_arr[1,0].plot(fprdtree, tprdtree, 'b', label = 'AUC = %0.2f' % roc_aucdtree)
ax_arr[1,0].plot([0, 1], [0, 1],'r--')
ax_arr[1,0].set_title('Receiver Operating Characteristic Decision Tree ',fontsize=20)
ax_arr[1,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,0].legend(loc = 'lower right', prop={'size': 16})

#GAUSSIAN ---------------------
probs = gaussiannb.predict_proba(X_test)
preds = probs[:,1]
fprgau, tprgau, thresholdgau = metrics.roc_curve(y_test, preds)
roc_aucgau = metrics.auc(fprgau, tprgau)

ax_arr[1,1].plot(fprgau, tprgau, 'b', label = 'AUC = %0.2f' % roc_aucgau)
ax_arr[1,1].plot([0, 1], [0, 1],'r--')
ax_arr[1,1].set_title('Receiver Operating Characteristic Gaussian ',fontsize=20)
ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,1].legend(loc = 'lower right', prop={'size': 16})

#ALL PLOTS ----------------------------------
ax_arr[1,2].plot(fprgau, tprgau, 'b', label = 'Gaussian', color='black')
ax_arr[1,2].plot(fprdtree, tprdtree, 'b', label = 'Decision Tree', color='blue')
ax_arr[1,2].plot(fprknn, tprknn, 'b', label = 'Knn', color='brown')
ax_arr[1,2].plot(fprrfc, tprrfc, 'b', label = 'Random Forest', color='green')
ax_arr[1,2].plot(fprlog, tprlog, 'b', label = 'Logistic', color='grey')
ax_arr[1,2].set_title('Receiver Operating Comparison ',fontsize=20)
ax_arr[1,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,2].legend(loc = 'lower right', prop={'size': 16})

plt.subplots_adjust(wspace=0.2)
plt.tight_layout()



def ks_calc_auc(data,score_col,class_col):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    score_col: 一维数组或series，代表模型得分（一般为预测正类的概率）
    class_col: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值
    '''
    fpr,tpr,threshold = roc_curve((1-data[class_col[0]]).ravel(),data[score_col[0]].ravel())
    ks = max(tpr-fpr)
    return ks








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









#!/usr/bin/env/python
# -*- coding: utf-8 -*-
# Author: 赵守风
# File name: adaboost.py
# Time:2018/10/27
# Email:1583769112@qq.com

# 建立数据，用于后面的训练
def loadsimdata():
    datmat = np.matrix([[1, 2.1],
                        [2., 1.1],
                        [1.3, 1],
                        [1, 1],
                        [2., 1.]])
    classlabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return datmat, classlabels


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
    numsteps = 10.0  # 设置步数，目的是在步数以内找到最优的决策树
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
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"\
                       % (i, threshval, inequal, weightederror))
                if weightederror < minerror:  # 如果误差比之前的还小则更新新返回值，反之继续循环直达循环结束，返回
                    minerror = weightederror
                    bestclasest = predictedvals.copy()
                    beststump['dim'] = i
                    beststump['thresh'] = threshval
                    beststump['ineq'] = inequal

    return beststump, minerror, bestclasest




#!/usr/bin/env/python
# -*- coding: utf-8 -*-
# Author: 赵守风
# File name: test.py
# Time:2018/10/27
# Email:1583769112@qq.com

datmat, classlabels = adaboost.loadsimdata()
print('datmat: ', datmat)
print('classlabels', classlabels)
D = np.mat(np.ones((5, 1))/5)



adaboost.buildstump(datmat, classlabels, D)


'''完整AdaBoost算法实现
算法实现伪代码
对每次迭代：
    利用buildStump()函数找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量D
    更新累计类别估计值
    如果错误率为等于0.0，退出循环
'''



def adaBoostTrainDS(dataArr,classLabels,numIt=100):
    weakClassArr = [] # 存储训练好的决策树使用的
    m = np.shape(dataArr)[0] # 取出数据的行
    D = np.mat(np.ones((m,1))/m)   #数据的权值初始相等
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildstump(dataArr,classLabels,D)# 建立第i个决策树
        print("D:",D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#计算决策树权值
        bestStump['alpha'] = alpha # 更新权值
        weakClassArr.append(bestStump)  # 把第i个决策树添存储
        print("classEst: ", classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) # 更新数据权值D
        D = np.multiply(D, np.exp(expon))  # 看不懂的建议吧原理搞懂再来看
        D = D/D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        print("aggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: break # 错误率为0时返回迭代
    return weakClassArr,aggClassEst


def ada_classify(dat2class, classifier_arr):
    # 输入是待分样例和弱分类器集合
    # 仅仅是利用了前面训练好的分类器
    data_matrix = np.mat(dat2class)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))  # 全0列向量
    for i in range(len(classifier_arr)):  # 遍历所有的弱分类器
        # 用单层决策树获得每个分类器自己的决策值
        class_est = stump_classify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
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


label_arr =y_train
classifier_array, agg_class_est = adaBoostTrainDS(data_arr, label_arr, 100)

# 用训练好的数据来
test_arr=X_test
test_label_arr =y_test
prediction = ada_classify(test_arr, classifier_array)

err_arr = np.mat(np.ones([len(test_label_arr), 1]))
err_num = err_arr[prediction != np.mat(test_label_arr).T].sum()
print("accuracy:%.2f" % (1 - err_num / float(len(test_label_arr))))
# 错了16个,准确率只有0.76
# 但是用logistic错误率有0.35，这里只用了很少弱分类器
# 分类器再多会导致过拟合
plotROC(agg_class_est.T, label_arr)




