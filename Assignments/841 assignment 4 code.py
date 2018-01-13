# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:33:48 2017

@author: hyq92
"""

from numpy import *

def loadSimpleData():
    dataMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMat, dimen, threshVal, ineq):
    retArr = ones((dataMat.shape[0], 1))
    if ineq == 'lt':
        retArr[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArr[dataMat[:, dimen] > threshVal] = -1.0
    return retArr

df =  loadSimpleData()
#retArr = stumpClassify(dataMat = df[0], dimen=1, threshVal =1, ineq = 'lt')

def buildStump(dataArr, classLabels, D):
    dataMat = mat(dataArr); labelMat = mat(classLabels).T
    m, n = dataMat.shape
    numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = min(dataMat[:, i]); rangeMax = max(dataMat[:, i])
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predicatedVal = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                #for the row that predicatedVal == labelMat, errArr[row] = 0
                errArr[predicatedVal == labelMat] = 0
                weightedError = D.T * errArr
                #print 'split: dim %d, thesh %.2f, ineqal: %s, \
                #weighted error:%.3f' %(i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predicatedVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    
    return bestStump, minError, bestClassEst   


'''
DS: decision stump
1. calculate alpha
alpha tells the total classifier how much to weight the output from the stump.
max(error, 1e-16) to make sure you don't have a divide-by-zero error in the 
case where there's no error.
2. calculate D
multiply(): element-wise product.
exp(array): return an array where each element is calculated by exp.
3. aggClassEst.
It is a floating point number, to get the binary class, use the sign() function. 
'''
def adaBoostDS(dataArr, classLabels, numIter = 40):
    bestStumpArr = []
    m = dataArr.shape[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIter):
        bestStump, error, bestClassEst = buildStump(dataArr, classLabels, D)
#        print('D:', D.T)
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        bestStumpArr.append(bestStump)
#        print('ClassEst:', bestClassEst.T)
        
        #multiply(): element-wise product. class real result X estimation
        expon = multiply(-1 * alpha * mat(classLabels).T, bestClassEst)
        #exp(expon): calculate exp for each element in mat expon
        D = multiply(D, exp(expon)) / sum(D)
        
        #aggClassEst is float mat.
        aggClassEst += alpha * bestClassEst
#        print('aggClassEst:', aggClassEst)
        
        #aggClassEst is float mat, use its sign to compare with mat classLabels
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
        errorRate = sum(aggError)/m
#        print('total error:', errorRate)
        
        if errorRate == 0.0:
            break
        
    return bestStumpArr

adaboost_cls_s = adaBoostDS(dataArr = df[0], classLabels=df[1], numIter = 40)

def adaClassify(dataToClass, classifierArr):
    dataMat = mat(dataToClass)
    m = dataMat.shape[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
#        print(aggClassEst)
    return sign(aggClassEst)

y_pred = adaClassify(dataToClass = df[0], classifierArr=adaboost_cls_s)
df[1]

# thes last col is class label.                                              
def loadDataSet(fileName):
    numFeature = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineList = []
        curLine = line.strip().split('\t')
        for i in range(numFeature - 1):
            lineList.append(float(curLine[i]))
        dataMat.append(lineList)
        labelMat.append(float(curLine[-1]))
    return array(dataMat), labelMat

# on real data
df2_train = loadDataSet(fileName = 'horseColicTraining2.txt')
df2_test = loadDataSet(fileName = 'horseColicTest2.txt')


adaboost_cls = adaBoostDS(dataArr = df2_train[0], classLabels=df2_train[1], numIter = 40)

def ErrorCalculation(train, test):
    y_pred_train = adaClassify(dataToClass = train[0], classifierArr=adaboost_cls)
    training_error = mean(y_pred_train == train[1])
    
    y_pred_test = adaClassify(dataToClass = test[0], classifierArr=adaboost_cls)
    testing_error = mean(y_pred_test == test[1])
    return training_error, testing_error

print("Training error for 40 interations is ", training_error, "Testing error for 40 interations is ", testing_error)
# Training error for 40 interations is  0.525184282055 Testing error for 40 interations is  0.545110269548

results_train_E = []
results_test_E = []
for i in range(10, 100, 1):
    adaboost_cls = adaBoostDS(dataArr = df2_train[0], classLabels=df2_train[1], numIter = i)
    results_train_E.append(ErrorCalculation(train = df2_train, test = df2_test)[0])
    results_test_E.append(ErrorCalculation(train =  df2_train, test = df2_test)[1])

# for n = 4, epoch = 50:900, lr = 0.01, l2_reg = 0.000001
import matplotlib.pyplot as plt
plt.plot(range(10, 100, 1), results_train_E, 'r--')
plt.axis([0, 100, 0.5, 0.6])

# testing error curve
plt.plot(range(10, 100, 1), results_test_E, 'r--', color='green')
plt.axis([0, 100, 0.5, 0.6])
plt.legend(['Training Error', 'Testing Error'], loc='upper left')
plt.title('Training & Testing Error vs. Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Error Rate')
#plt.text(50, 0.3, 'nodes = 4, lr = 0.01, l2_reg = 0.000001')
plt.show()


# Question 2
# generate training dataset
df3_train_X = random.standard_normal(20000)
df3_train_X = df3_train_X.reshape(2000,10)
rm1 =sum(df3_train_X**2, axis=1)

df3_train_Y = zeros(2000)
for i in range(0,2000):
    if rm1[i] >= 9.34:
        df3_train_Y[i] = 1
    else:
        df3_train_Y[i] = -1

# generate testing dataset
df3_test_X = random.standard_normal(100000)
df3_test_X = df3_test_X.reshape(10000,10)
rm2 =sum(df3_test_X**2, axis=1)

df3_test_Y = zeros(10000)
for i in range(0,10000):
    if rm2[i] >= 9.34:
        df3_test_Y[i] = 1
    else:
        df3_test_Y[i] = -1

# train model based on training set
adaboost_cls = adaBoostDS(dataArr = df3_train_X, classLabels=df3_train_Y, numIter = 30)
y_pred_train = adaClassify(dataToClass = df3_train_X, classifierArr=adaboost_cls_s)

def ErrorCalculation(trainX, trainY, testX, testY):
    y_pred_train = adaClassify(dataToClass = trainX, classifierArr=adaboost_cls)
    training_error = sum(y_pred_train.flatten() != trainY)/len(trainY)
    
    y_pred_test = adaClassify(dataToClass = testX, classifierArr=adaboost_cls)
    testing_error = sum(y_pred_test.flatten() != testY)/len(testY)
    return training_error, testing_error

adaboost_cls = adaBoostDS(dataArr = df3_train_X, classLabels=df3_train_Y, numIter = 1000)
ErrorCalculation(trainX = df3_train_X, trainY = df3_train_Y, testX = df3_test_X, testY = df3_test_Y)[1]

# iter = 30, test error = 0.29239999999999999
# iter = 40, test error = 0.2482
# iter = 70, test error = 0.2213
# iter = 300, test error = 0.1492
# iter = 500, test error = 0.13320000000000001 
# iter = 1000, test error = 0.1202
# iter = 1500, test error = 0.10879999999999999
# iter = 2500, test error = 0.1043
# iter = 3000, test error = 0.1023
# iter = 3500, test error = 0.1019
# iter = 4500, test error = 0.1013
# iter = 6000, test error = 0.10730000000000001


results_train_E = []
results_test_E = []
for i in range(40, 300, 5):
    adaboost_cls = adaBoostDS(dataArr = df3_train_X, classLabels=df3_train_Y, numIter = i)
    results_train_E.append(ErrorCalculation(trainX = df3_train_X, trainY = df3_train_Y, testX = df3_test_X, testY = df3_test_Y)[0])
    results_test_E.append(ErrorCalculation(trainX = df3_train_X, trainY = df3_train_Y, testX = df3_test_X, testY = df3_test_Y)[1])

# for n = 4, epoch = 50:900, lr = 0.01, l2_reg = 0.000001
import matplotlib.pyplot as plt
plt.plot(range(40, 300, 5), results_train_E, 'r--')
plt.axis([40, 300, 0.5, 0.53])

# testing error curve
plt.plot(range(40, 300, 5), results_test_E, 'r--', color='green')
plt.axis([40, 300, 0.5, 0.53])
plt.legend(['Training Error', 'Testing Error'], loc='upper left')
plt.title('Training & Testing Error vs. Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Error Rate')
plt.show()

for i in range(1,7000, 800):
    i = i +1
    print(i)
