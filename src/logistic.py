#!/usr/bin/env python

import numpy as np
import pandas as pd

exponentBound = 709

def invertible(A):
    import sys
    (row, col) = A.shape
    if (row != col):
        print "Only square matrix is accepted. "
        sys.exit(-1)
    inverse = np.linalg.inv(A)
    identity = np.diag(np.ones(row))
    eps = 1.0e-2
    error = np.linalg.norm(inverse.dot(A) - identity)
    return error < eps

def toMatrix(x):
    A = np.zeros((1 + len(x), 1 + len(x)))
    A[0, 0] = 1
    a = np.zeros((len(x), 1))
    a[:, 0] = x[:]
    A[0, 1:] = x[:]
    A[1:, 0] = x[:]
    A[1:, 1:] = a.dot(np.transpose(a))
    return A

def Loss(x, y, theta0, theta):
    assert(len(x) == len(y))
    assert(len(x[0, :]) == len(theta))
    numberOfSamples = len(x)
    result = 0.0
    for row in range(len(x)):
        w = theta0 + theta.dot(x[row, :])
        if (abs(w) <= exponentBound):
            result += y[row]*np.log(1 + np.exp(w)) + (1 - y[row])*np.log(1 + np.exp(-w))
        else:
            if (w > exponentBound):
                result += y[row]*w
            else:
                result += (1 - y[row])*(-w)
    return result

def gradient(x, y, theta0, theta):
    assert(len(x) == len(y))
    assert(len(x[0, :]) == len(theta))
    g0 = 0
    for row in range(len(y)):
        w = theta0 + theta.dot(x[row, :])
        if (abs(w) <= exponentBound):
            g0 += y[row]*(1.0)/(1.0 + np.exp(-w)) + (1 - y[row])*(-1)/(1 + np.exp(w))
        else:
            if (w > exponentBound):
                g0 += y[row]
            else:
                g0 += (1 - y[row])*(-1)
    g = np.zeros(len(theta))
    for row in range(len(y)):
        w = theta0 + theta.dot(x[row, :])
        if (abs(w) <= exponentBound):
            g = g + y[row]/(1 + np.exp(-w))*x[row, :] + (1 - y[row])/(1 + np.exp(w))*(-x[row, :])
        else:
            if (w > exponentBound):
                g = g + y[row]*x[row, :]
            else:
                g = g + (1 - y[row])*(-x[row, :])
    g = list(g)
    g.insert(0, g0)
    g = np.asarray(g)
    return g

def toString(Theta):
    result = "theta0 = "
    result += str(Theta[0]) + "\n"
    result += "theta = ["
    for i in range(1, len(Theta)):
        if (i < len(Theta) - 1):
            result += str(Theta[i]) + ", "
        else:
            result += str(Theta[i])
    result += "]\n"
    return result

def gradientDescent(x, y, theta0, theta, rate, iprint = False):
    assert(len(x) == len(y))
    assert(len(x[0, :]) == len(theta))
    Theta = list(theta)
    Theta.insert(0, theta0)
    Theta = np.asarray(Theta)
    Theta_old = Theta
    iterationMax = 10000
    eps = 1.0e-8
    counter = 0
    ofile = open("Theta_records.txt", "w")
    while(True):
        counter += 1
        if (counter > iterationMax):
            break
        trueRate = rate/np.log(counter+1)
        g = gradient(x, y, Theta_old[0], Theta_old[1:])
        Theta = Theta_old - trueRate*g
        ofile.write("counter = " + str(counter) + "\n" + toString(Theta) + "\n")
        diff = Theta - Theta_old
        error = np.linalg.norm(diff) 
        if (iprint):
            print "Counter = ", counter, ", error = ", error, ", norm of gradient = ", np.linalg.norm(g)
        Theta_old = Theta
        if (error < eps):
            break
    ofile.close()
    return Theta

def Hessian(x, y, theta0, theta):
    assert(len(x) == len(y))
    assert(len(x[0, :]) == len(theta))
    dimension = len(theta) + 1
    H = np.zeros((dimension, dimension))
    for i in range(len(y)):
        w = theta0 + theta.dot(x[i, :])
        if (abs(w) <= exponentBound):
            H = H + (1.0/(2*np.cosh(0.5*w))**2)*toMatrix(x[i, :])
        else:
            pass 
    return H

def sigmoid(x, theta0, theta):
    assert(len(x) == len(theta))
    w = theta0 + theta.dot(x)
    result = 0
    if (abs(w) <= exponentBound):
        result = 1.0/(1 + np.exp(w))
    else:
        if (w > exponentBound):
            result = 0.0
        else:
            result = 1.0
    return result

def Newton(x, y, theta0, theta, iprint = False):
    import sys
    assert(len(x) == len(y))
    assert(len(x[0, :]) == len(theta))
    dimension = len(theta) + 1
    Theta = list(theta)
    Theta.insert(0, theta0)
    Theta = np.asarray(Theta)
    Theta_old = Theta
    iterationMax = 40
    eps = 1.0e-8
    counter = 0
    ofile = open("Theta_records.txt", "w")
    while(True):
        counter += 1
        if (counter > iterationMax):
            break
        H = Hessian(x, y, Theta_old[0], Theta_old[1:])
        if (not invertible(H)):
            print "Hessian matrix is singular. exiting. "
            sys.exit(-1)
        g = gradient(x, y, Theta_old[0], Theta_old[1:])
        Theta = Theta_old - np.linalg.inv(H).dot(g)
        ofile.write("counter = " + str(counter) + "\n" + toString(Theta) + "\n")
        error = np.linalg.norm(Theta - Theta_old)
        Theta_old = Theta
        if (iprint):
            print "Counter = ", counter, ", error = ", error, ", norm of gradient = ", np.linalg.norm(g)
        if (error < eps):
            break
    ofile.close()
    return Theta

def extractFromDataFrame(df):
    (rowNumber, columnNumber) = df.shape
    y = df.values[:, columnNumber-1] # label
    x = df.values[:, 0:-1]
    return x, y

def printFile(x, y, outputFileName):
    assert(len(x) == len(y))
    ofile = open(outputFileName, "w")
    for i in range(len(x)):
        ofile.write(str(x[i]) + ", " + str(y[i]) + "\n")
    ofile.close()

def train(df, useInitialTheta = False, useNewton = True):
    import random
    (row, col) = df.shape
    x = df.values[:, 0:-1]
    y = df.values[:, -1]
    if (useInitialTheta):
        theta0 = 0
        theta = []
        theta = np.asarray(theta)
    else:
        bound = 0.3
        theta0 = random.uniform(-bound, bound)
        theta = np.zeros(col - 1)
        for i in range(len(theta)):
            theta[i] = random.uniform(-bound, bound)
    if (useNewton):
        Theta = Newton(x, y, theta0, theta, True)
    else:
        rate = 0.01
        Theta = gradientDescent(x, y, theta0, theta, True)
    return Theta

def predict(testData, Theta):
    (row, col) = testData.shape
    theta0 = Theta[0]
    theta = Theta[1:]
    x = testData.values[:, 0:-1]
    label = testData.values[:, -1]
    prediction = []
    for i in range(len(x)):
        prediction.append(sigmoid(x[i], theta0, theta))
    printFile(prediction, label, "prediction-label.txt")

def crossValidation(df, trainRatio, useInitialTheta, useNewton):
    assert(trainRatio < 1 and trainRatio > 0)
    import random
    (row, col) = df.shape
    x = df.values[:, 0:-1]
    y = df.values[:, -1]
    trainNumber = int(row*trainRatio)
    x_train = x[0:trainNumber]
    y_train = y[0:trainNumber]
    if (not useInitialTheta):
        bound = 1.2
        theta0 = random.uniform(-bound, bound)
        theta = np.zeros(col-1)
        for i in range(len(theta)):
            theta[i] = random.uniform(-bound, bound)
    else:
        theta0 = -2.17015296562
        theta = [1.53616673727, 0.111167244918, 0.640863444623, 5.41600748124, 2.18457203328, 14.1195884269, 9.58510914492, 9.80738132842, -10.4598737079, 5.78112749192] 
        theta = np.asarray(theta)
    if (useNewton):
        Theta = Newton(x_train, y_train, theta0, theta, True)
    else:
        rate = 0.1
        Theta = gradientDescent(x_train, y_train, theta0, theta, rate, True)
    theta0 = Theta[0]
    theta = Theta[1:]
    prediction = []
    trueLabel = []
    for i in range(trainNumber, row):
        prediction.append(sigmoid(x[i], theta0, theta))
        trueLabel.append(str(y[i]))
    printFile(prediction, trueLabel, "prediction-true-label.txt")
    ofile = open("Theta.txt", "w")
    ofile.write(str(Theta) + "\n")
    ofile.close()
    return prediction, trueLabel


'''
    Confusion matrix:
                        Actual N  |   Actual P
     Prediction N         TN      |      FN
     ------------------------------------------------
     Prediction P         FP      |     TP
'''
def generateConfusionMatrix(prediction, trueLabel, threshold):
    assert(len(prediction) == len(trueLabel))
    labelSet = set()
    for i in range(len(trueLabel)):
        labelSet.add(trueLabel[i])
    negativeLabel = "0.0"
    positiveLabel = "1.0"
    assert(negativeLabel in labelSet)
    assert(positiveLabel in labelSet)
    assert(len(labelSet) == 2)
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    result = np.zeros((2,2))
    for i in range(len(prediction)):
        if (prediction[i] < threshold): # prediction is negative
            if (trueLabel[i] == negativeLabel):
                trueNegative += 1
            else:
                falseNegative += 1
        else: # positive prediction
            if (trueLabel[i] == negativeLabel):
                falsePositive += 1
            else:
                truePositive += 1
    result[0, 0] = trueNegative
    result[0, 1] = falseNegative
    result[1, 0] = falsePositive
    result[1, 1] = truePositive
    return result

def getPrecision(confusionMatrix):
    (row, col) = confusionMatrix.shape
    assert(row == 2 and col == 2)
    truePositive = confusionMatrix[1,1]
    falsePositive = confusionMatrix[1,0]
    eps = 1.0e-10
    return float(truePositive + eps)/(truePositive + falsePositive + eps)

def getRecall(confusionMatrix):
    (row, col) = confusionMatrix.shape
    assert(row == 2 and col == 2)
    truePositive = confusionMatrix[1,1]
    falseNegative = confusionMatrix[0,1]
    eps = 1.0e-10
    return float(truePositive + eps)/float(truePositive + falseNegative + eps)

def getFPR(confusionMatrix): # get the fasle positive rate
    (row, col) = confusionMatrix.shape
    assert(row == 2 and col == 2)
    trueNegative = confusionMatrix[0,0]
    falsePositive = confusionMatrix[1,0]
    eps = 1.0e-10
    return float(falsePositive + eps)/float(falsePositive + trueNegative + eps)

def harmonicMean(a, b):
    return 2*a*b/(a + b)

def getROCAndPR(prediction, trueLabel):
    precision = []
    recall = []
    FPR = []
    F1 = []
    lower = 0.0
    upper = 1.0
    thresholdValues = []
    numberOfIntervals = 40
    delta = (upper - lower)/float(numberOfIntervals)
    for i in range(numberOfIntervals+1):
        thresholdValues.append(lower + delta*i)

    ofile = open("confusion-matrices.txt", "w")
    for threshold in thresholdValues:
        confusionMatrix = generateConfusionMatrix(prediction, trueLabel, threshold)
        ofile.write("threshold = " + str(threshold) + "\n")
        ofile.write(str(confusionMatrix) + "\n")
        precision.append(getPrecision(confusionMatrix))
        recall.append(getRecall(confusionMatrix))
        FPR.append(getFPR(confusionMatrix))
    ofile.close()
    printFile(FPR, recall, "ROC.txt")
    printFile(recall, precision, "PR.txt")
    for i in range(len(precision)):
        F1.append(harmonicMean(precision[i], recall[i]))
    printFile(thresholdValues, F1, "F1_theta.txt")

def main():
    import os
    import sys
    import random

    if (len(sys.argv) != 2):
        print "inputFileName = sys.argv[1]. "
        return -1

    inputFileName = sys.argv[1]
    print "Beginning to read in the file ... "
    df = pd.read_csv(inputFileName)
    print "File reading finished. "
    trainRatio = 0.5
    useInitialTheta = False
    useNewton = True
    prediction, trueLabel = crossValidation(df, trainRatio, useInitialTheta, useNewton)
    getROCAndPR(prediction, trueLabel)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
