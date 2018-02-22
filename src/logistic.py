#!/usr/bin/env python

import numpy as np
import pandas as pd

def invertible(A):
    import sys
    (row, col) = A.shape
    if (row != col):
        print "Only square matrix is accepted. "
        sys.exit(-1)
    inverse = np.linalg.inv(A)
    identity = np.diag(np.ones(row))
    eps = 1.0e-8
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
        result += y[row]*np.log(1 + np.exp(theta0 + theta.dot(x[row, :]))) + (1 - y[row])*np.log(1 + np.exp(-theta0 - theta.dot(x[row, :])))
    return result

def gradient(x, y, theta0, theta):
    assert(len(x) == len(y))
    assert(len(x[0, :]) == len(theta))
    g0 = 0
    for row in range(len(y)):
        g0 += y[row]*(1.0)/(1.0 + np.exp(-theta0 - theta.dot(x[row, :]))) + (1 - y[row])*(-1)/(1 + np.exp(theta0 + theta.dot(x[row, :])))
    g = np.zeros(len(theta))
    for row in range(len(y)):
        g = g + y[row]/(1 + np.exp(-theta0 - theta.dot(x[row, :])))*x[row, :] + (1 - y[row])/(1 + np.exp(theta0 + theta.dot(x[row, :])))*(-x[row, :])
    g = list(g)
    g.insert(0, g0)
    g = np.asarray(g)
    return g

def gradientDescent(x, y, theta0, theta, rate, iprint = False):
    assert(len(x) == len(y))
    assert(len(x[0, :]) == len(theta))
    Theta = list(theta)
    Theta.insert(0, theta0)
    Theta = np.asarray(Theta)
    Theta_old = Theta
    iterationMax = 3000
    eps = 1.0e-6
    counter = 0
    while(True):
        counter += 1
        if (counter > iterationMax):
            break
        Theta = Theta_old - rate*gradient(x, y, Theta_old[0], Theta_old[1:])
        diff = Theta - Theta_old
        error = np.sqrt(np.dot(diff, diff))
        if (iprint):
            print "Counter = ", counter, ", error = ", error
        Theta_old = Theta
        if (error < eps):
            break
    return Theta

def Hessian(x, y, theta0, theta):
    assert(len(x) == len(y))
    assert(len(x[0, :]) == len(theta))
    dimension = len(theta) + 1
    H = np.zeros((dimension, dimension))
    for i in range(len(y)):
        H = H + (1.0/(2*np.cosh(0.5*(theta0 + theta.dot(x[i, :]))))**2)*toMatrix(x[i, :])
    return H

def sigmoid(x, theta0, theta):
    assert(len(x) == len(theta))
    return 1.0/(1 + np.exp(theta0 + theta.dot(x)))

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
        error = np.linalg.norm(Theta - Theta_old)
        Theta_old = Theta
        if (iprint):
            print "Counter = ", counter, ", error = ", error
        if (error < eps):
            break
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

def crossValidation(df, trainRatio):
    assert(trainRatio < 1 and trainRatio > 0)
    import random
    (row, col) = df.shape
    x = df.values[:, 0:-1]
    y = df.values[:, -1]
    trainNumber = int(row*trainRatio)
    x_train = x[0:trainNumber]
    y_train = y[0:trainNumber]
    theta0 = random.uniform(-1, 1)
    theta = np.zeros(col-1)
    for i in range(len(theta)):
        theta[i] = random.uniform(-1, 1)
    Theta = Newton(x_train, y_train, theta0, theta, True)
    theta0 = Theta[0]
    theta = Theta[1:]
    prediction = []
    trueLabel = []
    for i in range(trainNumber, row):
        prediction.append(sigmoid(x[i], theta0, theta))
        trueLabel.append(str(y[i]))
    printFile(prediction, trueLabel, "logistic_result.txt")
    return prediction, trueLabel

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
    prediction, trueLabel = crossValidation(df, trainRatio)
    getROCAndPR(prediction, trueLabel)
    return 0

def main_test():
    import random
    dimension = 6
    matrix = np.zeros((dimension+1, dimension+1))
    for i in range(2*dimension):
        print "i = ", i+1
        x = np.zeros(dimension)
        for j in range(len(x)):
            x[j] = random.uniform(0, 1)
        matrix = matrix + random.uniform(0, 1)*toMatrix(x)
        print invertible(matrix)

if __name__ == "__main__":
    import sys
    sys.exit(main())
