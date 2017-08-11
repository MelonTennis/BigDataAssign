# decision tree in python
# By Yijia Jin
import os
import math
from collections import Counter
import numpy as np

import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

# read input data as X(float) and Y(str)
def ReadingInput(filepath, num):
    reader = csv.reader(open(filepath, "rb"))
    x = [r for r in reader]
    # train
    if num == 1:
        X = [y[:-1] for y in x]
        Y = [y[-1] for y in x]
        X = [[float(i) for i in x] for x in X]
        return [X, Y]
    # test
    elif num == 0:
        X = x
        X = [[float(i) for i in x] for x in X]
        return X

# Node class for decision Tree
class TreeNode:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        self.question = list()
        self.entropy = None

# calculate entropy
def CalEntrpopy(Y):
    res = list(set(Y))
    label_num = len(res)
    c = Counter(Y)
    cnt = 0.0
    for r in res:
        p = float(c[r])/len(Y)
        cnt = cnt - math.log(p)*p/math.log(2)
    return cnt


# Separate data by each attribute and finding separate point
def SeparateData(data):
    X = data[0]
    Y = data[1]
    # each attribute each separate point
    label = list(set(Y))
    # all labels same - no need to separate
    if len(label) == 1:
        return None
    # for each attribute
    minEntropy = sys.maxint
    minThre = None
    attribute = -1
    for i in len(X[0]):
        # for each value
        # data for ith attribute
        Xattri = [x[i] for x in X].sort()
        for j in range(0, len(X)):
            thre = (Xattri[j] + Xattri[j+1])/2
            tempEnt = CalEntrpopy(Y[0:j+1]) + CalEntrpopy(Y[j+1:])
            if minEntropy > tempEnt:
                minEntropy = tempEnt
                minThre = thre
                attribute = i
    leftX = [x for x in X if x[attribute] < minThre]
    leftY = [Y[i] for i in range(0, len(Y)) if X[i][attribute] < minThre]
    dataL = [leftX, leftY]
    rightX = [x for x in X if x[attribute] > minThre]
    rightY = [Y[i] for i in range(0, len(Y)) if X[i][attribute] > minThre]
    dataR = [rightX, rightY]
    return (dataL, dataR, minEntropy, minThre, attribute)

# separate data by value
def divideData(data, minThre, attribute):
    X = data[0]
    Y = data[1]
    leftX = [x for x in X if x[attribute] <= minThre]
    leftY = [Y[i] for i in range(0, len(Y)) if X[i][attribute] <= minThre]
    dataL = [leftX, leftY]
    rightX = [x for x in X if x[attribute] > minThre]
    rightY = [Y[i] for i in range(0, len(Y)) if X[i][attribute] > minThre]
    dataR = [rightX, rightY]
    #assert(len(rightX) + len(leftX) == len(X))
    return (dataL, dataR)

# building decision tree
def buildingTree(data, node):
    X = data[0]
    Y = data[1]
    if len(X) == 0 or len(X[0]) == 0:
        return node
    # each attribute each separate point
    label = list(set(Y))
    # all labels same - no need to separate
    if len(label) == 1:
        return node
    # for each attribute
    minEntropy = sys.maxint
    minThre = None
    attribute = -1
    entset = (0.0, 0.0)
    for i in range(0, len(X[0])):
        # for each value
        # data for ith attribute
        Xattri = sorted([x[i] for x in X])
        for j in range(0, len(X)-1):
            thre = (Xattri[j] + Xattri[j + 1]) / 2
            tempEnt1 = CalEntrpopy([Y[ii] for ii in range(0, len(Y)) if X[ii][i] > thre])
            tempEnt2 = CalEntrpopy([Y[iii] for iii in range(0, len(Y)) if X[iii][i] < thre])
            #print tempEnt1, tempEnt2
            tempEnt = tempEnt1 + tempEnt2
            entset = (tempEnt1, tempEnt2)
            #print i ,j, tempEnt
            if minEntropy > tempEnt:
                minEntropy = tempEnt
                minThre = thre
                attribute = i
    (dataL, dataR) = divideData(data, minThre, attribute)
    if(len(dataL[0]) == 0 or len(dataR[0]) == 0):
        node.left = None
        node.right = None
        node.entropy = node.entropy - minEntropy
        #node.question = [attribute, minThre]
    else:
        node.question = [attribute, minThre]
        node.entropy = node.entropy - minEntropy
        if len(dataL) > 0 and len(dataR) > 0:
            node.left = TreeNode(dataL)
            node.left.entropy = node.entropy - entset[1]
            node.right = TreeNode(dataR)
            node.right.entropy = node.entropy - entset[0]
            node.left = buildingTree(dataL, node.left)
            node.right = buildingTree(dataR, node.right)
    #strTree = printTree(node)
   # print strTree
    return node

# using decision tree and observe data to classify, for each data point
def classify(observe , tree):
    if(tree.left == None and tree.right == None):
        c = Counter(tree.data[1])
        label = set(tree.data[1])
        if(len(label) == 1):
            return tree.data[1][0]
        else:
            max = 0
            res = ""
            for l in label:
                if c[l] > max:
                    max = c[l]
                    res = l
            return l
    else:
        v = observe[tree.question[0]]
        branch = None
        if v >= tree.question[1]:
            branch = tree.right
        else:
            branch = tree.left
        return classify(observe, branch)

# print a tree
def printTree(node):
    if node == None:
        return ""
    string = ""
    if len(node.question) != 0:
        string = "attriute_" + str(node.question[0]) + " " + "threshold:" + str(node.question[1]) + " " + "entropy:"+ str(node.entropy)
    else:
        string = str(len(node.data[0])) + "+" + " " + "entropy:" + str(node.entropy)
    if node.left != None and node.right != None:
        string = string +" "+ str(len(node.left.data[0])) + "- " + str(len(node.right.data[0])) + "+"
    stringL = printTree(node.left)
    stringR = printTree(node.right)
    return string + "\n" + "|" + stringL + "\n" + "|" + stringR

# print a node
def printNode(node):
    name = ["sepal length", "sepal width", "petal length", "petal width"]
    if node == None:
        return ""
    string = ""
    if len(node.question) != 0:
        string = name[node.question[0]] + " " + "threshold:" + str(node.question[1]) #+ " " + "entropy:"+ str(node.entropy)
    else:
        string = "label: "+ node.data[1][0]+ " number:" +str(len(node.data[0])) #+ " " + "entropy:" + str(node.entropy)
    if node.left != None and node.right != None:
        string = string + " " + str(len(node.left.data[0])) + "- " + str(len(node.right.data[0])) + "+ " + "entropy:" + str(node.left.entropy) + str(node.right.entropy)
    return string

# print a lot of nodes
def printNodes(root, n):
    string = ""
    if(root == None):
        return string
    string = string + printNode(root)
    if(root.left != None):
        string = string + "\n" + "|"*n + printNodes(root.left, n+1)
    if(root.right != None):
        string = string + "\n" + "|"*n + printNodes(root.right, n+1)
    return string

# calculate error
def calAccuracy(label, pred):
    dif = [label[i] for i in range(0, len(label)) if label[i] == pred[i]]
    return float(len(dif))/len(label)

# confusion matrix
def confusionMatrix(pred, label):
    kinds = list(set(label))
    print kinds
    matrix = [[0 for i in range(0, len(kinds))] for j in range(0, len(kinds))]
    for k in range(0, len(pred)):
        matrix[kinds.index(label[k])][kinds.index(pred[k])] += 1
    return matrix

# cross validation
def crossValidation(dataTrain, dataTest):
    lenData = len(dataTrain[0])
    each = lenData/9
    datachunks = [dataTrain[0][each*i:each*(i+1)] for i in range(len(dataTrain[0])/each + 1)]
    labelchunks =  [dataTrain[1][each*i:each*(i+1)] for i in range(len(dataTrain[1])/each + 1)]
    bestTree = None
    bestAccuracy = None
    for N in range(0, 10):
        trainSet = []
        trainLabel = []
        for j in range(0, 10):
            if j != N:
                trainSet = trainSet + datachunks[j]
                trainLabel = trainLabel + labelchunks[j]
        testSet = datachunks[N]
        testLable = labelchunks[N]
        root = TreeNode(trainSet)
        rootEnt = CalEntrpopy(trainLabel)
        root.entropy = rootEnt
        root.data = [trainSet, trainLabel]
        root = buildingTree([trainSet, trainLabel], root)
        testPred = []
        for t in testSet:
            testPred.append(classify(t, root))
        accuracy = calAccuracy(testLable, testPred)
        #print accuracy
        if accuracy > bestAccuracy:
            bestTree = root
            bestAccuracy = accuracy
    #print bestAccuracy
    print printNodes(bestTree, 1)
    return bestTree

def main():
    dataTrain = ReadingInput("train.csv", 1)
    dataTest = ReadingInput("test.csv", 1)
    #crossValidation(dataTrain, dataTest)
    root = TreeNode(dataTrain)
    rootEnt = CalEntrpopy(dataTrain[1])
    #root.entropy = rootEnt
    #root.data = dataTrain
    #root = buildingTree(dataTrain, root)
    root = crossValidation(dataTrain, dataTest)
    #print printNodes(root, 1)
    testPred = []
    for t in dataTest[0]:
        testPred.append(classify(t, root))
    print testPred
    assert(len(testPred) == len(dataTest[1]))
    accuracy = calAccuracy(dataTest[1], testPred)
    print accuracy
    #print len(dataTest[1])
    print confusionMatrix(testPred, dataTest[1])


main()






