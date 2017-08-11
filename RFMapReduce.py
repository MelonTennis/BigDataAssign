from mrjob.job import MRJob
import math
from collections import Counter
import csv
import sys
import random

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
    #res = list(set(Y))
    res = []
    label_num = 0
    for y in Y:
        if y not in res:
            res.append(y)
            label_num = label_num + 1
    c = Counter(Y)
    cnt = 0.0
    for r in res:
        p = float(c[r])/len(Y)
        cnt = cnt - math.log(p)*p/math.log(2)
    return cnt

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
        Xattri = [x[i] for x in X]
        thre = sum(Xattri)/float(len(Xattri))
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

# calculate error
def calAccuracy(label, pred):
    dif = [label[i] for i in range(0, len(label)) if label[i] == pred[i]]
    return float(len(dif)) / len(label)

# Random choose k features from m features in train data
def randomChoose(data, k):
    X = data[0]
    Y = data[1]
    rand = random.sample(range(0, len(X[0])),  k)
    for x in X:
        x = list(x[i] for i in rand)
    return [X, Y]

# N - number of trees, k - number of features, data = [X, Y]
def RandomForest(N, k, data):
    # radom select 80% train and 20% test
    rand = random.sample(range(0, len(data[0])), int(round(len(data[0]) * 0.8)))
    unrand = list(set(range(0, len(data[0]))) - set(rand))
    dataTrain = [[data[0][i] for i in rand], [data[1][i] for i in rand]]
    dataTest = [[data[0][j] for j in unrand], [data[1][j] for j in unrand]]
    combine_Pred = []
    for iter in range(0, N):
        train = randomChoose(dataTrain, k)
        root = TreeNode(train)
        root.entropy = CalEntrpopy(train[1])
        root = buildingTree(data, root)
        test_Pred = []
        for t in dataTest[0]:
            test_Pred.append(classify(t, root))
        combine_Pred.append(test_Pred)
    predict = getPredict(combine_Pred)
    return dataTest, predict

# get prediction from combination, vote the most
def getPredict(combine):
    pred = [[x[i] for x in combine] for i in range(0, len(combine[0]))]
    res = []
    for m in range(0, len(pred)):
        res.append(Counter(pred[m]).most_common(1)[0][0])
    return res

class RF(MRJob):

    def mapper(self, _, input):
        data = csv.reader(open(input, "rb"))
        x = [r[-1].split(';') for r in data]
        X = [y[1:-1] for y in x]
        Y = [y[-1] for y in x]
        X = [[float(i) for i in x] for x in X]
        train = [X, Y]
        N = 1
        k = 4 # number of features
        test, predict = RandomForest(N, k, train)
        for i in range(0, len(predict)):
            yield test[0][i], predict[i]

    def reducer(self, key, values):
        p = Counter(values).most_common(1)[0][0]
        yield key, p


if __name__ == '__main__':
    RF.run()
