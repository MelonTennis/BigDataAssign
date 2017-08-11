from mrjob.job import MRJob
import math
from collections import Counter
import csv
import sys
import random

def ReadingID(filepath):
    reader = csv.reader(open(filepath, "rb"))
    x = [r for r in reader]
    x = x[1:]
    X = [y[0] for y in x]
    return X

# pre-process data
def preprocess(data):
    X = data[0]
    Y = data[1]
    for i in range(0, len(X)):
        if X[i][3] == "male":
            X[i][3] = 0
        else:
            X[i][3] = 1
        if X[i][4] == "":
            X[i][4] = 29.7
        if X[i][8] == "":
            X[i][8] = 32.2
        if len(X[i]) == 10:
            X[i] = X[i] + ["S"]
        if X[i][10] == "C":
            X[i][10] = 0
        elif X[i][10] == "Q":
            X[i][10] = 1
        else:
            X[i][10] = 2
        # 0 - id, 1 - pclass, 3 - sex, 4 - age, 5 - sibsp, 6 - parch, 8 - fare, 10 - embarked
        X[i] = [float(X[i][1]), float(X[i][3]), float(X[i][4]), float(X[i][5]), float(X[i][6]), float(X[i][8])]
        # assert(len(X[i]) == 7)
    return [X, Y]

def featureEngineer(data):
    #0-PassengerId, 1-Pclass, 2-Name, 3-Sex, 4-Age, 5-SibSp, 6-Parch, 7-Ticket, 8-Fare, 9-Cabin, 10-Embarked
    X = data[0]
    Y = data[1]
    for i in range(0, len(X)):
        if X[i][3] == "male":
            X[i][3] = 0
        else:
            X[i][3] = 1
        if X[i][4] == "":
            X[i][4] = 29.7
        if X[i][8] == "":
            X[i][8] = 32.2
        if len(X[i]) == 10:
            X[i] = X[i] + ["S"]
        if X[i][10] == "C":
            X[i][10] = 0
        elif X[i][10] == "Q":
            X[i][10] = 1
        else:
            X[i][10] = 2
        familysize = float(X[i][6]) + float(X[i][5]) + 1
        if familysize > 1:
            alone = 0
        else:
            alone = 1
        prefix = X[i][2].split(", ")[1].split(".")[0]
        if prefix == "Master":
            title = 0
        elif prefix == "Miss":
            title = 1
        elif prefix == "Mr":
            title = 2
        elif prefix == "Mrs":
            title = 3
        else:
            title = 4
        X[i] = X[i] + [familysize, alone] # 11, 12
        X[i] = X[i] + [title] # 13
        ticketlen = len(X[i][7]) + len(X[i][7].split(".")) + len(X[i][7].split(" "))
        X[i] = X[i] + [ticketlen] # 14
        # 0-PassengerId, 1-Pclass, 2-Name, 3-Sex, 4-Age, 5-SibSp, 6-Parch, 7-Ticket, 8-Fare, 9-Cabin, 10-Embarked, 11-familysize, 12-alone, 13-predix, 14-tickenlen
        X[i] = [float(X[i][1]), float(X[i][3]), float(X[i][4]), float(X[i][8]), float(X[i][12]), float(X[i][13]), float(X[i][14])]
    return [X, Y]


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
    # res = list(set(Y))
    res = []
    for y in Y:
        if y not in res:
            res.append(y)
    c = Counter(Y)
    cnt = 0.0
    for r in res:
        p = float(c[r]) / len(Y)
        cnt = cnt - math.log(p) * p / math.log(2)
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
    # assert(len(rightX) + len(leftX) == len(X))
    return (dataL, dataR)


# print a node
def printNode(node):
    name = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", ""]
    if node == None:
        return ""
    string = ""
    if len(node.question) != 0:
        string = name[node.question[0]] + " " + "threshold:" + str(
            node.question[1])  # + " " + "entropy:"+ str(node.entropy)
    else:
        string = "label: " + str(node.data[1][0]) + " number:" + str(
            len(node.data[0]))  # + " " + "entropy:" + str(node.entropy)
    if node.left != None and node.right != None:
        string = string + " " + str(len(node.left.data[0])) + "- " + str(
            len(node.right.data[0])) + "+ " + "entropy:" + str(node.left.entropy + node.right.entropy)
    c = Counter(node.data[1])
    string = string + " nodes: " + str(c)[9:-2]
    return string


# print a lot of nodes
def printNodes(root, n):
    string = ""
    if (root == None):
        return string
    string = string + printNode(root)
    if (root.left != None):
        string = string + "\n" + "|" * n + printNodes(root.left, n + 1)
    if (root.right != None):
        string = string + "\n" + "|" * n + printNodes(root.right, n + 1)
    return string


# building decision tree
def buildingTree(data, node, k):
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
    randk = random.sample(range(0, len(X[0])),  k)
    for i in randk:
        # for each value
        # data for ith attribute
        Xattri = sorted(list(set([x[i] for x in X])))
        # Xattri = [x[i] for x in X]
        for j in range(0, len(Xattri) - 1):
            #print Xattri
            thre = (Xattri[j] + Xattri[j + 1]) / 2
            # thre = sum(Xattri) / float(len(Xattri))
            tempEnt1 = CalEntrpopy([Y[ii] for ii in range(0, len(Y)) if X[ii][i] > thre])
            tempEnt2 = CalEntrpopy([Y[iii] for iii in range(0, len(Y)) if X[iii][i] < thre])
            # print tempEnt1, tempEnt2
            tempEnt = tempEnt1 + tempEnt2
            entset = (tempEnt1, tempEnt2)
            # print i ,j, tempEnt
            if minEntropy > tempEnt:
                minEntropy = tempEnt
                minThre = thre
                attribute = i
    (dataL, dataR) = divideData(data, minThre, attribute)
    if (len(dataL[0]) == 0 or len(dataR[0]) == 0):
        node.left = None
        node.right = None
        node.entropy = node.entropy - minEntropy
        # node.question = [attribute, minThre]
    else:
        node.question = [attribute, minThre]
        node.entropy = node.entropy - minEntropy
        if len(dataL) > 0 and len(dataR) > 0:
            # print len(dataL[0]), len(dataR[0])
            node.left = TreeNode(dataL)
            node.left.entropy = node.entropy - entset[1]
            node.right = TreeNode(dataR)
            node.right.entropy = node.entropy - entset[0]
            node.left = buildingTree(dataL, node.left, k)
            node.right = buildingTree(dataR, node.right, k)
            # strTree = printTree(node)
            # print strTree
    return node


# using decision tree and observe data to classify, for each data point
def classify(observe, tree):
    if (tree.left == None and tree.right == None):
        c = Counter(tree.data[1])
        label = set(tree.data[1])
        if (len(label) == 1):
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
    randx = random.sample(range(0, len(X)), int(round(len(X) * 0.67)))
    chooseX = [X[j] for j in randx]
    chooseY = [Y[j] for j in randx]
    return [chooseX, chooseY]


# get prediction from combination, vote the most
def getPredict(combine):
    pred = [[x[i] for x in combine] for i in range(0, len(combine[0]))]
    res = []
    for m in range(0, len(pred)):
        res.append(Counter(pred[m]).most_common(1)[0][0])
    return res


# N - number of trees, k - number of features, data = [X, Y]
def RandomForest(N, k, dataTrain, dataTest):
    # f1 = open('./RFtree_Titanic.txt', 'w+')
    # f1.write("Random forest tree grams\n")
    combine_Pred = []
    for iter in range(0, N):
        train = randomChoose(dataTrain, k)
        root = TreeNode(train)
        root.entropy = CalEntrpopy(train[1])
        root = buildingTree(train, root, k)
        # f1.write("Tree" + str(iter) + "\n")
        # f1.write(printNodes(root, 1))
        # f1.write("\n\n")
        test_Pred = []
        for t in dataTest[0]:
            test_Pred.append(classify(t, root))
        combine_Pred.append(test_Pred)
    predict = getPredict(combine_Pred)
    return dataTest, predict

class RF(MRJob):

    def mapper(self, _, input):
        data = csv.reader(open(input, "rb"))
        dataT = csv.reader(open("/home/yijiaj/test_new.csv", "rb"))
        x = [r for r in data]
        x = x[1:]
        x_test = [r for r in dataT]
        x_test = x_test[1:]
        # train
        X1 = [y[0:1] + y[2:-1] for y in x]
        Y1 = [int(y[1]) for y in x]
        train = [X1, Y1]
        # test
        X2 = [y[0:-1] for y in x_test]
        Y2 = [int(y[-1]) for y in x_test]
        test = [X2, Y2]
        N = 1
        k = 6
        data_train = featureEngineer(train)
        data_test = featureEngineer(test)
        data_test, predict = RandomForest(N, k, data_train, data_test)
        for i in range(0, len(predict)):
            yield test[0][i], predict[i]

    def reducer(self, key, values):
        p = Counter(values).most_common(1)[0][0]
        yield key, p

if __name__ == '__main__':
    RF.run()
