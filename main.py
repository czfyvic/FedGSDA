import time
from copy import deepcopy
import pyro

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, BlogCatalog, Flickr, Facebook, Gowalla
from dhg.models import GCNS, GCNT0, GCNT1, GIN, HyperGCN, GraphSAGE
from dhg.random import set_seed
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
from scipy import sparse
import random
import pandas as pd
import collections
import networkx as nx
from scipy.sparse import csr_matrix
import pickle
import math
import cmath
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import warnings
import matplotlib.pyplot as plt
import louvain.community as community_louvain
from sklearn import preprocessing, model_selection
from stellargraph.core.graph import StellarGraph
import stellargraph as sg
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore', category=FutureWarning)


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[1][idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
        return res
    else:
        res = evaluator.test(lbls, outs)
        predict = np.argmax(outs, axis=-1)
        recall0 = recall_score(lbls, predict, average='macro')
        precision0 = precision_score(lbls, predict, average='macro')
        f0 = f1_score(lbls, predict, average='macro')

        print("accuracy:", res["accuracy"], "f1:", f0, "recall:", recall0, "precision", precision0)
        return res, f0, recall0, precision0

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def collectingData(data):
    feature = data["features"].numpy()
    labels = data["labels"].numpy()
    edges = data["edge_list"]

    np.save("BlogCatalog/feature.npy", feature)
    np.save("BlogCatalog/edges.npy", edges)
    np.save("BlogCatalog/labels.npy", labels)

    return None

def getSamplingMatrix(samplingGlobalAdjWithReduceNode, sampling_idx_range):
    adjLen = len(samplingGlobalAdjWithReduceNode)
    samplingMatrix = np.zeros((adjLen, adjLen))

    for idx in sampling_idx_range:
        currentList = samplingGlobalAdjWithReduceNode[idx]
        for listIdx in currentList:
            samplingMatrix[idx, listIdx] = 1

    return samplingMatrix

def getSamplingAdj1(adjList, sampling_idx_range): #using for global adj and take the node out of sampling nodes
    newNodeAdjList = []
    for listIndex in adjList:
        withInFlag = listIndex in sampling_idx_range
        if withInFlag:
            newNodeAdjList.append(listIndex)
    return newNodeAdjList

def getSamplingAdj(adjList, sampling_idx_range):   #the index of sampling_idx_range is the current node index
    newNodeAdjList = []
    for listIndex in adjList:
        withInFlag = listIndex in sampling_idx_range
        if withInFlag:
            newNodeAdjList.append(sampling_idx_range.index(listIndex))
    return newNodeAdjList

def getSamplingGlobalAdj(graph, sampling_idx_range):  #pos the sampling nodes in global adj
    adjLen = len(graph)
    samplingGlobalAdj = collections.defaultdict(list)
    for idx in range(adjLen):
        withInFlag = idx in sampling_idx_range
        if withInFlag:
            currentList = graph[idx]
            newCurrentList = getSamplingAdj1(currentList, sampling_idx_range)
            samplingGlobalAdj[idx] = newCurrentList
        else:
            samplingGlobalAdj[idx] = []
    samplingMatrix = getSamplingMatrix(samplingGlobalAdj, sampling_idx_range)
    return samplingMatrix

def loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, classNum, saveName):
    saveName = saveName + '-TrainLabelIndex'
    trainsl = samplingTrainsetLabel.tolist()
    label = trainsl

    samplingTrainIndex111 = []
    # -----getting the train index from the file---------#
    # nd = np.genfromtxt(saveName, delimiter=',', skip_header=True)
    # samplingIndex = np.array(nd).astype(int)
    # for i in range(classNum):
    #     currentSamplingIndex = samplingIndex[:, i]
    #     samplingTrainIndex111 += currentSamplingIndex.tolist()

    # --------getting train index and save train index--------#
    samplingClassList = []
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(label) if x == k]
        samplingIndex = random.sample(range(0, len(labelIndex)), sampleNumEachClass[k])
        samplingFixedIndex = np.array(labelIndex)[samplingIndex].tolist()
        samplingClassList.append(samplingFixedIndex)
        samplingTrainIndex111 += samplingFixedIndex

    fileCol = {}
    for i in range(classNum):
        colName = 'TrainLabelIndex' + str(i)
        fileCol[colName] = samplingClassList[i]

    # dataframe = pd.DataFrame(fileCol)  # save the samplingIndex of every client
    # dataframe.to_csv(saveName, index=False, sep=',')  #

    return samplingTrainIndex111
#

#----------------------for federated average-----------------------#
def getSamplingIndex(nodesNum, samplingRate, testNodesNum, valNodesNum, trainLabel, labels):
    totalSampleNum = nodesNum - testNodesNum - valNodesNum  # get the train num according to the fixed testset and valset
    samplingNum = int(samplingRate * totalSampleNum)  # get
    testAndValIndex = [i for i in range(totalSampleNum, nodesNum)]

    #-----analysis the label distributation result of test and val------#
    classNum = 6
    classDict = {}
    classDictIndex = {}
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(trainLabel) if x == k]
        labelCount = len(labelIndex)
        classDict[k] = labelCount
        classDictIndex[k] = labelIndex

    lastSamplingNum = int(samplingNum / (classNum + 5))
    beforeSamplingNum = int((samplingNum - lastSamplingNum) / 5)
    finalSamplingNum = samplingNum - beforeSamplingNum * 5

    samplingNumList = [beforeSamplingNum] * 6
    samplingNumList[1] = finalSamplingNum

    samplingIndex = []
    for k in range(classNum):
        currentSamplingNum = samplingNumList[k]
        currentSamplingIdex = random.sample(classDictIndex[k], currentSamplingNum)
        samplingIndex += currentSamplingIdex

    return samplingIndex



def get_graph1(data):
    currentGraph = {}
    nodesNum = data["num_vertices"]
    for i in range(nodesNum):
        currentGraph[i] = []

    for edge in data["edge_list"]:
        currentNIdx = edge[0]
        currentGraph[currentNIdx].append(edge[1])

    graph = collections.defaultdict(list)
    for i in range(nodesNum):
        graph[i] = currentGraph[i]
    return graph

def get_graph(data):
    nodesNum = data["num_vertices"]
    graphList = []
    nodeIdx = 0
    row = []
    for edge in data["edge_list"]:
        currentNIdx = edge[0]
        if currentNIdx != nodeIdx:
            nodeIdx = currentNIdx
            graphList.append(row)
            row = []
        row.append(edge[1])
    graphList.append(row)
    graph = collections.defaultdict(list)
    for i in range(nodesNum):
        graph[i] = graphList[i]
    return graph

def get_edge_list(samplingAdj):
    nodesNum = len(samplingAdj)
    edge_list = []
    for i in range(nodesNum):
        rowNodes = samplingAdj[i]
        for j in rowNodes:
            node = tuple([i, j])
            edge_list.append(node)
    return edge_list


def getCommunitData(data, communities, node_subjects, clientNum):
    graph = get_graph1(data)

    dataList = []
    train_mask_list = []
    val_mask_list = []
    test_mask_list = []
    test_idx_list = []
    val_idx_list = []
    train_list = []
    for i in range(clientNum):
        communitie = list(communities[i])
        communitieLen = len(communitie)
        count = 0
        samplingAdj = collections.defaultdict(
            list)  # getting current sampling adj which is used for training is this client
        for index in communitie:
            currentList = graph[index]
            newCurrentList = getSamplingAdj(currentList, communitie)
            samplingAdj[count] = newCurrentList
            count += 1

        print('clientId:', {i})
        print(len(communitie))
        print(communitie)

        X = data["features"][communitie, :]
        edge_list = get_edge_list(samplingAdj)
        lbls = data["labels"][communitie]

        samplingAdjNum = len(samplingAdj)
        sub_node_subjects = node_subjects[communitie]

        trainIdx, testIdx = model_selection.train_test_split(
            sub_node_subjects, train_size=0.5, test_size=0.2, stratify=sub_node_subjects
        )
        trainIdx = list(trainIdx.index.values)
        testIdx = list(testIdx.index.values)
        valIdx = [idx for idx in communitie if idx not in trainIdx and idx not in testIdx]

        idx_train = []
        idx_test = []
        idx_val = []
        for idx, val in enumerate(communitie):
            if val in trainIdx:
                idx_train.append(idx)
            elif val in testIdx:
                idx_test.append(idx)
            else:
                idx_val.append(idx)

        train_mask = sample_mask(idx_train, samplingAdjNum)
        val_mask = sample_mask(idx_val, samplingAdjNum)
        test_mask = sample_mask(idx_test, samplingAdjNum)

        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        train_mask_list.append(train_mask)
        val_mask_list.append(val_mask)
        test_mask_list.append(test_mask)

        dataDict = {"num_classes": data["num_classes"],
                    "num_vertices": communitieLen,
                    "num_edges": len(edge_list),
                    "dim_features": X.shape[1],
                    "features": X,
                    "edge_list": edge_list,
                    "labels": lbls,
                    "sampling_idx_range": communitie}

        dataList.append(dataDict)
        train_list.append(trainIdx)
        test_idx_list.append(testIdx)
        val_idx_list.append(valIdx)

    return dataList, train_mask_list, val_mask_list, test_mask_list, test_idx_list, train_list, val_idx_list




def getClientData(data, samplingRate, sampleNumEachClass, classNum, saveName):
    X, lbl = data["features"], data["labels"]

    graph = get_graph1(data)

    nodesNums = data["num_vertices"]
    testNodesNum = int(nodesNums * 0.4)
    valNodesNum = int(nodesNums * 0.2)
    trainNum = nodesNums - testNodesNum - valNodesNum
    idx_train = range(0, trainNum)
    trainLabel = lbl[idx_train]

    samplingAdj, samplingNum, samplingTrainFixedIndex,\
    sampling_idx_range, samplingLabels, samplingMatrix = dataSplitting(nodesNums, testNodesNum, valNodesNum, graph,
                  trainLabel, lbl, samplingRate,
                  sampleNumEachClass, classNum, saveName)

    # samplingAdj = getTheCutSampling(samplingAdj, saveName)

    X = X[sampling_idx_range, :]
    edge_list = get_edge_list(samplingAdj)

    samplingAdjNum = len(samplingAdj)
    idx_test = range(samplingAdjNum - testNodesNum, samplingAdjNum)  # get the last 2800 indexes as test set
    idx_val = range(samplingAdjNum - testNodesNum - valNodesNum,
                    samplingAdjNum - testNodesNum)  # sampling 420 indexes as train set, each class has 60 labels
    idx_train = samplingTrainFixedIndex #samplingTrainFixedIndex  # get 1400 indexes as val set
    #range(0, samplingAdjNum - testNodesNum - valNodesNum)

    train_mask = sample_mask(idx_train, samplingAdjNum)
    val_mask = sample_mask(idx_val, samplingAdjNum)
    test_mask = sample_mask(idx_test, samplingAdjNum)

    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)

    dataDict = {"num_classes": data["num_classes"],
                "num_vertices": samplingAdjNum,
                "num_edges": data["num_edges"],
                "dim_features": X.shape[1],
                "features": X,
                "edge_list": edge_list,
                "labels": torch.tensor(samplingLabels),
                "sampling_idx_range": sampling_idx_range}

    return dataDict, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum

def loadFoursquare(data_str):
    edgesfilePath = data_str + '/following.npy'
    userlabelPath = data_str + '/multilabel2id.pkl'
    userFeaturesPath = data_str + '/userattr.npy'
    user = data_str + '/user'

    features = np.load(userFeaturesPath)
    features = np.float32(features)

    edges = np.load(edgesfilePath)

    f = open(userlabelPath, 'rb')
    labels = pickle.load(f)

    users = []
    for userNode in range(len(features)):
        if userNode in labels.keys():
            users.append(userNode)

    currentEdges = []
    for edge in edges:
        startNode = edge[0]
        endNode = edge[1]
        if startNode in labels.keys() and endNode in labels.keys():
            currentEdges.append(tuple([users.index(startNode), users.index(endNode)]))

    userlabels = []
    for userid in range(len(users)):
        userlabels.append(sum(labels[users[userid]])-1)

    currentFeatures = features[users]
    currentNodeNums = len(currentFeatures)
    current_num_edges = len(currentEdges)

    dataDict = {"num_classes": 9,
                "num_vertices": currentNodeNums,
                "num_edges": current_num_edges,
                "dim_features": features.shape[1],
                "features": torch.tensor(currentFeatures),
                "edge_list": currentEdges,
                "labels": torch.tensor(userlabels)}

    return dataDict

def getSamplingData(data):
    globalLabelStatistic = {}
    globalLabelIndex = {}
    classNum = 6  # flickr_9 facebook_4 BlogCatalog_6
    labels = data["labels"].numpy().tolist()
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(labels) if x == k]
        labelCount = len(labelIndex)
        globalLabelStatistic[k] = labelCount
        globalLabelIndex[k] = labelIndex

    #sampling 300 from class 1 and construct new edges__Blogcatalog
    samplingNum = 300
    samplingClassIndex = random.sample(globalLabelIndex[1], samplingNum)
    newSamplingNode = []
    newSamplingLabels = []
    newSamplingFeatures = []
    labels = data["labels"].numpy()
    for k in range(classNum):
        if k == 1:
            newSamplingNode += samplingClassIndex
            newSamplingLabels += list(labels[samplingClassIndex])
            newSamplingFeatures += data["features"][samplingClassIndex]
        else:
            newSamplingNode += globalLabelIndex[k]
            newSamplingLabels += list(labels[globalLabelIndex[k]])
            newSamplingFeatures += data["features"][globalLabelIndex[k]]

    currentGlobalNode = []
    for i in range(len(labels)):
        if i in newSamplingNode:
            currentGlobalNode.append(i)
    currentGlobalLabels = labels[currentGlobalNode]

    newSamplingEdges = []
    edges = data["edge_list"]
    for edge in edges:
        start = edge[0]
        end = edge[1]
        if start in currentGlobalNode and end in currentGlobalNode:
            newStartIndex = currentGlobalNode.index(start)
            newEndIndex = currentGlobalNode.index(end)
            newSamplingEdges.append(tuple([newStartIndex, newEndIndex]))
    np.save('BlogCatalog/newSampling_edge_list.npy', newSamplingEdges)
    np.save('BlogCatalog/newSamplingNode.npy', currentGlobalNode)
    np.save('BlogCatalog/newSamplingLabels.npy', currentGlobalLabels)
    # np.save('BlogCatalog/newSamplingFeatures.npy', newSamplingFeatures)

    dataDict = {"num_classes": classNum,
                "num_vertices": len(newSamplingNode),
                "num_edges": len(newSamplingEdges),
                "dim_features": data["dim_features"],  # data["dim_features"], #facebook_feature_dim,
                "features": newSamplingFeatures,
                "edge_list": newSamplingEdges,
                "labels": newSamplingLabels}

    return dataDict

def loadSamplingData(data):
    classNum = 6
    edges = np.load('BlogCatalog/newSampling_edge_list.npy')
    samplingNode = np.load('BlogCatalog/newSamplingNode.npy')
    samplingLables = np.load('BlogCatalog/newSamplingLabels.npy')
    # np.save('BlogCatalog/newSamplingFeatures.npy', newSamplingFeatures)

    newSamplingFeatures = data["features"][samplingNode]

    # edges_list = [tuple([edge[0], edge[1]]) for edge in edges]

    dataDict = {"num_classes": classNum,
                "num_vertices": len(samplingNode),
                "num_edges": len(samplingLables),
                "dim_features": data["dim_features"],  # data["dim_features"], #facebook_feature_dim,
                "features": newSamplingFeatures,
                "edge_list": edges,
                "labels": torch.tensor(samplingLables)}

    return dataDict

def getCutedGradient(grad, clip):
    gradShape = np.array(grad).shape
    norm2 = np.linalg.norm(grad, ord=2, axis=1, keepdims=True)
    norm2 = norm2 / clip
    cutedGrad = []
    for i in range(gradShape[0]):
        currentNorm = norm2[i]

        if currentNorm > 1:
            currentGrad = grad[i] / norm2[i]
        else:
            currentGrad = grad[i]

        cutedGrad.append(np.array(currentGrad).tolist())

    return cutedGrad


def getGlobalAdjMatrix(clientList):
    globalMatrix = 0
    for client in clientList:
        globalMatrix += client.samplingMatrix
    return globalMatrix

def getOverlapNodes1(clientList, clientNum):
    overlapNodes = []
    for i in range(clientNum):
        for j in range(i + 1, clientNum):
            samplingIndex_i = clientList[i]
            samplingIndex_j = clientList[j]
            for ind in samplingIndex_i:
                if ind in samplingIndex_j and ind not in overlapNodes:
                    overlapNodes.append(ind)
    return overlapNodes


def getOverlapNodes(globalMatrix):
    overlapNodes = []
    globalMatrixShape = globalMatrix.shape
    for idx in range(globalMatrixShape[0]):
        eachRow = globalMatrix[idx]
        rowIndexs = [i for i, x in enumerate(eachRow) if x != 1]

        for rowIdx in rowIndexs:
            if rowIdx not in overlapNodes:
                overlapNodes.append(rowIdx)
    return overlapNodes

def getGlobalOverlapNodesEmd(clientList, overlapNodes, clientOuts):
    globalNodeEmbeddings = []
    for idx in overlapNodes:
        mean = 0
        count = 0
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmb = clientOut[nodeIndex]
                # mean += currentNodeEmb
                mean += currentNodeEmb.detach().numpy()
                count += 1
        if count == 0:count = 1
        mean = mean / count
        # mean = mean.detach().numpy()
        expDis = []
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmd = clientOut[nodeIndex]
                currentNodeEmd = currentNodeEmd.detach().numpy()
                dist = np.linalg.norm(currentNodeEmd - mean)
                # dist = torch.norm(currentNodeEmd - mean)
                try:
                    expDis.append(math.exp(dist))
                except OverflowError:
                    expDis.append(math.exp(700))
                # print('dist:', dist)
                # print('math.exp(dist):', cmath.exp(dist))
                # expDis.append(math.exp(dist))

        finalNodeEmb = 0
        count = 0
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmd = clientOut[nodeIndex]
                currentNodeEmd = currentNodeEmd.detach().numpy()
                finalNodeEmb += (expDis[count] / sum(expDis)) * currentNodeEmd
        globalNodeEmbeddings.append(finalNodeEmb)
    return globalNodeEmbeddings

def setGlobalNodeEmdForLocalNodes(clientList, clientOuts, overlapNodes, globalNodeEmbeddings):      ##设置不同客户端的重叠节点的增量节点
    for idx in overlapNodes:
        for i in range(clientNum):
            net = clientList[i]
            # clientOut = clientOuts[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                # nodeIndexs[clientNum].append(nodeIndex)
                clientOuts[i][nodeIndex].data = torch.Tensor(globalNodeEmbeddings[idx])
    return clientOuts

def getLeftSamplingNodes(totalSampleNum, samplingOverlappedNodesIndex):
    totalIndex = list(range(0, totalSampleNum))
    totalCopyIndex = deepcopy(totalIndex)

    for currentIndex in samplingOverlappedNodesIndex:
        totalIndex.remove(currentIndex)

    return totalIndex, totalCopyIndex


class GlobalGCN(nn.Module):
    def __init__(self, data: dict,
                 test_idx: list):
        super().__init__()
        self.data = data
        self.test_idx = test_idx
        hid_channels = 16

        self.globalModel = GCNS(data["dim_features"], hid_channels, data["num_classes"], 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.X, self.lbls = self.data["features"], self.data["labels"]
        self.A = Graph(self.data["num_vertices"], self.data["edge_list"])
        self.testNodesNum = int(self.data["num_vertices"] * 0.4)
        self.test_mask = self.getTestMask()
        self.acc = []
        self.f1 = []
        self.pre = []
        self.recall = []

    def getTestMask(self):
        idx_test = []
        for i in range(len(self.test_idx)):
            idx_test += self.test_idx[i]
        # idx_test = range(self.data["num_vertices"] - self.testNodesNum, self.data["num_vertices"])
        test_mask = sample_mask(idx_test, self.data["num_vertices"])
        test_mask = torch.tensor(test_mask)
        return test_mask

    def test(self, globalParam, epoch, iteration):
        # test
        print(f"global--test...")
        for pid, param in enumerate(list(self.globalModel.parameters())):
            param.data = torch.tensor(globalParam[pid+4], dtype=torch.float32) #+7

        res, f0, recall0, precision0 = infer(self.globalModel, self.X, self.A.L_GCN.to_dense(), self.lbls, self.test_mask, test=True)
        self.acc.append(res['accuracy'])
        self.f1.append(f0)
        self.recall.append(recall0)
        self.pre.append(precision0)

        if epoch == iteration - 1:
            acc_avg = sum(self.acc[iteration - 1 - 10:iteration - 1])
            f1_avg = sum(self.f1[iteration - 1 - 10:iteration - 1])
            recall_avg = sum(self.recall[iteration - 1 - 10:iteration - 1])
            pre_avg = sum(self.pre[iteration - 1 - 10:iteration - 1])
            print('avg acc:', acc_avg / 10, 'avg f2:', f1_avg / 10, 'avg recall:', recall_avg / 10, 'avg pre:',
                  pre_avg / 10)

            idx_test = []
            for i in range(len(self.test_idx)):
                idx_test += self.test_idx[i]

            self.globalModel.eval()
            outs = self.globalModel(self.X, self.A.L_GCN.to_dense())
            tsne = TSNE(n_components=2)
            x_tsne = tsne.fit_transform(outs[0].detach().numpy())
            fig = plt.figure()
            preLabel = np.argmax(outs[1].detach().numpy(), axis=1)

            dataframe = pd.DataFrame(
                {'x0': x_tsne[:, 0], 'x1': x_tsne[:, 1],
                 'c': preLabel})  # save data
            dataframe.to_csv('cora/Community3/cora_scatter.csv', index=False, sep=',')

            plt.scatter(x_tsne[:, 0][idx_test], x_tsne[:, 1][idx_test], c=preLabel[idx_test], label="t-SNE")
            fig.savefig('cora/Community3/cora_scatter.png')
            plt.show()

        print(res)


class ClientGCN(nn.Module): #
    def __init__(self, data: dict,
                       train_idx: torch.Tensor,
                       test_mask: torch.Tensor,
                       val_mask: torch.Tensor,
                       trainNum: int
                       ):
        super().__init__()
        self.data = data
        hid_channels = 16
        project_channels = 300

        self.globalCModel = GCNT1(data["dim_features"], 16, data["num_classes"], temperature=0.1)

        self.localSModel = GCNS(data["dim_features"], hid_channels, data["num_classes"])

        self.relate = torch.nn.Parameter(torch.rand(hid_channels, hid_channels))
        self.act = nn.Sigmoid()
        self.intratc = 0.5
        self.intertc = 0.5

        self.alpha = 0.5
        self.temp = 0.05

        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train_idx = train_idx
        self.X, self.lbls = self.data["features"], self.data["labels"]
        self.A = Graph(self.data["num_vertices"], self.data["edge_list"])
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.best_state = None
        self.best_epoch = 0
        self.best_val = 0
        self.trainNum = trainNum
        self.valAccuracy = []


    def getLocalTrainOut(self):
        self.train()
        self.st = time.time()
        self.optimizer.zero_grad()  # 清空过往的梯度

        outs = self.localSModel(self.X, self.A.L_GCN.to_dense())#self.A.L_GCN.to_dense()
        return outs

    def getAugGraph(self, localOut):
        h = localOut[0]
        # adj_logits = h @ h.T
        adj_logits = h @ self.relate @ h.T
        adj_logits = self.act(adj_logits)

        edge_probs = adj_logits / torch.max(adj_logits)

        adj_orig = self.A.L_GCN.to_dense()
        adj_orig[adj_orig <= 0] = 0
        adj_orig[adj_orig > 0] = 1
        edge_probs = self.alpha * edge_probs + (1 - self.alpha) * adj_orig

        # Gumbel-Softmax Sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp,
                                                                         probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T

        adj_sampled.fill_diagonal_(1)
        D_norm = torch.diag(torch.pow(adj_sampled.sum(1), -0.5))
        adj_sampled = D_norm @ adj_sampled @ D_norm

        return adj_sampled, adj_logits

    def getKnnGraph(self, localOut):
        globalknnGraph= kneighbors_graph(localOut[0].detach().numpy(), n_neighbors=10, metric='minkowski')
        edge_list = []
        for row, col in zip(*globalknnGraph.nonzero()):
            edge_list.append(tuple([row, col]))
        globalKnnAdj = Graph(self.data["num_vertices"], edge_list)

        return globalKnnAdj.L_GCN.to_dense()


    def trainNet(self, localOut, clientId, epoch):
        self.augAdj, adj_logits = self.getAugGraph(localOut)

        outg1 = self.globalCModel(self.X, self.augAdj)
        outl, lbls = localOut[1][self.train_idx], self.lbls[self.train_idx]

        intraCenterLoss, interCenterLoss = self.getIntraAndInterCenterContrastLoss(localOut[0][self.train_idx], lbls)
        crossLoss1 = self.globalCModel.getCrossLoss(localOut[0][self.train_idx], outg1[0][self.train_idx],
                                                    self.lbls[self.train_idx])
        a = 0.0001
        b = 0.0001
        c = 0.01
        d = 0.05

        adj_orig = self.A.L_GCN.to_dense()
        adj_orig[adj_orig <= 0] = 0
        adj_orig[adj_orig > 0] = 1
        norm_w = self.data["num_vertices"] ** 2 / float((self.data["num_vertices"] ** 2 - adj_orig.sum()) * 2)
        pos_weight = torch.FloatTensor([float(self.data["num_vertices"] ** 2 - adj_orig.sum()) / adj_orig.sum()])
        genAdj_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)

        class_loss0 = F.cross_entropy(outl, lbls)
        class_loss1 = F.cross_entropy(outg1[1][self.train_idx], lbls)

        self.loss = class_loss0 + class_loss1 + a * intraCenterLoss + d * interCenterLoss +\
                    b * crossLoss1 + c * genAdj_loss

        self.loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新梯度
        print(f"clientId:{clientId}, Epoch: {epoch}, Time: {time.time() - self.st:.5f}s, Loss: {self.loss.item():.5f}")
        return self.loss.item()


    def getIntraAndInterCenterContrastLoss(self, x, label):
        """
             Args:
                 x: feature matrix with shape (batch_size, feat_dim).
                 labels: ground truth labels with shape (batch_size).
             """
        #--------intra----------------#
        batch_size = x.size(0)
        currentCenter = self.centers[label]
        # print(self.centers)

        ###############based on similarity##########################
        posSim = torch.cosine_similarity(x, currentCenter, dim=1)
        posSim = torch.exp((posSim / self.intratc) / 100)

        eachX = x[0].repeat([data["num_classes"], 1])
        negSim = torch.cosine_similarity(eachX, self.centers, dim=1)
        negSim = negSim.sum(dim=0, keepdim=True)

        for i in range(1, batch_size):
            eachX = x[i].repeat([data["num_classes"], 1])
            eachSim = torch.cosine_similarity(eachX, self.centers, dim=1)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            negSim = torch.cat((negSim, eachSim), dim=0)

        negSim = torch.exp((negSim / self.intratc) / 100)
        intraloss = (-torch.log(posSim / negSim)).mean()

        ##--------inter---------------#
        with torch.autograd.set_detect_anomaly(True):
            classLoss = 0
            #将center投影到一个新的线性空间，作为正例
            self.pCenters = torch.mm(self.centers, self.centerPW)

            #得到正例
            interPosSim = torch.cosine_similarity(self.centers, self.pCenters, dim=1)
            interPosSim = torch.exp((interPosSim / self.intertc) / 100) #得到正例
            #
            # #得到第一个负例
            for i in range(len(self.centers)):
                eachX = x[i].repeat([data["num_classes"], 1])
                curCenters = self.centers.clone()
                curCenters[i] = self.pCenters[i]
                eachSim = torch.cosine_similarity(eachX, curCenters, dim=1)
                eachSim = eachSim.sum(dim=0, keepdim=True)
                if i == 0:
                   interNegSim = eachSim
                else:
                   interNegSim = torch.cat((interNegSim, eachSim), dim=0)

            interNegSim = torch.exp((interNegSim / self.intertc) / 100)
            interloss = (-torch.log(interPosSim / interNegSim)).mean()


        return intraloss, interloss


    def val(self, clientId, epoch):
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(self.localSModel, self.X, self.A.L_GCN.to_dense(), self.lbls, self.val_mask)
                self.valAccuracy.append(val_res)
            if val_res > self.best_val:
                print(f"clientId:{clientId}, update best: {val_res:.5f}")
                self.best_epoch = epoch
                self.best_val = val_res
                self.best_state = deepcopy(self.localSModel.state_dict())
            return val_res

    def test(self, clientId):
        print(f"clientId:{clientId}, best val: {self.best_val:.5f}")
        # test
        print(f"clientId:{clientId}, test...")
        self.localSModel.load_state_dict(self.best_state)
        res = infer(self.localSModel, self.X, self.A.L_GCN.to_dense(), self.lbls, self.test_mask, test=True)
        print(f"clientId:{clientId}, final result: epoch: {self.best_epoch}")
        print(res)

    def saveFinalEmbedding(self, saveName):
        self.localSModel.eval()
        outs = self.localSModel(self.X, self.A.L_GCN.to_dense())
        outs = outs[0].detach().numpy()
        # valAcc = np.array(self.valAccuracy)
        # #------save  embedding---------------#
        # valFileOuts = saveName + "-valAcc.npy"
        # np.save(valFileOuts, valAcc)
        fileOuts = saveName + "-outs.npy"
        np.save(fileOuts, outs)
        #------save sampling index-----------#
        # fileIndex = saveName + "-samplingIndex.npy"
        # np.save(fileIndex, self.data["sampling_idx_range"])
        print(saveName + "  save success")



def load_graph(edges):
    G = collections.defaultdict(dict)
    for edge in edges:
        w = 1.0  # 数据集有权重的话则读取数据集中的权重
        G[edge[0]][edge[1]] = w
    return G

def get_global_graph(all_edges, all_features, nodes_id):
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in all_edges]
    df['target'] = [edge[1] for edge in all_edges]

    nodes = sg.IndexedArray(all_features, nodes_id)
    G = StellarGraph(nodes=nodes, edges=df)

    return G

def louvain_graph_cut(whole_graph: StellarGraph, node_subjects, num_owners):   #对数据的处理
    delta = 70
    edges = np.copy(whole_graph.edges())
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in edges]
    df['target'] = [edge[1] for edge in edges]
    G = StellarGraph.to_networkx(whole_graph)

    partition = community_louvain.best_partition(G)

    groups = []

    for key in partition.keys():   #这个是找到有多少组
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():  #初始化set，将相应的节点对应到不同的组中
        partition_groups[partition[key]].append(key)

    group_len_max = len(list(whole_graph.nodes()))//num_owners-delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups)+1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]

    print(groups)

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1],reverse=True)}

    owner_node_ids = {owner_id: [] for owner_id in range(num_owners)}

    owner_nodes_len = len(list(G.nodes()))//num_owners
    owner_list = [i for i in range(num_owners)]
    owner_ind = 0


    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) >= owner_nodes_len + delta:
            owner_ind = (owner_ind + 1) % len(owner_list)
            if owner_ind + 1 == len(owner_list):
               break
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    for owner_i in owner_node_ids.keys():
        print('nodes len for ' + str(owner_i) + ' = '+str(len(owner_node_ids[owner_i])))

    subj_set = list(set(node_subjects.values))  #类别
    local_node_subj_0 = []
    for owner_i in range(num_owners):
        partition_i = owner_node_ids[owner_i]
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        sbj_i = node_subjects.copy(deep=True)
        sbj_i.values[:] = "" if node_subjects.values[0].__class__ == str else 0
        sbj_i.values[locs_i] = node_subjects.values[locs_i]
        local_node_subj_0.append(sbj_i)
    count = []
    for owner_i in range(num_owners):
        count_i = {k: [] for k in subj_set}
        sbj_i = local_node_subj_0[owner_i]
        for i in sbj_i.index:
            if sbj_i[i] != 0 and sbj_i[i] != "":
                count_i[sbj_i[i]].append(i)
        count.append(count_i)
    for k in subj_set:
        for owner_i in range(num_owners):
            if len(count[owner_i][k]) < 2:
                for j in range(num_owners):
                    if len(count[j][k]) > 2:
                        id = count[j][k][-1]
                        count[j][k].remove(id)
                        count[owner_i][k].append(id)
                        owner_node_ids[owner_i].append(id)
                        owner_node_ids[j].remove(id)
                        j = num_owners

    return owner_node_ids



if __name__ == "__main__":
    set_seed(2022)# set_seed(2023) #BlogCatalog #Flickr
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    # data = BlogCatalog()
    # data = Facebook()
    # data = Flickr()
    data = Cora()
    facebook_feature_dim = 4714
    dataDict = {"num_classes": data["num_classes"],
                "num_vertices": data["num_vertices"],
                "num_edges": data["num_edges"],
                "dim_features": data["dim_features"], #data["dim_features"], #data["dim_features"], #facebook_feature_dim, #data["dim_features"], #facebook_feature_dim,
                "features": data["features"],
                "edge_list": data["edge_list"],
                "labels": data["labels"]}


    clientNum = 3

    G = get_global_graph(data["edge_list"], data["features"].numpy(), range(0, data["num_vertices"]))

    node_subjects = pd.Series(data["labels"])

    communities = louvain_graph_cut(G, node_subjects, clientNum)


    dataList, train_mask_list, val_mask_list, test_mask_list, \
    test_idx_list, trainN_list, val_idx_list = getCommunitData(data, communities, node_subjects, clientNum)

    #--------------splitting data and create client net----------------#
    train_local = False
    train_fedavg = False
    center_flag = True

    clientList = []

    # samplingRate = [0.3, 0.4, 0.5, 0.5, 0.6, 0.7]
    samplingRate = [0.3, 0.4, 0.6]
    # samplingRate = [1, 1, 1, 1, 1, 1]
    # samplingRate = [0.7, 0.4, 0.5, 0.5, 0.6, 0.7]
    classNum = data["num_classes"]
    # sampleNumEachClass = [5, 5, 10, 10, 20, 30]                #Citeseer 200  others_10
    sampleNumEachClass = [20, 20, 10, 10, 5, 5]                   #
    filePath = 'cora/Community3/'              #"BlogCatalog/"  'flickr/'   #BlogCatalog   #facebook_200
    overlapRate = 0.15
    # clientSamplingIndex = random_dataset_overlap(data, clientNum, overlapRate)
    globalKModels = []
    n_clusters = 60
    hid_channels = 16
    globalNet = GlobalGCN(data, test_idx_list)
    valist = {clientId: [] for clientId in range(clientNum)}

    for i in range(clientNum):
        saveName = filePath + 'client' + str(i)
        clientData = dataList[i]
        train_mask = train_mask_list[i]
        val_mask = val_mask_list[i]
        test_mask = test_mask_list[i]
        train_Num = trainN_list[i]

        net = ClientGCN(clientData, train_mask, val_mask, test_mask, train_Num)


        #---------------保存初始化参数-----------#
        path0 = "client" + str(i) + "_cora_3_c_localModel_lv.pkl"
        # net.localSModel.load_state_dict(torch.load(path0))
        # torch.save(net.localSModel.state_dict(), path0)
        path1 = "client" + str(i) + "_cora_3_c_globalKModel.pkl"
        # net.globalKModel.load_state_dict(torch.load(path1))
        # torch.save(net.globalKModel.state_dict(), path1)
        path2 = "client" + str(i) + "_cora_3_c_globalCModel.pkl"
        # torch.save(net.globalCModel.state_dict(), path2)
        # net.globalCModel.load_state_dict(torch.load(path1))

        clientList.append(net)

    iteration = 2000
    if center_flag:
        aggParams = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
        for epoch in range(iteration):
            # 1、-------training and aggerating------------------#
            localOuts = []
            for i in range(clientNum):
                localOut = clientList[i].getLocalTrainOut()
                localOuts.append(localOut)

            # getClusterAdj(localOuts, clientList, n_clusters)

            for i in range(clientNum):
                clientList[i].trainNet(localOuts[i], i, epoch)
                valll = clientList[i].val(i, epoch)
                valist[i].append(valll)
                #####################################################
                for pid, param in enumerate(list(clientList[i].parameters())):
                       aggParams[pid] += param.detach().numpy()

            # 2、--------getting average parameters-------------#
            for id, aggParam in aggParams.items():
                   aggParams[id] = aggParam / clientNum

            expDis = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [],
                      6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: []}  # , 4: [], 5: [], 6: [], 7: [], 8: []
            for i in range(clientNum):
                for pid, param in enumerate(list(clientList[i].parameters())):
                    currentParam = param.detach().numpy()
                    meanParam = aggParams[pid]
                    dist = np.linalg.norm(currentParam - meanParam)
                    if dist > 700:
                        currentDis = math.exp(dist / 100)  # math.exp(dist/100)*math.exp(100)
                    else:
                        currentDis = math.exp(dist)
                    expDis[pid].append(currentDis)

            globalAggParam = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
                              6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}  # , 4: 0, 5: 0, 6: 0, 7: 0, 8: 0
            for i in range(clientNum):
                for pid, param in enumerate(list(clientList[i].parameters())):
                    globalAggParam[pid] += (expDis[pid][i] / sum(expDis[pid])) * param.detach().numpy()
            #
            # # 3、-----------setting aggerated parameters for each client-----------#
            u = 0.01
            b = 0.1
            for i in range(clientNum):
                for pid, param in enumerate(list(clientList[i].parameters())):
                    # if pid == 2:
                        param.data = b * param.data + (1-b) * torch.tensor(globalAggParam[pid])
                    # else:
                    #     param.data = torch.tensor(globalAggParam[pid])
                    # param.data = torch.tensor(globalAggParam[pid])  # aggParams[pid] + u * (aggParams[pid] - param.data) #aggParams[pid] #finalAggParam#aggParams[pid]
                    # param.data = torch.tensor(aggParams[pid] + u * (
                    #         aggParams[pid] - param.detach().numpy()))
            # 4、----------clear aggParams--------------------------#
            for key in aggParams.keys():
                aggParams[key] = 0

            # if epoch == 1999:
            globalNet.test(globalAggParam, epoch, iteration)


    if train_local:
        aggParams = {}
        for pid, param in enumerate(list(clientList[0].parameters())):
            aggParams[pid] = 0

        for epoch in range(iteration):
            #1、-------training and aggerating------------------#
            localOuts = []
            for i in range(clientNum):
                localOut = clientList[i].getLocalTrainOut()
                localOuts.append(localOut)

            for i in range(clientNum):
                clientList[i].trainNet(localOuts[i], i, epoch)
                valll = clientList[i].val(i, epoch)
                valist[i].append(valll)
               #####################################################
                if train_fedavg:
                   for pid, param in enumerate(list(clientList[i].parameters())):
                       aggParams[pid] += param.detach().numpy()

            #2、--------getting average parameters-------------#
            if train_fedavg:
                for id, aggParam in aggParams.items():
                    aggParams[id] = aggParam/clientNum

                #3、-----------setting aggerated parameters for each client-----------#
                u = 0.05
                for i in range(clientNum):
                    for pid, param in enumerate(list(clientList[i].parameters())):
                        # param.data = torch.tensor(aggParams[pid])#aggParams[pid] + u * (aggParams[pid] - param.data) #aggParams[pid] #finalAggParam#aggParams[pid]
                        param.data = torch.tensor(aggParams[pid] + u * (
                                aggParams[pid] - param.detach().numpy()))
                # if epoch == 999:
                globalNet.test(aggParams, epoch, iteration)

                #4、----------clear aggParams--------------------------#
                for key in aggParams.keys():
                    aggParams[key] = 0
    ###-------plot-------
    color = 'red'
    plt.figure(1)
    plt.plot(range(iteration), globalNet.acc, '-', color=color)

    plt.figure(2)
    plt.plot(range(iteration), valist[0], '-', color='red')
    plt.figure(3)
    plt.plot(range(iteration), valist[1], '-', color='green')
    plt.figure(4)
    plt.plot(range(iteration), valist[2], '-', color='yellow')

    plt.show()
    dataframe0 = pd.DataFrame(
        {'x': range(iteration), 'acc': globalNet.acc})  # save data
    dataframe1 = pd.DataFrame(
        {'x': range(iteration), 'f1': globalNet.f1})  # save data
    dataframe2 = pd.DataFrame(
        {'x': range(iteration), 'precision': globalNet.pre})  # save data
    dataframe3 = pd.DataFrame(
        {'x': range(iteration), 'recall': globalNet.recall})  # save data
    dataframe0.to_csv('cora/Community3/3_acc.csv', index=False, sep=',')
    dataframe1.to_csv('cora/Community3/3_f1_acc.csv', index=False, sep=',')
    dataframe2.to_csv('cora/Community3/cc/3_precision_acc.csv', index=False, sep=',')
    dataframe3.to_csv('cora/Community3/cc/3_recall_acc.csv', index=False, sep=',')

    # print("\nsave loss")
    # np.save('BlogCatalog/loss.npy', clientLoss)  # 注意带上后缀名
    #
    # print("\nsave acc")
    # np.save('BlogCatalog/acc.npy', clientAcc)  # 注意带上后缀名

    print("\ntrain finished!")
    for i in range(clientNum):
        clientList[i].test(i)


    print("\nsave final embedding")
    for i in range(clientNum):
        saveName = filePath + "client" + str(i)
        clientList[i].saveFinalEmbedding(saveName)


