from networkx.algorithms.operators.unary import reverse
import torch
import random
from FileReader import file_reader
import PerformanceMetrics as PM
import binaryOperatorsForLearningEdgeFeatures as BOLEF
import read_Edge_Stream as EdgeStream
from predictActualDistancePath import predictProbaActualShortestPathDistance
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score,mean_squared_error, recall_score, precision_score
import math
import time
import pandas as pd
import networkx as nx

def readEmbeddings(embeddingFileName):

    embeddingFileRead = open(embeddingFileName, "r")
    embeddings_dict = {}

    for line in file_reader(embeddingFileRead):
        line = line.split(' ')
        node = int(line[0])
        embedding = []
        for num in range(1,len(line)):
            embedding.append(float(line[num]))
        embeddings_dict.update({node:embedding})
    

    return embeddings_dict

def readGraph(graphFileName):

    graphFileRead = open(graphFileName, "r")
    edgesList = []
    nodesList = []
    for line in file_reader(graphFileRead):
        line = line.split(',')
        edgesList.append((int(line[0]),int(line[1])))
        if (line[0] not in nodesList):
            nodesList.append(int(line[0]))
        if (line[1] not in nodesList):
            nodesList.append(int(line[1]))

    return edgesList, nodesList

def createTrainSet(edgesList,nodesList,embeddings_dict,deletedEdgesList):
    embeddingEdgesList = []
    edgeExistList = []
    edgeList = []
    for edge in edgesList:
        reverse_edge = (edge[1],edge[0])
        if not((edge[0],edge[1]) in deletedEdgesList or reverse_edge in deletedEdgesList):
            edgeList.append(edge)
            embedding1 = embeddings_dict.get(edge[0])
            
            embedding2 = embeddings_dict.get(edge[1])
            embList = []
            for i in range(len(embedding1)):
                emb_feature =(embedding1[i]*embedding2[i])
                #emb_feature= embedding1[i]
                embList.append(emb_feature)
            '''for i in range(len(embedding2)):
                #emb_feature =(embedding1[i]+embedding2[i])/2
                emb_feature= embedding2[i]
                embList.append(emb_feature)'''
            embeddingEdgesList.append(embList)
            edgeExistList.append(1)
    alreadyaddedembeddings = len(embeddingEdgesList)
    addedembeddings = 0
    for nodeNum in range(len(nodesList)):
        node1 = nodesList[nodeNum]
        node2Num = random.randint(0, len(nodesList)-1)
        while (nodeNum == node2Num):
            node2Num = random.randint(0, len(nodesList)-1)
        node2 = nodesList[node2Num]
        edge = (node1,node2)
        reverse_edge = (node2,node1)
        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList):
            node2Num = random.randint(0, len(nodesList)-1)
            while (nodeNum == node2Num):
                node2Num = random.randint(0, len(nodesList)-1)
            node2 = nodesList[node2Num]
            edge = (node1,node2)
            reverse_edge = (node2,node1)
        embedding1 = embeddings_dict.get(edge[0])
        embedding2 = embeddings_dict.get(edge[1])
        embList = []
        for i in range(len(embedding1)):
            emb_feature =(embedding1[i]*embedding2[i])
            #emb_feature= embedding1[i]
            embList.append(emb_feature)
        '''for i in range(len(embedding2)):
            #emb_feature =(embedding1[i]+embedding2[i])/2
            emb_feature= embedding2[i]
            embList.append(emb_feature)'''
        embeddingEdgesList.append(embList)
        addedembeddings+=1
        
        
        edgeExistList.append(0)
        if(addedembeddings>= alreadyaddedembeddings):
            break
    return embeddingEdgesList,edgeExistList

def createTestSet(deletedEdgesList,embeddings_dict,nodesList,edgeList):
    X_test = []
    y_test = []

    for edge in deletedEdgesList:
        #print(edge)
        embedding1 = embeddings_dict.get(edge[0])
        embedding2 = embeddings_dict.get(edge[1])
        embList = []
        for i in range(len(embedding1)):
            emb_feature =(embedding1[i]*embedding2[i])
            #emb_feature= embedding1[i]
            embList.append(emb_feature)
        '''for i in range(len(embedding2)):
            #emb_feature =(embedding1[i]+embedding2[i])/2
            emb_feature= embedding2[i]
            embList.append(emb_feature)'''
        X_test.append(embList)
        y_test.append(1)
    alreadyaddedembeddings = len(X_test)
    addedembeddings = 0
    for nodeNum in range(len(nodesList)):
        node1 = nodesList[nodeNum]
        node2Num = random.randint(0, len(nodesList)-1)
        while (nodeNum == node2Num):
            node2Num = random.randint(0, len(nodesList)-1)
        node2 = nodesList[node2Num]
        edge = (node1,node2)
        reverse_edge = (node2,node1)
        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList):
            node2Num = random.randint(0, len(nodesList)-1)
            while (nodeNum == node2Num):
                node2Num = random.randint(0, len(nodesList)-1)
            node2 = nodesList[node2Num]
            edge = (node1,node2)
            reverse_edge = (node2,node1)
        embedding1 = embeddings_dict.get(edge[0])
        embedding2 = embeddings_dict.get(edge[1])
        embList = []
        for i in range(len(embedding1)):
            emb_feature =(embedding1[i]*embedding2[i])
            #emb_feature= embedding1[i]
            embList.append(emb_feature)
        '''for i in range(len(embedding2)):
            #emb_feature =(embedding1[i]+embedding2[i])/2
            emb_feature= embedding2[i]
            embList.append(emb_feature)'''
        X_test.append(embList)
        addedembeddings+=1
        
        
        y_test.append(0)
        if(addedembeddings>= alreadyaddedembeddings):
            break
    return X_test, y_test




testFilesList = ['email-Eu-core-temporal_Batches_testSet.csv','reality-call_Batches_testSet.csv']
trainFilesList = ['email-Eu-core-temporal_Batches_trainSet.csv','reality-call_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_email-Eu-core-temporal_Batches_trainSet.emb','embeddings_reality-call_Batches_trainSet.emb']
shortedStreamFilesList = ['email-Eu-core-temporal_Batches_sortedEdgeStream.csv','reality-call_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['email-Eu-core-temporal_Batches_sortedEdgeStream_tests.csv','reality-call_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['email-Eu-core-temporal_Batches_sortedEdgeStream_Path_tests.csv','reality-call_Batches_sortedEdgeStream_Path_tests.csv']
pivotList = [293,343,1086]


 


testFilesList = ['email-Eu-core-temporal_Batches_testSet.csv','soc-sign-bitcoinalpha_Batches_testSet.csv']
trainFilesList = ['email-Eu-core-temporal_Batches_trainSet.csv','soc-sign-bitcoinalpha_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_email-Eu-core-temporal_Batches_trainSet.emb',]
shortedStreamFilesList = ['email-Eu-core-temporal_Batches_sortedEdgeStream.csv','soc-sign-bitcoinalpha_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['email-Eu-core-temporal_Batches_sortedEdgeStream_tests.csv','soc-sign-bitcoinalpha_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['email-Eu-core-temporal_Batches_sortedEdgeStream_Path_tests.csv','soc-sign-bitcoinalpha_Batches_sortedEdgeStream_Path_tests.csv']

pivotList = [31,109]







'''testFilesList = ['email-Eu-core-temporal_Batches_testSet.csv']
trainFilesList = ['email-Eu-core-temporal_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_email-Eu-core-temporal_Batches_trainSet.emb']
shortedStreamFilesList = ['email-Eu-core-temporal_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['email-Eu-core-temporal_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['email-Eu-core-temporal_Batches_sortedEdgeStream_Path_tests.csv']

pivotList = [293]'''

'''testFilesList = ['reality-call_Batches_testSet.csv']
trainFilesList = ['reality-call_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_reality-call_Batches_trainSet.emb']
#embeddingFilesList = ['reality-call_Batches_sortedEdgeStream_multilens_TS_s_emb.tsv']
shortedStreamFilesList = ['reality-call_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['reality-call_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['reality-call_Batches_sortedEdgeStream_Path_tests.csv']

pivotList = [343]'''





'''testFilesList = ['soc-sign-bitcoinalpha_Batches_testSet.csv']
trainFilesList = ['soc-sign-bitcoinalpha_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_soc-sign-bitcoinalpha_Batches_trainSet.emb']
shortedStreamFilesList = ['soc-sign-bitcoinalpha_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['soc-sign-bitcoinalpha_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['soc-sign-bitcoinalpha_Batches_sortedEdgeStream_Path_tests.csv']


pivotList = [1086]'''




'''testFilesList = ['testGraph_testSet.csv','testGraph2_testSet.csv','testGraph3_testSet.csv','testGraph4_testSet.csv','testGraph5_testSet.csv','testGraph6_testSet.csv']
trainFilesList = ['testGraph_trainSet.csv','testGraph2_trainSet.csv','testGraph3_trainSet.csv','testGraph4_trainSet.csv','testGraph5_trainSet.csv','testGraph6_trainSet.csv']
embeddingFilesList = ['embeddings_testGraph_trainSet.emb','embeddings_testGraph2_trainSet.emb','embeddings_testGraph3_trainSet.emb','embeddings_testGraph4_trainSet.emb','embeddings_testGraph5_trainSet.emb','embeddings_testGraph6_trainSet.emb']
shortedStreamFilesList = ['testGraph_sortedEdgeStream.csv','testGraph2_sortedEdgeStream.csv','testGraph3_sortedEdgeStream.csv','testGraph4_sortedEdgeStream.csv','testGraph5_sortedEdgeStream.csv','testGraph6_sortedEdgeStream.csv']
groundTruthFilesList = ['testGraph_sortedEdgeStream_tests.csv','testGraph2_sortedEdgeStream_tests.csv','testGraph3_sortedEdgeStream_tests.csv','testGraph4_sortedEdgeStream_tests.csv','testGraph5_sortedEdgeStream_tests.csv','testGraph6_sortedEdgeStream_tests.csv']
groundTruthpathList = ['testGraph_sortedEdgeStream_Path_tests.csv','testGraph2_sortedEdgeStream_Path_tests.csv','testGraph3_sortedEdgeStream_Path_tests.csv','testGraph4_sortedEdgeStream_Path_tests.csv','testGraph5_sortedEdgeStream_Path_tests.csv','testGraph6_sortedEdgeStream_Path_tests.csv']'''

'''testFilesList = ['testGraph3_testSet.csv','testGraph4_testSet.csv','testGraph5_testSet.csv','testGraph6_testSet.csv']
trainFilesList = ['testGraph3_trainSet.csv','testGraph4_trainSet.csv','testGraph5_trainSet.csv','testGraph6_trainSet.csv']
embeddingFilesList = ['embeddings_testGraph3_trainSet.emb','embeddings_testGraph4_trainSet.emb','embeddings_testGraph5_trainSet.emb','embeddings_testGraph6_trainSet.emb']
shortedStreamFilesList = ['testGraph3_sortedEdgeStream.csv','testGraph4_sortedEdgeStream.csv','testGraph5_sortedEdgeStream.csv','testGraph6_sortedEdgeStream.csv']
groundTruthFilesList = ['testGraph3_sortedEdgeStream_tests.csv','testGraph4_sortedEdgeStream_tests.csv','testGraph5_sortedEdgeStream_tests.csv','testGraph6_sortedEdgeStream_tests.csv']
groundTruthpathList = ['testGraph3_sortedEdgeStream_Path_tests.csv','testGraph4_sortedEdgeStream_Path_tests.csv','testGraph5_sortedEdgeStream_Path_tests.csv','testGraph6_sortedEdgeStream_Path_tests.csv']
pivotTimeList = [42031355]'''
infolist = []
#pivotList = [293,343,1086]




testFilesList = ['email-Eu-core-temporal_Batches_testSet.csv']
trainFilesList = ['email-Eu-core-temporal_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_email-Eu-core-temporal_Batches_trainSet.emb']
shortedStreamFilesList = ['email-Eu-core-temporal_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['email-Eu-core-temporal_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['email-Eu-core-temporal_Batches_sortedEdgeStream_Path_tests.csv']

pivotList = [31]



'''testFilesList = ['soc-sign-bitcoinalpha_Batches_testSet.csv']
trainFilesList = ['soc-sign-bitcoinalpha_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_soc-sign-bitcoinalpha_Batches_trainSet.emb']
shortedStreamFilesList = ['soc-sign-bitcoinalpha_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['soc-sign-bitcoinalpha_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['soc-sign-bitcoinalpha_Batches_sortedEdgeStream_Path_tests.csv']

pivotList = [109]'''

'''testFilesList = ['ia-enron-employees_Batches_testSet.csv']
trainFilesList = ['ia-enron-employees_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_ia-enron-employees_Batches_trainSet.emb']
shortedStreamFilesList = ['ia-enron-employees_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['ia-enron-employees_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['ia-enron-employees_Batches_sortedEdgeStream_Path_tests.csv']

pivotList = [9]



testFilesList = ['CollegeMsg_Batches_testSet.csv']
trainFilesList = ['CollegeMsg_Batches_trainSet.csv']
embeddingFilesList = ['embeddings_CollegeMsg_Batches_trainSet.emb']
shortedStreamFilesList = ['CollegeMsg_Batches_sortedEdgeStream.csv']
groundTruthFilesList = ['CollegeMsg_Batches_sortedEdgeStream_tests.csv']
groundTruthpathList = ['CollegeMsg_Batches_sortedEdgeStream_Path_tests.csv']

pivotList = [10]'''



testFilesList = ['email-Eu-core-temporal_MonthInt_testSet.csv','reality-call_MonthInt_testSet.csv','soc-sign-bitcoinalpha_MonthInt_testSet.csv','ia-enron-employees_MonthInt_testSet.csv','CollegeMsg_MonthInt_testSet.csv']
trainFilesList = ['email-Eu-core-temporal_MonthInt_trainSet.csv','reality-call_MonthInt_trainSet.csv','soc-sign-bitcoinalpha_MonthInt_trainSet.csv','ia-enron-employees_MonthInt_trainSet.csv','CollegeMsg_MonthInt_trainSet.csv']
embeddingFilesList = ['embeddings_email-Eu-core-temporal_MonthInt_trainSet.emb','embeddings_reality-call_MonthInt_trainSet.emb','embeddings_soc-sign-bitcoinalpha_MonthInt_trainSet.emb','embeddings_ia-enron-employees_MonthInt_trainSet.emb','embeddings_CollegeMsg_MonthInt_trainSet.emb']
shortedStreamFilesList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream.csv','reality-call_MonthInt_sortedEdgeStream.csv','soc-sign-bitcoinalpha_MonthInt_sortedEdgeStream.csv','ia-enron-employees_MonthInt_sortedEdgeStream.csv','CollegeMsg_MonthInt_sortedEdgeStream.csv']
groundTruthFilesList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream_tests.csv','reality-call_MonthInt_sortedEdgeStream_tests.csv','soc-sign-bitcoinalpha_MonthInt_sortedEdgeStream_tests.csv','ia-enron-employees_MonthInt_sortedEdgeStream_tests.csv','CollegeMsg_MonthInt_sortedEdgeStream_tests.csv']
groundTruthpathList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream_Path_tests.csv','reality-call_MonthInt_sortedEdgeStream_Path_tests.csv','soc-sign-bitcoinalpha_MonthInt_sortedEdgeStream_Path_tests.csv','ia-enron-employees_MonthInt_sortedEdgeStream_Path_tests.csv','CollegeMsg_MonthInt_sortedEdgeStream_Path_tests.csv']

pivotList = [11,1,30,24,2] #most new edges
pivotList = [18,4,62,37,6] #last time instance


'''testFilesList = ['email-Eu-core-temporal_MonthInt_testSet.csv','reality-call_MonthInt_testSet.csv','CollegeMsg_MonthInt_testSet.csv']
trainFilesList = ['email-Eu-core-temporal_MonthInt_trainSet.csv','reality-call_MonthInt_trainSet.csv','CollegeMsg_MonthInt_trainSet.csv']
embeddingFilesList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream_multilens_TS_s_emb.tsv','reality-call_MonthInt_sortedEdgeStream_multilens_TS_s_emb.tsv','CollegeMsg_MonthInt_sortedEdgeStream_multilens_TS_s_emb.tsv']
shortedStreamFilesList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream.csv','reality-call_MonthInt_sortedEdgeStream.csv','CollegeMsg_MonthInt_sortedEdgeStream.csv']
groundTruthFilesList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream_tests.csv','reality-call_MonthInt_sortedEdgeStream_tests.csv','CollegeMsg_MonthInt_sortedEdgeStream_tests.csv']
groundTruthpathList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream_Path_tests.csv','reality-call_MonthInt_sortedEdgeStream_Path_tests.csv','CollegeMsg_MonthInt_sortedEdgeStream_Path_tests.csv']

pivotList = [11,1,13,108]'''


testFilesList = ['email-Eu-core-temporal_MonthInt_testSet.csv','reality-call_MonthInt_testSet.csv','soc-sign-bitcoinalpha_MonthInt_testSet.csv','ia-enron-employees_MonthInt_testSet.csv','CollegeMsg_MonthInt_testSet.csv']
trainFilesList = ['email-Eu-core-temporal_MonthInt_trainSet.csv','reality-call_MonthInt_trainSet.csv','soc-sign-bitcoinalpha_MonthInt_trainSet.csv','ia-enron-employees_MonthInt_trainSet.csv','CollegeMsg_MonthInt_trainSet.csv']
embeddingFilesList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream_multilens_TS_s_emb.tsv','reality-call_MonthInt_sortedEdgeStream_multilens_TS_s_emb.tsv','soc-sign-bitcoinalpha_MonthInt_sortedEdgeStream_multilens_TS_s_emb.tsv','ia-enron-employees_MonthInt_sortedEdgeStream_multilens_TS_s_emb.tsv','CollegeMsg_MonthInt_sortedEdgeStream_multilens_TS_s_emb.tsv']
shortedStreamFilesList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream.csv','reality-call_MonthInt_sortedEdgeStream.csv','soc-sign-bitcoinalpha_MonthInt_sortedEdgeStream.csv','ia-enron-employees_MonthInt_sortedEdgeStream.csv','CollegeMsg_MonthInt_sortedEdgeStream.csv']
groundTruthFilesList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream_tests.csv','reality-call_MonthInt_sortedEdgeStream_tests.csv','soc-sign-bitcoinalpha_MonthInt_sortedEdgeStream_tests.csv','ia-enron-employees_MonthInt_sortedEdgeStream_tests.csv','CollegeMsg_MonthInt_sortedEdgeStream_tests.csv']
groundTruthpathList = ['email-Eu-core-temporal_MonthInt_sortedEdgeStream_Path_tests.csv','reality-call_MonthInt_sortedEdgeStream_Path_tests.csv','soc-sign-bitcoinalpha_MonthInt_sortedEdgeStream_Path_tests.csv','ia-enron-employees_MonthInt_sortedEdgeStream_Path_tests.csv','CollegeMsg_MonthInt_sortedEdgeStream_Path_tests.csv']

pivotList = [18,4,62,37,6] #last time instance


#Values: Average, Hadamard, WeightedL1, WeightedL2
embeddingPassingList = ['Average', 'Hadamard', 'WeightedL1', 'WeightedL2']
binaryPassEmbedding = 'Average'
bestEmbedingPassingScore = 0
aucScoreList = []
aucDataset_dict = {}
pathsFoundDataset_dict = {}
pathsFound_dict = {}
pathsDistanceDatasetFound_dict = {}
pathsDistanceFound_dict = {}
dtatasetResults_dict = {}
for binaryPassEmbedding in embeddingPassingList:
    for test_file_path,train_file_path,shorted_file_path,embedding_file_path,ground_truth_file_path,ground_truth_path_file_path,pivotTime in zip(testFilesList,trainFilesList,shortedStreamFilesList,embeddingFilesList,groundTruthFilesList,groundTruthpathList,pivotList):
        gF = test_file_path.split('.')[0]
        print(gF)
        line = 'Dataset:\t'+str(gF)+'\n'
        infolist.append(line)
        if gF not in list(aucDataset_dict.keys()):
            aucScoreList = []
        else:
            aucScoreList = aucDataset_dict.get(gF)
        
        embeddings_dict = readEmbeddings(embedding_file_path)
        #print(embeddings_dict)
        graph_df = pd.read_csv(shorted_file_path,sep=',', usecols= ['source','target','time'])
        timesList = list(graph_df['time'].unique())

        #pivotTime = timesList[len(timesList)-1]
        records = graph_df.to_records(index=False)
        #edgeList = list(records)
        edgeList = []
        for source,target, time1 in zip(list(graph_df['source']),list(graph_df['target']),list(graph_df['time'])):
            source = int(source)
            target = int(target)
            time1 = int(time1)
            edge = (source,target,time1)
            edgeList.append(edge)
        traingraph_df = pd.read_csv(train_file_path,sep=',', usecols= ['source','target'])
        #records = traingraph_df.to_records(index=False)
        #edgesList = list(records)
        edgesList = []
        for source,target in zip(list(traingraph_df['source']),list(traingraph_df['target'])):
            source = int(source)
            target = int(target)
            edge = (source,target)
            edgesList.append(edge)
        G=nx.from_pandas_edgelist(traingraph_df)
        nodesList = []
        for node in list(list(embeddings_dict.keys())):
            node = int(node)
            nodesList.append(node)


        testgraph_df = pd.read_csv(test_file_path,sep=',', usecols= ['source','target'])
        #records = traingraph_df.to_records(index=False)
        #deletedEdgesList = list(records)
        deletedEdgesList = []
        for source,target in zip(list(testgraph_df['source']),list(testgraph_df['target'])):
            source = int(source)
            target = int(target)
            if source in nodesList and target in nodesList:
                edge = (source,target)
                deletedEdgesList.append(edge)
        #print('deletedEdgesList:\t'+str(deletedEdgesList))


        groundTruthGraph_df = pd.read_csv(ground_truth_file_path,sep=',', usecols= ['source','destination','prev_dist','future_dist'])
        #print(groundTruthGraph_df)
        testsourceNodesList = []
        test_dict = {}
        for node,target,prev_distance,future_distance in zip(list(groundTruthGraph_df['source']),list(groundTruthGraph_df['destination']),list(groundTruthGraph_df['prev_dist']),list(groundTruthGraph_df['future_dist'])):
            node = int(node)
            target = int(target)
            
            if not(prev_distance == math.inf):
                prev_distance = int(prev_distance)
            if not(future_distance == math.inf):
                future_distance = int(future_distance)
            #print(type(prev_distance))
            if node not in testsourceNodesList:
                testsourceNodesList.append(node)
                test_dict.update({node:{target:future_distance}})
            else:
                distance_dict = test_dict.get(node)
                distance_dict.update({target:future_distance})
                test_dict.update({node:distance_dict})
        

        

        groundTruthPathGraph_df = pd.read_csv(ground_truth_path_file_path,sep='\t', usecols= ['source','destination','prev_Path','future_Path'])

        #number of shortest temporal path tests 
        numberOfTests = int(len(list(groundTruthPathGraph_df['destination'])))

        testPath_dict = {}
        #print(list(groundTruthPathGraph_df['prev_Path']))
        distance_dict = {}
        for node,target,prev_path,future_path in zip(list(groundTruthPathGraph_df['source']),list(groundTruthPathGraph_df['destination']),list(groundTruthPathGraph_df['prev_Path']),list(groundTruthPathGraph_df['future_Path'])):
            node = int(node)
            target = int(target)

            future_path = future_path.split(',')
            future_path1 = []
            for i in range(1,len(future_path)):
                future_path1.append(int(future_path[i]))

            future_path = future_path1
            if type(prev_path) == float:
                prev_path = []
            else:
                prev_path = prev_path.split(',')
                prev_path1 = []
                for i in range(1,len(prev_path)):
                    prev_path1.append(int(prev_path[i]))
                prev_path = prev_path1
            '''if not(prev_distance == math.inf):
                prev_distance = int(prev_distance)
            if not(future_distance == math.inf):
                future_distance = int(future_distance)'''
            #print('prev_path')
            #print(type(prev_path))
            #print(list(prev_path))
            #print(type(prev_distance))
            '''if math.isnan(prev_path):
                prev_path = []'''
            #print(future_path)
            '''if math.isnan(future_path):
                future_path = []'''
            #print(future_path)
            if node not in list(testPath_dict.keys()):
                
                testPath_dict.update({node:{target:future_path}})
            else:
                distance_dict = testPath_dict.get(node)
                distance_dict.update({target:future_path})
                testPath_dict.update({node:distance_dict})
        
        #print(testPath_dict)
        print("Binary Operation:\t"+str(binaryPassEmbedding))
        if binaryPassEmbedding == 'Average':
            X_train,y_train = BOLEF.createTrainSetAverage(edgesList,nodesList,embeddings_dict,deletedEdgesList)
            X_train_array,y_train_array = np.array(X_train),np.array(y_train)

            X_test, y_test  = BOLEF.createTestSetAverage(deletedEdgesList,embeddings_dict,nodesList,edgesList,X_train)
            X_test_array,y_test_array = np.array(X_test),np.array(y_test)
        elif binaryPassEmbedding == 'Hadamard':
            X_train,y_train = BOLEF.createTrainSetHadamard(edgesList,nodesList,embeddings_dict,deletedEdgesList)
            X_train_array,y_train_array = np.array(X_train),np.array(y_train)

            X_test, y_test  = BOLEF.createTestSetHadamard(deletedEdgesList,embeddings_dict,nodesList,edgesList,X_train)
            X_test_array,y_test_array = np.array(X_test),np.array(y_test)
        elif binaryPassEmbedding == 'WeightedL1':
            X_train,y_train = BOLEF.createTrainSetWeightedL1(edgesList,nodesList,embeddings_dict,deletedEdgesList)
            X_train_array,y_train_array = np.array(X_train),np.array(y_train)

            X_test, y_test  = BOLEF.createTestSetWeightedL1(deletedEdgesList,embeddings_dict,nodesList,edgesList,X_train)
            X_test_array,y_test_array = np.array(X_test),np.array(y_test)
        elif binaryPassEmbedding == 'WeightedL2':
            X_train,y_train = BOLEF.createTrainSetWeightedL2(edgesList,nodesList,embeddings_dict,deletedEdgesList)
            X_train_array,y_train_array = np.array(X_train),np.array(y_train)

            X_test, y_test  = BOLEF.createTestSetWeightedL2(deletedEdgesList,embeddings_dict,nodesList,edgesList,X_train)
            X_test_array,y_test_array = np.array(X_test),np.array(y_test)

        start_time = time.time()
        clf = LogisticRegression(max_iter = 1000).fit(X_train_array, y_train_array)
        predictionList = clf.predict(X_test_array)
        end_time = time.time()
        time_needed = end_time - start_time
        print("Classifier Training Time:  "+str(time_needed))
        line = 'Binary Operation: Average\n'
        infolist.append(line)
        line = "Classifier Training Time:  "+str(time_needed)+'\n'
        infolist.append(line)
        '''print(y_test_array)
        print(predictionList)'''
        if len(y_test_array)>1:
            areaUnderCurve = roc_auc_score(y_test_array, predictionList, average = 'macro')
            print("Area Under Curve for average = macro:")
            print(areaUnderCurve)
            print('\n')
            line = "Area Under Curve for average:  "+str(areaUnderCurve)+'\n'
            infolist.append(line)
            aucScoreList.append((binaryPassEmbedding,areaUnderCurve))
            aucDataset_dict.update({gF:aucScoreList})
            



        #Temporal Test
        TPList = [] 
        TNList = []  
        FPList = []  
        FNList = []  





        print(binaryPassEmbedding)
        
        embeddingPassingType = binaryPassEmbedding
        #Prediction
        

        
        streamEdgeList = [] #should be shorted based on time
        vertexList = nodesList
        minTime = math.inf
        maxTime = 0
        for edge in edgeList:
            
            if edge[2] < pivotTime:
                streamEdgeList.append((edge[0],edge[1],edge[2]))
                '''if edge[0] not in vertexList:
                    vertexList.append(edge[0])
                if edge[1] not in vertexList:
                    vertexList.append(edge[1])'''
                if minTime> edge[2]:
                    minTime = edge[2]
                if maxTime< edge[2]:
                    maxTime = edge[2]

        
        timeInterval = [minTime,pivotTime]
        sortedEdgeStream = sorted(streamEdgeList, key=lambda x: float(x[2]))
        sortedReverseEdgeStream = sorted(streamEdgeList, key=lambda x: float(x[2]),reverse=True)

        
        #True
    
        streamEdgeList = [] #should be shorted based on time

        minTime = math.inf
        maxTime = 0
        

        for edge in edgeList:
            if edge[0] in list(embeddings_dict.keys()) and edge[1] in list(embeddings_dict.keys()):
                streamEdgeList.append((edge[0],edge[1],edge[2]))
                '''if edge[0] not in vertexList:
                    vertexList.append(edge[0])
                if edge[1] not in vertexList:
                    vertexList.append(edge[1])'''
                if minTime> edge[2]:
                    minTime = edge[2]
                if maxTime< edge[2]:
                    maxTime = edge[2]

        #real_timeInterval = [minTime,maxTime]
        true_sortedEdgeStream = sorted(streamEdgeList, key=lambda x: float(x[2]))


        #numTests = 100

        print("Max Time Interval is:   "+str((minTime,maxTime)))
        #test_vetrex = vertexList[len(vertexList)-2]
        MSEAllStartNodesList = []
        start_time = time.time()
        testMetric = 0
        averagepredictionTime = 0
        rightPathFound = 0
        rightDistancePathFound = 0
        longerThanOnePathsList = []
        longerThanOnePaths = 0
        wrongPathFound = 0
        wrongDistancePathFound = 0
        for test_vetrex in testsourceNodesList:
            #test_vetrex = 2384
            if test_vetrex in list(embeddings_dict.keys()):
                groundTruthPath_dict = testPath_dict.get(test_vetrex)
                prediction_true_dict = test_dict.get(test_vetrex)
                #print(prediction_true_dict)
                destinationNodes  = list(prediction_true_dict.keys())
                
                startTime = time.time()
                prediction_dict = predictProbaActualShortestPathDistance(clf,embeddings_dict,test_vetrex,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationNodes)
                #print(len(prediction_dict.keys()))
                endTime = time.time()
                #print(prediction_dict)
                predictionTime = endTime - startTime
                averagepredictionTime +=predictionTime
                max_pathNotinf = 0
                for vertex in list(prediction_dict.keys()):
                    fastest_path_distance,fastest_path = prediction_dict.get(vertex)
                    if fastest_path_distance> max_pathNotinf and not(fastest_path_distance==math.inf):
                        max_pathNotinf = fastest_path_distance

                


                

                
                y_prediction = []
                y_truth = []
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                for vertex in list(destinationNodes):
                        
                        '''print(vertex)
                        print(list(prediction_dict.keys()))'''
                        if vertex in list(prediction_dict.keys()):
                            predicted_fastest_path_distance = prediction_dict.get(vertex)[0]
                            predicted_fastest_path = prediction_dict.get(vertex)[1]
                        else:
                            predicted_fastest_path_distance = math.inf
                            predicted_fastest_path = []

                        print('for source:\t'+str(test_vetrex)+'\tfor destination:\t'+str(vertex)+'\tpredicted stp:\t'+str(predicted_fastest_path_distance))
                        true_fastest_path_distance = int(prediction_true_dict.get(vertex))
                        print('for source:\t'+str(test_vetrex)+'\tfor destination:\t'+str(vertex)+'\ttrue stp:\t'+str(true_fastest_path_distance))
                        true_fastest_path = groundTruthPath_dict.get(vertex)
                        '''print('predicted path')
                        print(predicted_fastest_path)
                        print('true path')
                        print(true_fastest_path)'''
                        '''print('true_fastest_path_distance')
                        print(true_fastest_path_distance)
                        print('predicted_fastest_path_distance')
                        print(predicted_fastest_path_distance)'''
                        if (not(true_fastest_path_distance == math.inf) and  not(predicted_fastest_path_distance == math.inf)):
                            if set(predicted_fastest_path) == set(true_fastest_path):
                                rightPathFound +=1
                                if len(predicted_fastest_path)-1>1:
                                    longerThanOnePaths +=1
                                    longerThanOnePathsList.append(predicted_fastest_path)
                            else:
                                wrongPathFound +=1

                            if len(predicted_fastest_path) == len(true_fastest_path):
                                rightDistancePathFound +=1
                            else:
                                wrongDistancePathFound +=1
                            y_prediction.append(predicted_fastest_path_distance)
                            y_truth.append(true_fastest_path_distance)
                            TP += 1
                        if (true_fastest_path_distance == math.inf and predicted_fastest_path_distance == math.inf):
                            print('True Negative')
                            TN += 1
                        if (not(true_fastest_path_distance == math.inf) and predicted_fastest_path_distance == math.inf):
                            print('False Negative')
                            FN += 1
                        if (true_fastest_path_distance == math.inf and not(predicted_fastest_path_distance == math.inf)):
                            print('False Positive')
                            FP +=1
                        
                
                '''print('y_prediction')
                print(y_prediction)
                print('\n')
                print('y_truth')
                print(y_truth)
                print('\n')'''
                '''print('TP:\t'+str(TP))
                print('TN:\t'+str(TN))
                print('FP:\t'+str(FP))
                print('FN:\t'+str(FN))'''
                #if len(y_truth)>0 or len(y_prediction)>0:
                allValues = TP + TN + FP + FN
                TP = TP / allValues
                TN = TN / allValues
                FP = FP / allValues
                FN = FN / allValues
                TPList.append(TP)
                TNList.append(TN)  
                FPList.append(FP) 
                FNList.append(FN)  
                #print(len(list(prediction_dict.keys()))==len(list(prediction_true_dict.keys())))
                if len(y_truth)>0 and len(y_prediction)>0:
                            meanSquaredError = mean_squared_error(y_truth,y_prediction)
                elif len(y_truth)==0 and len(y_prediction)==0:
                            meanSquaredError =0
                elif len(y_truth)>0 and len(y_prediction)==0:
                        meanSquaredError =0
                        for yt in y_truth:
                                meanSquaredError+=yt**2
                        meanSquaredError = meanSquaredError/len(y_truth)
                elif len(y_truth)==0 and len(y_prediction)>0:
                            meanSquaredError =0
                            for yp in y_prediction:
                                meanSquaredError+=yp**2
                            meanSquaredError = meanSquaredError/len(y_prediction)


                print('Prediction Mean Squared Error (classifier = Average):  '+str(meanSquaredError))
                MSEAllStartNodesList.append(meanSquaredError)
            end_time = time.time()
            time_needed = end_time - start_time
            averagepredictionTime = averagepredictionTime / len(testsourceNodesList)
            print('Average Prediction Time:\t'+str(averagepredictionTime))
            print("Computed MSE in:  "+str(time_needed))
            finalMeanSquaredError = 0
            if len(MSEAllStartNodesList)>0:
                for meanSquaredError in MSEAllStartNodesList:
                    finalMeanSquaredError += meanSquaredError
                finalMeanSquaredError = finalMeanSquaredError/ len(MSEAllStartNodesList)
            print('Prediction Mean Squared Error for all starting Nodes(classifier = Average):  '+str(finalMeanSquaredError))
            allValues = len(TPList) + len(TNList) + len(FPList) + len(FNList)

            finalTP = 0
            for TP in TPList:
                finalTP += TP

            finalTN = 0
            for TN in TNList:
                finalTN += TN

            finalFP = 0
            for FP in FPList:
                finalFP += FP

            finalFN = 0
            for FN in FNList:
                finalTP += FN

            '''print('TPList:\t'+str(TPList))
            print('TNList:\t'+str(TNList))
            print('FPList:\t'+str(FPList))
            print('FNList:\t'+str(FNList))

            print('finalTP:\t'+str(finalTP))
            print('finalTN:\t'+str(finalTN))
            print('finalFP:\t'+str(finalFP))
            print('finalFN:\t'+str(finalFN))'''
            if len(TPList)>0:
                TP = finalTP / len(TPList)
            if len(TNList)>0:
                TN = finalTN / len(TNList)
            if len(FPList)>0:
                FP = finalFP / len(FPList)
            if len(FNList)>0:
                FN = finalFN / len(FNList)
            print('TP: '+str(TP)+"  TN:  "+str(TN)+ ' FP: '+str(FP)+' FN: '+str(FN))


            line = 'Binary Operation: Average\n'
            infolist.append(line)
            line = 'Average Prediction Time:\t'+str(averagepredictionTime)+'\n'
            infolist.append(line)
            line = "Computed MSE in:  "+str(time_needed)+'\n'
            infolist.append(line)
            line = 'Prediction Mean Squared Error for all starting Nodes(classifier = Average):  '+str(finalMeanSquaredError)+'\n'
            infolist.append(line)
            line = 'TP: '+str(TP)+"  TN:  "+str(TN)+ ' FP: '+str(FP)+' FN: '+str(FN)+'\n'
            infolist.append(line)
            print('Correctly Found Paths:\t'+str(rightPathFound)+'\n')
            print('Wrongly Found Paths:\t'+str(wrongPathFound)+'\n')
            print('Longer than distance one paths found:\t'+str(longerThanOnePaths))
        correctFoundPathsPerc = rightPathFound/numberOfTests
        line = "percentage of paths found correctly"+str(correctFoundPathsPerc)
        print(line)


        wrongFoundPathsPerc = wrongPathFound/numberOfTests
        line = "percentage of paths found wrongly"+str(wrongFoundPathsPerc)
        print(line)


        correctDistanceFoundPathsPerc = rightDistancePathFound/numberOfTests
        line = "percentage of distance of paths found correctly"+str(correctDistanceFoundPathsPerc)
        print(line)

        wrongDistanceFoundPathsPerc = wrongDistancePathFound/numberOfTests
        line = "percentage of distance of paths found wrongly"+str(wrongDistanceFoundPathsPerc)
        print(line)
        if gF not in list(pathsFoundDataset_dict.keys()):

            pathsFound_dict = {}
            pathsDistanceFound_dict = {}
        else:
            pathsFound_dict = pathsFoundDataset_dict.get(gF)
            pathsDistanceFound_dict = pathsDistanceDatasetFound_dict.get(gF)
        pathsDistanceFound_dict.update({binaryPassEmbedding:[("percentage of distance of paths found correctly",correctDistanceFoundPathsPerc),("percentage of distance of paths found wrongly",wrongDistanceFoundPathsPerc)]})
        pathsDistanceDatasetFound_dict.update({gF:pathsDistanceFound_dict})
        pathsFound_dict.update({binaryPassEmbedding:[("percentage of paths found correctly",correctFoundPathsPerc),("percentage of paths found wrongly",wrongFoundPathsPerc)]})
        pathsFoundDataset_dict.update({gF:pathsFound_dict})
    for path in longerThanOnePathsList:
        print('source:\t'+str(path[0])+'\tdestination:\t'+str(path[len(path)-1])+'\tpath:\t'+str(path))

infodFileName = gF+'__info.txt'
infodFile = open(infodFileName, "w")
for line in infolist:
        infodFile.write(line)
infodFile.close()


print(aucScoreList)
print(pathsFound_dict)
print(aucDataset_dict)
print(pathsFoundDataset_dict)
resultsList = []
for dataset in list(aucDataset_dict.keys()):
    line = 'Dataset:\t'+str(dataset)+'\n'
    print(line)
    resultsList.append(line)
    aucScoreList = aucDataset_dict.get(dataset)
    pathsFound_dict = pathsFoundDataset_dict.get(dataset)
    pathsDistanceFound_dict = pathsDistanceDatasetFound_dict.get(dataset)
    for value in aucScoreList:
        embeddingPassingMethod = value[0]
        line = 'Embedding passing Method:\t'+embeddingPassingMethod+'\n'
        resultsList.append(line)
        aucScore = value[1]
        line = 'AUC Score:\t'+str(aucScore)+'\n'
        resultsList.append(line)
        percentagesList = pathsFound_dict.get(embeddingPassingMethod)
        correctlyFound = percentagesList[0]
        line =correctlyFound[0]+'\t' +str(correctlyFound[1])+'\n'
        resultsList.append(line)
        wronglylyFound = percentagesList[1]
        line =wronglylyFound[0]+'\t' +str(wronglylyFound[1])+'\n'
        resultsList.append(line)
        
        percentagesList = pathsDistanceFound_dict.get(embeddingPassingMethod)
        correctlyFound = percentagesList[0]
        line =correctlyFound[0]+'\t' +str(correctlyFound[1])+'\n'
        resultsList.append(line)
        wronglylyFound = percentagesList[1]
        line =wronglylyFound[0]+'\t' +str(wronglylyFound[1])+'\n'
        resultsList.append(line)
    line = '\n\n\n'
    resultsList.append(line)

resultsFileName = 'Evaluation_Results.txt'
resultsFile = open(resultsFileName, "w")
for line in resultsList:
        resultsFile.write(line)
resultsFile.close()