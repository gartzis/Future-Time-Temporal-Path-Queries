from networkx.algorithms.operators.unary import reverse
import torch
import random
from FileReader import file_reader
import PerformanceMetrics as PM
import binaryOperatorsForLearningEdgeFeatures as BOLEF
import read_Edge_Stream as EdgeStream
from predictActualDistancePath import predictProbaActualShortestPathDistance,predictLinkLikeJodie
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score,mean_squared_error, recall_score, precision_score
import math
import time
import pandas as pd
import networkx as nx
import statistics

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
                
                embList.append(emb_feature)
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
            
            embList.append(emb_feature)
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
        
        embedding1 = embeddings_dict.get(edge[0])
        embedding2 = embeddings_dict.get(edge[1])
        embList = []
        for i in range(len(embedding1)):
            emb_feature =(embedding1[i]*embedding2[i])
            
            embList.append(emb_feature)
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
            
            embList.append(emb_feature)
        X_test.append(embList)
        addedembeddings+=1
        
        
        y_test.append(0)
        if(addedembeddings>= alreadyaddedembeddings):
            break
    return X_test, y_test










testFilesList = ['CollegeMsg_TimestampZero_testSet.csv']
trainFilesList = ['CollegeMsg_TimestampZero_90trainSet.csv']
embeddingFilesList = ['embeddings_CollegeMsg_TimestampZero_80trainSet.emb']
embedding90FilesList = ['embeddings_CollegeMsg_TimestampZero_90trainSet.emb']
#embeddingFilesList = ['CollegeMsg_TimestampZero_multilens_TS_s_emb.tsv']

shortedStreamFilesList = ['CollegeMsg_TimestampZero_sortedEdgeStream.csv']
groundTruthFilesList = ['CollegeMsg_TimestampZero_tests.csv']
groundTruthpathList = ['CollegeMsg_TimestampZero_Path_tests.csv']

pivotList = [8949520] #last time instance


testFilesList = ['ia-enron-employees_TimestampZero_testSet.csv']
trainFilesList = ['ia-enron-employees_TimestampZero_90trainSet.csv']
embeddingFilesList = ['embeddings_ia-enron-employees_TimestampZero_80trainSet.emb']
embedding90FilesList = ['embeddings_ia-enron-employees_TimestampZero_90trainSet.emb']
#embeddingFilesList = ['ia-enron-employees_TimestampZero_multilens_TS_s_emb.tsv']
shortedStreamFilesList = ['ia-enron-employees_TimestampZero_sortedEdgeStream.csv']
groundTruthpathList = ['ia-enron-employees_TimestampZero_Path_tests_FutureSubgraph_Path_tests.tsv']
groundTruthFilesList = ['ia-enron-employees_TimestampZero_Path_tests_FutureSubgraph_tests.csv']

pivotList = [85238942]


'''embeddingFilesList = ['embeddings_soc-sign-bitcoinalpha_TimestampZero_80trainSet.emb']
embedding90FilesList = ['embeddings_soc-sign-bitcoinalpha_TimestampZero_90trainSet.emb']
#embeddingFilesList = ['soc-sign-bitcoinalpha_TimestampZero_multilens_TS_s_emb.tsv']
testFilesList = ['soc-sign-bitcoinalpha_TimestampZero_testSet.csv']
trainFilesList = ['soc-sign-bitcoinalpha_TimestampZero_90trainSet.csv']
shortedStreamFilesList = ['soc-sign-bitcoinalpha_TimestampZero_sortedEdgeStream.csv']
groundTruthFilesList = ['soc-sign-bitcoinalpha_TimestampZero__Path_tests_FutureSubgraph_tests.csv']
groundTruthpathList = ['soc-sign-bitcoinalpha_TimestampZero_Path_tests_FutureSubgraph_Path_tests.tsv']

pivotList = [106441200]'''











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
infolist = []

for test_file_path,train_file_path,shorted_file_path,embedding_file_path,embedding90_file_path,ground_truth_file_path,ground_truth_path_file_path,pivotTime in zip(testFilesList,trainFilesList,shortedStreamFilesList,embeddingFilesList,embedding90FilesList,groundTruthFilesList,groundTruthpathList,pivotList):
        gF = test_file_path.split('.')[0]
        print(gF)
        line = 'Dataset:\t'+str(gF)+'\n'
        infolist.append(line)
        if gF not in list(aucDataset_dict.keys()):
            aucScoreList = []
        else:
            aucScoreList = aucDataset_dict.get(gF)
        
        embeddings_dict = readEmbeddings(embedding_file_path)
        
        graph_df = pd.read_csv(shorted_file_path,sep=',', usecols= ['source','target','time'])
        timesList = list(graph_df['time'].unique())

        
        records = graph_df.to_records(index=False)
        
        edgeList = []
        nodesInPast = []
        for source,target, time1 in zip(list(graph_df['source']),list(graph_df['target']),list(graph_df['time'])):
            source = int(source)
            target = int(target)
            time1 = int(time1)
            if source in list(embeddings_dict.keys()) and source in list(embeddings_dict.keys()):
                edge = (source,target,time1)
                edgeList.append(edge)
                if time1 < pivotTime:
                    if source not in nodesInPast:
                        nodesInPast.append(source)
                    if target not in nodesInPast:
                        nodesInPast.append(target)
        traingraph_df = pd.read_csv(train_file_path,sep=',', usecols= ['source','target'])
        
        
        edgesList = []
        for source,target in zip(list(traingraph_df['source']),list(traingraph_df['target'])):
            source = int(source)
            target = int(target)
            if source in list(embeddings_dict.keys()) and source in list(embeddings_dict.keys()):
                edge = (source,target)
                edgesList.append(edge)
        G=nx.from_pandas_edgelist(traingraph_df)
        nodesList = []
        for node in list(list(embeddings_dict.keys())):
            node = int(node)
            nodesList.append(node)


        testgraph_df = pd.read_csv(test_file_path,sep=',', usecols= ['source','target'])
        
        
        deletedEdgesList = []
        for source,target in zip(list(testgraph_df['source']),list(testgraph_df['target'])):
            source = int(source)
            target = int(target)
            if source in list(embeddings_dict.keys()) and source in list(embeddings_dict.keys()):
                if source in nodesList and target in nodesList:
                    edge = (source,target)
                    deletedEdgesList.append(edge)
        


        groundTruthGraph_df = pd.read_csv(ground_truth_file_path,sep=',', usecols= ['source','destination','prev_dist','future_dist'])
        
        testsourceNodesList = []
        test_dict = {}
        for node,target,prev_distance,future_distance in zip(list(groundTruthGraph_df['source']),list(groundTruthGraph_df['destination']),list(groundTruthGraph_df['prev_dist']),list(groundTruthGraph_df['future_dist'])):
            node = int(node)
            target = int(target)
            
            if not(prev_distance == math.inf):
                prev_distance = int(prev_distance)
            if not(future_distance == math.inf):
                future_distance = int(future_distance)
            
            if node in nodesInPast and target in nodesInPast:
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
        
        distance_dict = {}
        prevPath_dict = {}
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
            if node in nodesInPast and target in nodesInPast:
                if node not in list(prevPath_dict.keys()):
                    
                    prevPath_dict.update({node:{target:prev_path}})
                else:
                    distance_dict = prevPath_dict.get(node)
                    distance_dict.update({target:prev_path})
                    prevPath_dict.update({node:distance_dict})

                if node not in list(testPath_dict.keys()):
                    
                    testPath_dict.update({node:{target:future_path}})
                else:
                    distance_dict = testPath_dict.get(node)
                    distance_dict.update({target:future_path})
                    testPath_dict.update({node:distance_dict})
        
        
        for binaryPassEmbedding in embeddingPassingList:
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
            line = 'Binary Operation:\t'+str(binaryPassEmbedding)+'\n'
            infolist.append(line)
            line = "Classifier Training Time:  "+str(time_needed)+'\n'
            infolist.append(line)
            if len(y_test_array)>1:
                areaUnderCurve = roc_auc_score(y_test_array, predictionList, average = 'macro')
                print("Area Under Curve for average = macro:",areaUnderCurve)
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





            
            
            embeddingPassingType = binaryPassEmbedding
            #Prediction
            

            
            streamEdgeList = [] #should be shorted based on time
            vertexList = nodesList
            minTime = math.inf
            maxTime = 0
            for edge in edgeList:
                
                if edge[2] < pivotTime:
                    streamEdgeList.append((edge[0],edge[1],edge[2]))
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
                    if minTime> edge[2]:
                        minTime = edge[2]
                    if maxTime< edge[2]:
                        maxTime = edge[2]

            
            true_sortedEdgeStream = sorted(streamEdgeList, key=lambda x: float(x[2]))


            #numTests = 100

            
            
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
            predictedDistanceList = []
            embeddings_dict = readEmbeddings(embedding90_file_path)
            predictionTimeList = []
            for test_vetrex in testsourceNodesList:
                
                if test_vetrex in list(embeddings_dict.keys()):
                    
                    groundTruthPath_dict = testPath_dict.get(test_vetrex)
                    prediction_true_dict = test_dict.get(test_vetrex)                    
                    destinationNodes  = list(prediction_true_dict.keys())
                    prediction_dict = {}
                    previousPathSpecificSource_dict = prevPath_dict.get(test_vetrex)
                    for destinationVertex in list(previousPathSpecificSource_dict.keys()):
                        if destinationVertex in list(embeddings_dict.keys()):
                            previousPathList = previousPathSpecificSource_dict.get(destinationVertex)
                            startTime = time.time()
                            prediction = predictLinkLikeJodie(clf,previousPathList,destinationVertex,embeddings_dict,embeddingPassingType)
                            endTime = time.time()
                            prediction_dict.update({destinationVertex:prediction})
                            predictionTime = endTime - startTime
                            predictionTimeList.append(predictionTime)                                  
                    max_pathNotinf = 0
                    for vertex in list(prediction_dict.keys()):
                        fastest_path_probability,fastest_path_distance,fastest_path = prediction_dict.get(vertex)
                        if fastest_path_distance> max_pathNotinf and not(fastest_path_distance==math.inf):
                            max_pathNotinf = fastest_path_distance

                    


                    

                    
                    y_prediction = []
                    y_truth = []
                    TP = 0
                    TN = 0
                    FP = 0
                    FN = 0
                    
                    for vertex in list(destinationNodes):
                            
                            if vertex in list(prediction_dict.keys()):
                                predicted_fastest_path_distance = prediction_dict.get(vertex)[1]
                                predicted_fastest_path = prediction_dict.get(vertex)[2]
                            else:
                                predicted_fastest_path_distance = math.inf
                                predicted_fastest_path = []
                            if predicted_fastest_path_distance != math.inf:
                                predictedDistanceList.append(predicted_fastest_path_distance)
                            
                            true_fastest_path_distance = int(prediction_true_dict.get(vertex))
                           
                            true_fastest_path = groundTruthPath_dict.get(vertex)
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
                                
                                TN += 1
                            if (not(true_fastest_path_distance == math.inf) and predicted_fastest_path_distance == math.inf):
                                
                                FN += 1
                            if (true_fastest_path_distance == math.inf and not(predicted_fastest_path_distance == math.inf)):
                               FP +=1
                            
                    
                    allValues = TP + TN + FP + FN
                    TP = TP / allValues
                    TN = TN / allValues
                    FP = FP / allValues
                    FN = FN / allValues
                    TPList.append(TP)
                    TNList.append(TN)  
                    FPList.append(FP) 
                    FNList.append(FN)  
                    
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


                    
                    MSEAllStartNodesList.append(meanSquaredError)
                end_time = time.time()
                time_needed = end_time - start_time
                
                
            finalMeanSquaredError = 0
            if len(MSEAllStartNodesList)>0:
                for meanSquaredError in MSEAllStartNodesList:
                    finalMeanSquaredError += meanSquaredError
                finalMeanSquaredError = finalMeanSquaredError/ len(MSEAllStartNodesList)
            print('Prediction Mean Squared Error for all starting Nodes(classifier = Average):  '+str(finalMeanSquaredError))
            allValues = len(TPList) + len(TNList) + len(FPList) + len(FNList)


            
            averagepredictionTime = sum(predictionTimeList) / len(predictionTimeList)
            line = 'Average time needed for the prediction:\t'+str(averagepredictionTime)+'\n'
            infolist.append(line)

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

            
            if len(TPList)>0:
                TP = finalTP / len(TPList)
            if len(TNList)>0:
                TN = finalTN / len(TNList)
            if len(FPList)>0:
                FP = finalFP / len(FPList)
            if len(FNList)>0:
                FN = finalFN / len(FNList)
            
            maxPredictedDistance = max(predictedDistanceList)
            minpredictedDistance = min(predictedDistanceList)
            avgPredictedDistance = statistics.mean(predictedDistanceList)

            line = 'Binary Operation:\t'+ str(binaryPassEmbedding) +'\n'
            infolist.append(line)
            line = 'Average Prediction Time:\t'+str(averagepredictionTime)+'\n'
            infolist.append(line)
            line = "Computed MSE in:  "+str(time_needed)+'\n'
            infolist.append(line)
            line = 'Prediction Mean Squared Error for all starting Nodes(classifier = Average):  '+str(finalMeanSquaredError)+'\n'
            infolist.append(line)
            line = 'TP: '+str(TP)+"  TN:  "+str(TN)+ ' FP: '+str(FP)+' FN: '+str(FN)+'\n'
            infolist.append(line)
            line = 'Maximum Predicted Distance:\t'+str(maxPredictedDistance)+'\n'
            infolist.append(line)
            line = 'Minimum Predicted Distance:\t'+str(minpredictedDistance)+'\n'
            infolist.append(line)
            line = 'Average Predicted Distance:\t'+str(avgPredictedDistance)+'\n'
            infolist.append(line)
            print('Correctly Found Paths:\t'+str(rightPathFound)+'\n')
            print('Wrongly Found Paths:\t'+str(wrongPathFound)+'\n')
            print('Longer than distance one paths found:\t'+str(longerThanOnePaths))
            correctFoundPathsPerc = rightPathFound/numberOfTests
            line = "percentage of paths found correctly:\t"+str(correctFoundPathsPerc)+'\n'
            infolist.append(line)
            print(line)


            wrongFoundPathsPerc = wrongPathFound/numberOfTests
            line = "percentage of paths found wrongly:\t"+str(wrongFoundPathsPerc)+'\n'
            infolist.append(line)
            print(line)


            correctDistanceFoundPathsPerc = rightDistancePathFound/numberOfTests
            line = "percentage of distance of paths found correctly:\t"+str(correctDistanceFoundPathsPerc)+'\n'
            infolist.append(line)
            print(line)

            wrongDistanceFoundPathsPerc = wrongDistancePathFound/numberOfTests
            line = "percentage of distance of paths found wrongly:\t"+str(wrongDistanceFoundPathsPerc)+'\n'
            infolist.append(line)
            line = '\n\n\n'
            infolist.append(line)
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

resultsFileName = str(gF)+'_Multilens_Evaluation_Results.txt'
resultsFile = open(resultsFileName, "w")
for line in resultsList:
        resultsFile.write(line)
resultsFile.close()