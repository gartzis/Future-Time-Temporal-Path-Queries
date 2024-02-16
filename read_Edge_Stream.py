import binaryOperatorsForLearningEdgeFeatures as BOLEF
import networkx as nx
import math
import random

def computeEarliestArivalTime(sortedEdgeStream,vertex,vertexList,timeInterval):
    time_dict = {}
    #path_dict = {}
    pathList = []
    time_dict.update({vertex:timeInterval[0]})
    #path_dict.update({vertex:[]})
    for v in vertexList:
        if not(v==vertex):
            time_dict.update({v:math.inf})
            #path_dict.update({v:[]})
    for incoming_edge in sortedEdgeStream:
        prev_vertex = incoming_edge[0]
        prev_time = time_dict.get(prev_vertex)
        if incoming_edge[2]<=timeInterval[1] and incoming_edge[2]>prev_time:
            current_vertex = incoming_edge[1]
            current_time = time_dict.get(current_vertex)
            #pathList = path_dict.get(current_vertex)
            pathList.append(current_vertex)
            #path_dict.update({current_vertex:pathList}) 
            if incoming_edge[2]<current_time:
                time_dict.update({current_vertex:incoming_edge[2]})
 
        elif (incoming_edge[2]>timeInterval[1]):
            break
    return time_dict, pathList

def computeLatestDepartureTime(sortedEdgeStream,vertex,vertexList,timeInterval):# sortedEdgeStream: in reverse
    time_dict = {}
    time_dict.update({vertex:timeInterval[1]})
    for v in vertexList:
        if not(v==vertex):
            time_dict.update({v:-math.inf})
    for edge in sortedEdgeStream:
        if edge[2]>= timeInterval[0]:
            
            current_vertex = edge[1]
            current_time = time_dict.get(current_vertex)
            if edge[2]<= current_time:
                prev_vertex = edge[0]
                prev_time = time_dict.get(prev_vertex)
                if edge[2]>prev_time:
                    time_dict.update({prev_vertex:edge[2]})
        else:
            break
    return time_dict

def computeFastestPathDuration(sortedEdgeStream,vertex,vertexList,timeInterval):#multiple passess
    fastestPath_dict = {}
    fastestPath_dict.update({vertex:0})
    S = []
    for v in vertexList:
        if not(v==vertex):
            fastestPath_dict.update({v:math.inf})
    for edge in sortedEdgeStream:
        
            if (edge[0]== vertex):
                if edge[2]>= timeInterval[0] and edge[2]<=timeInterval[1]:
                    S.append(edge[2])
    #print(len(S))
    for time in S:
        timeInterval1 = [time,timeInterval[1]]
        time_dict, pathList = computeEarliestArivalTime(sortedEdgeStream,vertex,vertexList,timeInterval1)
        for key in list(time_dict.keys()):
            time1 = time_dict.get(key)

            fp = fastestPath_dict.get(key)
            timeDiff = time1 - time
            upValue = min(fp,timeDiff)
            fastestPath_dict.update({key:upValue})
    return fastestPath_dict

def computeFastestPathDurationOnePass(sortedEdgeStream,vertex,vertexList,timeInterval):#multiple passess
    vertexToLv_dict ={}
    startingTime_dict = {}
    arrivalTime_dict = {}
    fastestPath_dict = {}
    startTime_dict = {}
    fastestPath_dict.update({vertex:0})

    for v in vertexList:
        if not(v==vertex):
            fastestPath_dict.update({v:math.inf})


    for v in vertexList:
        vertexToLv_dict.update({v:[]})
        startingTime_dict.update({v:[]})
        arrivalTime_dict.update({v:[]})
        startTime_dict.update({v:[]})
    for edge in sortedEdgeStream:
        startingTimeList = startingTime_dict.get(edge[0])
        arrivalTimeList = arrivalTime_dict.get(edge[1])
        startingTimeList.append(edge[2])
        arrivalTimeList.append(edge[2])
        startingTime_dict.update({edge[0]:startingTimeList})
        arrivalTime_dict.update({edge[1]:arrivalTimeList})

    for v in vertexList:
        Lv = vertexToLv_dict.get(v)
        startTimeList = startingTime_dict.get(v)
        sTList =startTime_dict.get(v)
        arrivalTimeList = arrivalTime_dict.get(v)
        for time1 in startTimeList:
            for time2 in arrivalTimeList:
                if time1<= time2:
                    if time1 not in sTList:
                        sTList.append(time1)
                    Lv.append((time1,time2))
        Lv = sorted(Lv, key=lambda x: float(x[1]))
        startTime_dict.update({v:sTList})
        vertexToLv_dict.update({v:Lv})
        
    for edge in sortedEdgeStream:
        if edge[2]>= timeInterval[0] and edge[2]<= timeInterval[1]:
            Lv = vertexToLv_dict.get(edge[0])
            if edge[0] == vertex:
                if (edge[2],edge[2]) not in Lv:
                    Lv.append((edge[2],edge[2]))
                    Lv = sorted(Lv, key=lambda x: float(x[1]))
            max_atime = 0
            chosen_stime =0 
            for stime,atime in Lv:
                if atime> max_atime and atime<=edge[2]:
                    chosen_stime = stime
                    max_atime = atime
            arrivalTime = edge[2]
            Lv = vertexToLv_dict.get(edge[1])
            flag = True
            for i in range(len(Lv)):
                timeSet = Lv[i]
                if timeSet[0] == chosen_stime:
                    Lv[i] = (chosen_stime,arrivalTime)
                    Lv = sorted(Lv, key=lambda x: float(x[1]))
                    flag = False
                    break
            if (flag):
                Lv.appen((chosen_stime,arrivalTime))
                Lv = sorted(Lv, key=lambda x: float(x[1]))
            dominatedElementsList = []
            for i in range(len(Lv)-1):
                s1,a1 = Lv[i]
                s2,a2 = Lv[i+1]
                if (s1>s2 and  a1 <= a2) or (s1 == s2 and a1 < a2):
                    dominatedElementsList.append(Lv[i+1])
                elif (s1<s2 and  a1 >= a2) or (s1 == s2 and a1 > a2):
                    dominatedElementsList.append(Lv[i])
            for element in dominatedElementsList:
                if element in Lv:
                    Lv.remove(element)
            fp = fastestPath_dict.get(edge[1])
            if arrivalTime - chosen_stime < fp:
                fp = arrivalTime - chosen_stime
                fastestPath_dict.update({edge[1]:fp})
        elif edge[2] > timeInterval[1]:
            break
    return  fastestPath_dict


def computeShortestPathDistance(sortedEdgeStream,vertex,vertexList,timeInterval):
    vertexToLv_dict ={}
    startingTime_dict = {}
    arrivalTime_dict = {}
    shortestPath_dict = {}
    startTime_dict = {}
    shortestPath_dict.update({vertex:0})

    for v in vertexList:
        if not(v==vertex):
            shortestPath_dict.update({v:math.inf})
        vertexToLv_dict.update({v:[]})

    '''for edge in sortedEdgeStream:
        if edge[1] in vertexList:
            if edge[0] == vertex:
                Lv = vertexToLv_dict.get(edge[0])
                Lv.append((0,edge[2]))
                Lv = sorted(Lv, key=lambda x: float(x[0]))
                vertexToLv_dict.update({edge[0]:Lv})'''
    foundEdge = False
    for edge in sortedEdgeStream:
        #print(edge)
        if edge[0] in vertexList and edge[1] in vertexList:
            if edge[0] == vertex:
                foundEdge = True
        if foundEdge == True:
            
            if edge[2]>= timeInterval[0] and edge[2]< timeInterval[1]:
                
                
                if edge[0] == vertex:
                    Lv = vertexToLv_dict.get(edge[0])
                    if (0,edge[2]) not in Lv:
                        Lv.append((0,edge[2]))
                        Lv = sorted(Lv, key=lambda x: float(x[0]))
                        vertexToLv_dict.update({edge[0]:Lv})
                max_atime = 0
                chosen_distance =0
                
                Lv = vertexToLv_dict.get(edge[0])
                for distance,atime in Lv:
                    #print(Lv)
                    if atime> max_atime and atime<=edge[2]:
                        chosen_distance = distance
                        #print(chosen_distance)
                        #print(max_atime)
                        max_atime = atime
                        
                #print('chosen_distance:\t'+str(chosen_distance))
                chosen_distance += 1
                atime = edge[2]
                #print('new chosen_distance:\t'+str(chosen_distance))
                #print('new time:\t'+str(atime))
                Lv = vertexToLv_dict.get(edge[1])
                #print('Lv of edge[1]\t'+str(Lv))
                flag = True
                for i in range(len(Lv)):
                    timeSet = Lv[i]
                    if timeSet[1] == atime:
                        #print('timeSet[1] == atime')
                        Lv[i] = (chosen_distance,atime)
                        Lv = sorted(Lv, key=lambda x: float(x[0]))
                        flag = False
                        break
                #print(Lv)
                if (flag):
                    Lv.append((chosen_distance,atime))
                    Lv = sorted(Lv, key=lambda x: float(x[0]))
                dominatedElementsList = []
                for i in range(len(Lv)-1):
                    s1,a1 = Lv[i]
                    s2,a2 = Lv[i+1]
                    if (s1<s2 and  a1 <= a2) or (s1 == s2 and a1 < a2):
                        if Lv[i+1] not in dominatedElementsList:
                            dominatedElementsList.append(Lv[i+1])
                    elif (s1>s2 and  a1 >= a2) or (s1 == s2 and a1 > a2):
                        if Lv[i] not in dominatedElementsList:
                            dominatedElementsList.append(Lv[i])
                #print('dominatedElementsList:\t'+str(dominatedElementsList))
                for element in dominatedElementsList:
                    if element in Lv:
                        Lv.remove(element)
                vertexToLv_dict.update({edge[1]:Lv})
                #print('for destination:\t'+str(edge[1]))
                #print(vertexToLv_dict)
                sp = shortestPath_dict.get(edge[1])
                if (sp >chosen_distance):
                    #sp[0] = chosen_distance
                    shortestPath_dict.update({edge[1]:chosen_distance})
            elif edge[2] > timeInterval[1]:
                break
    return shortestPath_dict


def computeActualShortestPathAndDistance(sortedEdgeStream,vertex,vertexList,timeInterval):
    vertexToLv_dict ={}
    startingTime_dict = {}
    arrivalTime_dict = {}
    shortestPath_dict = {}
    startTime_dict = {}
    actualPath_dict = {}
    shortestPath_dict.update({vertex:(0,[])})
    actualPath_dict.update({vertex:[vertex]})
    for v in vertexList:
        #print(v)
        if not(v==vertex):
            shortestPath_dict.update({v:(math.inf,[])})
        actualPath_dict.update({v:[]})
        vertexToLv_dict.update({v:[]})
    #print(actualPath_dict)
    for edge in sortedEdgeStream:
        if edge[1] in vertexList:
            if edge[0] == vertex:
                Lv = vertexToLv_dict.get(edge[1])
                Lv.append((1,edge[2]))
                #actualPath_dict.update({edge[1]:[edge[0]]})
                Lv = sorted(Lv, key=lambda x: float(x[0]))
                vertexToLv_dict.update({edge[1]:Lv})
    foundEdge = False
    
    prevTime_dict = {}

    visitedList = []
    for edge in sortedEdgeStream:
        #print(edge)
        if edge[0] == vertex:
            foundEdge = True
        if foundEdge == True:
            if edge[0] in vertexList and edge[1] in vertexList:
                if edge[0] in prevTime_dict:
                    prevTimeList = prevTime_dict.get(edge[0])
                    #print(prevTimeList)
                else:
                    prevTimeList = []
                    prevTime_dict.update({edge[0]:[]})
                if edge[2]>= timeInterval[0] and edge[2]< timeInterval[1] and ((edge[2],edge[0]) not in prevTimeList):
                    prevTimeList.append((edge[2],edge[1]))
                    #prevTimeList.append((edge[2],edge[1]))
                    prevTime_dict.update({edge[0]:prevTimeList})

                    prevTime_dict.update({edge[1]:prevTimeList})
                    
                    #print(edge)
                    if edge[0] == vertex:
                        Lv = vertexToLv_dict.get(edge[0])
                        if (0,edge[2]) not in Lv:
                            Lv.append((0,edge[2]))
                            Lv = sorted(Lv, key=lambda x: float(x[0]))
                            vertexToLv_dict.update({edge[0]:Lv})
                    max_atime = 0
                    chosen_distance =0
                    
                    
                    Lv = vertexToLv_dict.get(edge[0])
                    for distance,atime in Lv:
                        if atime> max_atime and atime<edge[2]:
                            chosen_distance = distance
                            #print(chosen_distance)
                            max_atime = atime
                            #maxPath = path
                            
                    
                    chosen_distance += 1
                    atime = edge[2]
                    '''if edge[1] not in maxPath and len(maxPath)<=chosen_distance:
                        maxPath.append(edge[1])
                    print(maxPath)'''
                    Lv = vertexToLv_dict.get(edge[1])
                    flag = True
                    for i in range(len(Lv)):
                        timeSet = Lv[i]
                        if timeSet[1] == atime:
                            Lv[i] = (chosen_distance,atime)
                            Lv = sorted(Lv, key=lambda x: float(x[0]))
                            flag = False
                            break
                    if (flag):
                        Lv.append((chosen_distance,atime))
                        Lv = sorted(Lv, key=lambda x: float(x[0]))
                    dominatedElementsList = []
                    for i in range(len(Lv)-1):
                        s1,a1 = Lv[i]
                        s2,a2 = Lv[i+1]
                        if (s1<s2 and  a1 <= a2) or (s1 == s2 and a1 < a2):
                            if Lv[i+1] not in dominatedElementsList:
                                dominatedElementsList.append(Lv[i+1])
                        elif (s1>s2 and  a1 >= a2) or (s1 == s2 and a1 > a2):
                            if Lv[i] not in dominatedElementsList:
                                dominatedElementsList.append(Lv[i])
                    for element in dominatedElementsList:
                        if element in Lv:
                            Lv.remove(element)
                    #print('###########')
                    '''print(edge)
                    print('###########')
                    print(actualPath_dict)
                    print('-----------')'''
                    path = actualPath_dict.get(edge[0])
                    maxPath = path.copy()
                    new_maxPath = path.copy()
                    '''if edge[1] == 16:
                        print(actualPath_dict)
                        print(edge)
                        print('Before:')
                        print(maxPath)'''
                    #print(vertex)
                    
                    
                    #print(maxPath)
                    
                    if ((vertex,edge[1]) not in visitedList or edge[0]==vertex) and edge[1]!=vertex:
                        '''if vertex == edge[0]:
                            print(edge)
                            print(new_maxPath)'''
                        visitedList.append((vertex,edge[1]))
                        if edge[0] not in new_maxPath:
                            new_maxPath.append(edge[0])
                        if edge[1] not in new_maxPath:
                            new_maxPath.append(edge[1])
                        '''if vertex == edge[0]:
                            print(new_maxPath)'''
                        actualPath_dict.update({edge[1]:new_maxPath})
                    '''if len(new_maxPath) > chosen_distance:
                        new_maxPath = maxPath.copy()
                        if edge[1] not in new_maxPath:
                            new_maxPath.append(edge[1])'''
                    '''if edge[1] ==16:
                        print('After:')
                        print(maxPath)
                        print('-------')'''
                    
                    #maxPath = path
                    #print(maxPath)
                    #if vertex in new_maxPath:
                        
                        
                        
                    vertexToLv_dict.update({edge[1]:Lv})
                    #print(shortestPath_dict)
                    sp = shortestPath_dict.get(edge[1])
                        
                    sp = sp[0]
                    #print(sp)
                    #print('-------')
                    #print(chosen_distance)
                    #print(sp)
                    if (sp >chosen_distance):
                            #sp[0] = chosen_distance
                            maxPath1 = actualPath_dict.get(edge[1])
                            if vertex not in maxPath1:
                                maxPath1 = []
                                chosen_distance = math.inf
                            if len(maxPath1)-1>chosen_distance:
                                chosen_distance = len(maxPath1)-1

                            
                            shortestPath_dict.update({edge[1]:(chosen_distance,maxPath1)})
                elif edge[2] > timeInterval[1]:
                    break
        #print(actualPath_dict)
        shortestPath_dict.update({vertex:(0,[vertex])})
    #print(shortestPath_dict)
    return shortestPath_dict

def predictShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationNodesNum):

    shortestPath_dict = computeShortestPathDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)
    
    distanceNodeList_dict = {}

    for node in list(shortestPath_dict.keys()):
        if node in vertexList:
            distance = shortestPath_dict.get(node)
            if distance in list(distanceNodeList_dict.keys()):
                nodeList =distanceNodeList_dict.get(distance)
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
            else:
                nodeList = []
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
    nodesList = list(shortestPath_dict.keys())
    destinationVertexList = []
    while len(destinationVertexList)<destinationNodesNum:
        destNode = random.choice(nodesList)
        if destNode not in destinationVertexList and not(destNode == vertex0):
            destinationVertexList.append(destNode)

    #for vertex in vertexList:
    for vertex in destinationVertexList:
        distanceNodeKeysList = list(distanceNodeList_dict.keys())
        distanceNodeKeysList.sort()

        node1_distance = shortestPath_dict.get(vertex)
        if (node1_distance == math.inf):
            node1_distance = distanceNodeKeysList[len(distanceNodeKeysList)-2]+1

        for distance in range(0,node1_distance-1):

            nodeList = distanceNodeList_dict.get(distance)
            for node in nodeList:
                edge = (node,vertex)
                if embeddingPassingType == 'Hadamard':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeHadamard(edge,embeddings_dict)
                elif embeddingPassingType == 'WeightedL1':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL1(edge,embeddings_dict)
                elif embeddingPassingType == 'WeightedL2':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL2(edge,embeddings_dict)
                else:
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeAverage(edge,embeddings_dict)
                predictionResult = clf.predict([embeddingEdgeRepresentation])
                predictionResultProba = clf.predict_proba([embeddingEdgeRepresentation])
                '''print("predictionResult:\t"+str(predictionResult))
                print("predictionResultProba:\t"+str(predictionResultProba))'''
                if predictionResult == 1:
                    if node1_distance == distanceNodeKeysList[len(distanceNodeKeysList)-2]+1:
                        node1_distance = math.inf
                    
                    if node1_distance in list(distanceNodeList_dict.keys()):
                        nodeList1 = distanceNodeList_dict.get(node1_distance)
                        if vertex in nodeList1:
                            nodeList1.remove(vertex)
                        distanceNodeList_dict.update({node1_distance:nodeList1})
                        node1_distance = distance + 1
                        shortestPath_dict.update({vertex:node1_distance})
                        #print(node1_distance)
                        nodeList1 = distanceNodeList_dict.get(node1_distance)
                        if vertex not in nodeList1:
                            nodeList1.append(vertex)
                            distanceNodeList_dict.update({node1_distance:nodeList1})
                        break
    return shortestPath_dict

def predictShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationNodesNum):

    shortestPath_dict = computeShortestPathDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)
    
    distanceNodeList_dict = {}

    for node in list(shortestPath_dict.keys()):
        if node in vertexList:
            distance = shortestPath_dict.get(node)
            if distance in list(distanceNodeList_dict.keys()):
                nodeList =distanceNodeList_dict.get(distance)
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
            else:
                nodeList = []
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
    nodesList = list(shortestPath_dict.keys())
    destinationVertexList = []
    while len(destinationVertexList)<destinationNodesNum:
        destNode = random.choice(nodesList)
        if destNode not in destinationVertexList and not(destNode == vertex0):
            destinationVertexList.append(destNode)

    #for vertex in vertexList:
    for vertex in destinationVertexList:
        distanceNodeKeysList = list(distanceNodeList_dict.keys())
        distanceNodeKeysList.sort()

        node1_distance = shortestPath_dict.get(vertex)
        if (node1_distance == math.inf):
            node1_distance = distanceNodeKeysList[len(distanceNodeKeysList)-2]+1

        for distance in range(0,node1_distance-1):

            nodeList = distanceNodeList_dict.get(distance)
            for node in nodeList:
                edge = (node,vertex)
                if embeddingPassingType == 'Hadamard':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeHadamard(edge,embeddings_dict)
                elif embeddingPassingType == 'WeightedL1':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL1(edge,embeddings_dict)
                elif embeddingPassingType == 'WeightedL2':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL2(edge,embeddings_dict)
                else:
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeAverage(edge,embeddings_dict)
                predictionResult = clf.predict([embeddingEdgeRepresentation])
                predictionResultProba = clf.predict_proba([embeddingEdgeRepresentation])
                '''print("predictionResult:\t"+str(predictionResult))
                print("predictionResultProba:\t"+str(predictionResultProba))'''
                if predictionResult == 1:
                    if node1_distance == distanceNodeKeysList[len(distanceNodeKeysList)-2]+1:
                        node1_distance = math.inf
                    
                    if node1_distance in list(distanceNodeList_dict.keys()):
                        nodeList1 = distanceNodeList_dict.get(node1_distance)
                        if vertex in nodeList1:
                            nodeList1.remove(vertex)
                        distanceNodeList_dict.update({node1_distance:nodeList1})
                        node1_distance = distance + 1
                        shortestPath_dict.update({vertex:node1_distance})
                        #print(node1_distance)
                        nodeList1 = distanceNodeList_dict.get(node1_distance)
                        if vertex not in nodeList1:
                            nodeList1.append(vertex)
                            distanceNodeList_dict.update({node1_distance:nodeList1})
                        break
    return shortestPath_dict


def predictProbaShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationNodesNum):

    shortestPath_dict = computeShortestPathDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)
    
    distanceNodeList_dict = {}

    for node in list(shortestPath_dict.keys()):
        if node in vertexList:
            distance = shortestPath_dict.get(node)
            if distance in list(distanceNodeList_dict.keys()):
                nodeList =distanceNodeList_dict.get(distance)
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
            else:
                nodeList = []
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
    nodesList = list(shortestPath_dict.keys())
    destinationVertexList = []
    while len(destinationVertexList)<destinationNodesNum:
        destNode = random.choice(nodesList)
        if destNode not in destinationVertexList and not(destNode == vertex0):
            destinationVertexList.append(destNode)

    #for vertex in vertexList:
    for vertex in destinationVertexList:
        distanceNodeKeysList = list(distanceNodeList_dict.keys())
        distanceNodeKeysList.sort()
        maxPosibility = 0.5

        node1_distance = shortestPath_dict.get(vertex)
        if (node1_distance == math.inf):
            node1_distance = distanceNodeKeysList[len(distanceNodeKeysList)-2]+1

        for distance in range(0,node1_distance-1):

            nodeList = distanceNodeList_dict.get(distance)
            for node in nodeList:
                edge = (node,vertex)
                if embeddingPassingType == 'Hadamard':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeHadamard(edge,embeddings_dict)
                elif embeddingPassingType == 'WeightedL1':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL1(edge,embeddings_dict)
                elif embeddingPassingType == 'WeightedL2':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL2(edge,embeddings_dict)
                else:
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeAverage(edge,embeddings_dict)
                predictionResult = clf.predict([embeddingEdgeRepresentation])
                predictionResultProba = list(clf.predict_proba([embeddingEdgeRepresentation]))[0]
                currentProba = list(predictionResultProba)[1]
                '''print("predictionResult:\t"+str(predictionResult))
                print("predictionResultProba:\t"+str(predictionResultProba))'''
                if currentProba >maxPosibility:
                    maxPosibility = currentProba
                    if node1_distance == distanceNodeKeysList[len(distanceNodeKeysList)-2]+1:
                        node1_distance = math.inf
                    
                    if node1_distance in list(distanceNodeList_dict.keys()):
                        nodeList1 = distanceNodeList_dict.get(node1_distance)
                        if vertex in nodeList1:
                            nodeList1.remove(vertex)
                        distanceNodeList_dict.update({node1_distance:nodeList1})
                        node1_distance = distance + 1
                        shortestPath_dict.update({vertex:node1_distance})
                        #print(node1_distance)
                        nodeList1 = distanceNodeList_dict.get(node1_distance)
                        if vertex not in nodeList1:
                            nodeList1.append(vertex)
                            distanceNodeList_dict.update({node1_distance:nodeList1})
                        break
    return shortestPath_dict
            
def predictExpectedLengthShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,destinatoinVertexList,timeInterval,embeddingPassingType,destinationNodesNum):

    shortestPath_dict = computeShortestPathDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)
    
    distanceNodeList_dict = {}

    for node in list(shortestPath_dict.keys()):
        if node in vertexList:
            distance = shortestPath_dict.get(node)
            if distance in list(distanceNodeList_dict.keys()):
                nodeList =distanceNodeList_dict.get(distance)
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
            else:
                nodeList = []
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
    #nodesList = list(shortestPath_dict.keys())
    #destinatoinVertexList = []
    '''while len(destinationVertexList)<destinationNodesNum:
        destNode = random.choice(nodesList)
        if destNode not in destinationVertexList and not(destNode == vertex0):
            destinationVertexList.append(destNode)'''

    #for vertex in vertexList:
    for vertex in destinatoinVertexList:
        distanceNodeKeysList = list(distanceNodeList_dict.keys())
        distanceNodeKeysList.sort()
        maxPosibility = 0.5
        averageDistanceProba_dict = {}
        node1_distance = shortestPath_dict.get(vertex)
        #print(node1_distance)
        if (node1_distance == math.inf):
            node1_distance = distanceNodeKeysList[len(distanceNodeKeysList)-2]+1

        for distance in range(0,node1_distance):
            currentProba = 1
            nodeList = distanceNodeList_dict.get(distance)
            for node in nodeList:
                edge = (node,vertex)
                if embeddingPassingType == 'Hadamard':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeHadamard(edge,embeddings_dict)
                elif embeddingPassingType == 'WeightedL1':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL1(edge,embeddings_dict)
                elif embeddingPassingType == 'WeightedL2':
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL2(edge,embeddings_dict)
                else:
                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeAverage(edge,embeddings_dict)
                predictionResult = clf.predict([embeddingEdgeRepresentation])
                predictionResultProba = list(clf.predict_proba([embeddingEdgeRepresentation]))[0]
                #print(predictionResultProba)
                currentProba = currentProba*(1-list(predictionResultProba)[1])
                #print(currentProba)
                '''print("predictionResult:\t"+str(predictionResult))
                print("predictionResultProba:\t"+str(predictionResultProba))'''
                '''if currentProba >maxPosibility:
                    maxPosibility = currentProba'''
            distKey = distance+1
            currentProba = 1 - currentProba

            #print('currentProba:\t'+str(currentProba))
            '''if currentProba >=1:
                currentProba = 0'''
            averageDistanceProba_dict.update({distKey:currentProba})
            if node1_distance == distanceNodeKeysList[len(distanceNodeKeysList)-2]+1:
                        node1_distance = math.inf
                    
            '''if node1_distance in list(distanceNodeList_dict.keys()):
                        nodeList1 = distanceNodeList_dict.get(node1_distance)
                        if vertex in nodeList1:
                            nodeList1.remove(vertex)
                        distanceNodeList_dict.update({node1_distance:nodeList1})
                        node1_distance = distance + 1'''
        tempDist = 0
        #print(len(list(averageDistanceProba_dict.keys())))
        '''if len(list(averageDistanceProba_dict.keys()))>0:
            print(averageDistanceProba_dict)'''
        for distKey in list(averageDistanceProba_dict.keys()):
                distProba = averageDistanceProba_dict.get(distKey)

                #print('distKey:\t'+str(distKey)+'\tdistProba:\t'+str(distProba))
                tempDist = tempDist + int(distKey)*distProba
                #print(tempDist)
        if vertex == destinatoinVertexList[1]:
            '''print('vertex:\t'+str(vertex)+'\tdistance:\t'+str(tempDist))
            print(tempDist)'''
        if tempDist !=0:
            shortestPath_dict.update({vertex:math.ceil(tempDist)})
        
        #print(node1_distance)
        #nodeList1 = distanceNodeList_dict.get(node1_distance)
        '''if vertex not in nodeList1:
                            nodeList1.append(vertex)
                            distanceNodeList_dict.update({node1_distance:nodeList1})'''
    return shortestPath_dict

'''graph_file_path = 'Temporal_Stream_File.txt'
graphFileRead = open(graph_file_path, "r")
streamEdgeList = [] #should be shorted based on time
vertexList = []
minTime = math.inf
maxTime = 0
for line in file_reader(graphFileRead):
    line = line.split(',')
    edge =(int(line[0]),int(line[1]),float(line[2]))
    if edge[0] not in vertexList:
         vertexList.append(edge[0])
    if edge[1] not in vertexList:
         vertexList.append(edge[1])
    if minTime> edge[2]:
        minTime = edge[2]
    if maxTime< edge[2]:
        maxTime = edge[2]
    streamEdgeList.append(edge)
timeInterval = [minTime,maxTime]
sortedEdgeStream = sorted(streamEdgeList, key=lambda x: float(x[2]))
sortedReverseEdgeStream = sorted(streamEdgeList, key=lambda x: float(x[2]),reverse=True)
print("Max Time Interval is:   "+str((minTime,maxTime)))'''
'''for i in range(len(sortedReverseEdgeStream)-1):
    edge0 = sortedReverseEdgeStream[i]
    edge1 = sortedReverseEdgeStream[i+1]
    if (edge0[2]>edge1[2]):
        print(i)'''
#print(sortedEdgeStream)
'''time_dict,pathList = computeEarliestArivalTime(sortedEdgeStream,vertexList[0],vertexList,timeInterval)
print(len(pathList))'''
#print(time_dict)
'''for vertex in list(time_dict.keys()):
    earliest_time = time_dict.get(vertex)

    print("Earliest Arival path at vertex  "+str(vertex)+' is '+str(earliest_time))'''

#latestDepartureTime_dict = computeLatestDepartureTime(sortedReverseEdgeStream,vertexList[0],vertexList,timeInterval)
'''for vertex in list(latestDepartureTime_dict.keys()):
    earliest_time = latestDepartureTime_dict.get(vertex)

    print("Latest Departure path at vertex  "+str(vertex)+' is '+str(earliest_time))'''



#fastestPath_dict = computeFastestPathDuration(sortedEdgeStream,vertexList[0],vertexList,timeInterval)

'''for vertex in list(fastestPath_dict.keys()):
    fastest_path = fastestPath_dict.get(vertex)

    print("Latest Departure path at vertex  "+str(vertex)+' is '+str(fastest_path))'''


'''fastestPath_dict =  computeShortestPathDistance(sortedEdgeStream,vertexList[0],vertexList,timeInterval)

for vertex in list(fastestPath_dict.keys()):
    fastest_path = fastestPath_dict.get(vertex)

    print("Shortest path at vertex  (one pass)   "+str(vertex)+' is of length '+str(fastest_path))
distanceNodeList_dict = {}
for node in range(len(list(fastestPath_dict.keys()))):
    distance = fastestPath_dict.get(node)
    if distance in list(distanceNodeList_dict.keys()):
        nodeList =distanceNodeList_dict.get(distance)
        nodeList.append(node)
        distanceNodeList_dict.update({distance:nodeList})
    else:
        nodeList = []
        nodeList.append(node)
        distanceNodeList_dict.update({distance:nodeList})'''

