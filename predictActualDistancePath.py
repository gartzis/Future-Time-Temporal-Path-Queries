from read_Edge_Stream import computeActualShortestPathAndDistance
import binaryOperatorsForLearningEdgeFeatures as BOLEF
import networkx as nx
import math
import random

def predictProbaActualShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationVertexList):
    visitedList = []
    shortestPath_dict = computeActualShortestPathAndDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)
    
    distanceNodeList_dict = {}
    path_dict = {}
    for node in list(shortestPath_dict.keys()):
        if node in vertexList:
            distance,prev_path = shortestPath_dict.get(node)
            path_dict.update({(distance,node):prev_path})
            if distance in list(distanceNodeList_dict.keys()):
                nodeList =distanceNodeList_dict.get(distance)
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
            else:
                nodeList = []
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
    nodesList = list(shortestPath_dict.keys())
    
    destinationVertexList1 = []
    for vertex in destinationVertexList:
        if vertex in vertexList:
            destinationVertexList1.append(vertex)
    destinationVertexList = destinationVertexList1
    distanceNodeKeysList = []
    
    for vertex in destinationVertexList:
        for dist in list(distanceNodeList_dict.keys()):
            if dist != math.inf:
                distanceNodeKeysList.append(dist)
        distanceNodeKeysList.sort()
        maxPosibility = 0

        node1_distance,prev_path = shortestPath_dict.get(vertex)

        if (node1_distance == math.inf):
            node1_distance = int(max(distanceNodeKeysList))
        
        for distance in range(0,node1_distance):
            
            if distance in list(distanceNodeList_dict.keys()):
                nodeList = distanceNodeList_dict.get(distance)
                
                for node in nodeList:
                        
                        if not(distance==math.inf):
                            
                            prev_path = path_dict.get((distance,node))
                            
                        else:
                            
                            prev_path = [vertex0]
                        
                        
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
                        
                        if currentProba >maxPosibility:
                            
                            maxPosibility = currentProba
                            
                            node2_distance = distance + 1
                            
                            if ((vertex0,edge[0]) not in visitedList):
                                
                                visitedList.append((vertex0,edge[1]))
                                new_path = prev_path.copy()
                                
                                if edge[0] not in new_path:
                                    new_path.append(edge[0])
                                if edge[1] not in new_path:
                                    new_path.append(edge[1])
                                
                                if node1_distance in list(distanceNodeList_dict.keys()):
                                    nodeList1 = distanceNodeList_dict.get(node1_distance)
                                    if vertex in nodeList1:
                                        nodeList1.remove(vertex)
                                    distanceNodeList_dict.update({node1_distance:nodeList1})
                                    if (node1_distance,vertex) in list(path_dict.keys()):
                                        path_dict.pop((node1_distance,vertex))
                                
                                if vertex0 not in new_path:
                                    new_path = []
                                    node2_distance = math.inf
                                if len(new_path)-1>node2_distance:
                                    node2_distance = len(new_path)-1
                                
                                shortestPath_dict.update({vertex:(node2_distance,new_path)})
                                
                                if node2_distance in list(distanceNodeList_dict.keys()):
                                    nodeList1 = distanceNodeList_dict.get(node2_distance)
                                else:
                                    nodeList1 = []
                                
                                if vertex not in nodeList1:
                                    nodeList1.append(vertex)
                                    path_dict.update({(node2_distance,vertex):new_path})
                                    distanceNodeList_dict.update({node2_distance:nodeList1})
                                
    return shortestPath_dict





def recommendProbaActualShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationVertexList):
    visitedList = []
    recommendedPathsAllVertices_dict = {}
    shortestPath_dict = computeActualShortestPathAndDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)
    
    distanceNodeList_dict = {}
    path_dict = {}
    distanceNodeKeysList = []
    for node in list(shortestPath_dict.keys()):
        if node in vertexList:
            distance,prev_path = shortestPath_dict.get(node)
            if distance not in distanceNodeKeysList and distance != math.inf:
                distanceNodeKeysList.append(distance)
            path_dict.update({(distance,node):prev_path})
            if distance in list(distanceNodeList_dict.keys()):
                nodeList =distanceNodeList_dict.get(distance)
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
            else:
                nodeList = []
                nodeList.append(node)
                distanceNodeList_dict.update({distance:nodeList})
    distanceNodeKeysList.append(int(max(distanceNodeKeysList)+1))
    distanceNodeKeysList.sort()
    nodesList = list(shortestPath_dict.keys())
    
    destinationVertexList1 = []
    for vertex in destinationVertexList:
        if vertex in vertexList:
            destinationVertexList1.append(vertex)
    destinationVertexList = destinationVertexList1
    
    recommendedPaths_dict = {}
    for vertex in destinationVertexList:
        recommendedPaths = []
        
        
        maxPosibility = 0

        node1_distance,prev_path = shortestPath_dict.get(vertex)
        
        if (node1_distance == math.inf):
            node1_distance = int(max(distanceNodeKeysList))
            
        
        for distance in range(0,node1_distance):
            
            

            nodeList = distanceNodeList_dict.get(distance)
            
            for node in nodeList:
                    
                    if not(distance==math.inf):
                        
                        prev_path = path_dict.get((distance,node))
                        
                    else:
                        
                        prev_path = [vertex0]
                    
                    
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
                    
                    if currentProba >=maxPosibility:
                        node2_distance = distance + 1
                        
                        if ((vertex0,edge[1]) not in visitedList):
                            
                            visitedList.append((vertex0,edge[1]))
                            new_path = prev_path.copy()
                            
                            if edge[0] not in new_path:
                                new_path.append(edge[0])
                            if edge[1] not in new_path:
                                new_path.append(edge[1])
                            
                            if node1_distance in list(distanceNodeList_dict.keys()):
                                nodeList1 = distanceNodeList_dict.get(node1_distance)
                                if vertex in nodeList1:
                                    nodeList1.remove(vertex)
                                distanceNodeList_dict.update({node1_distance:nodeList1})
                                if (node1_distance,vertex) in list(path_dict.keys()):
                                    path_dict.pop((node1_distance,vertex))
                            
                            if vertex0 not in new_path:
                                new_path = []
                                node2_distance = math.inf
                            if len(new_path)-1>node2_distance:
                                node2_distance = len(new_path)-1
                            if vertex in list(recommendedPaths_dict.keys()):
                                recommendedPaths = recommendedPaths_dict.get(vertex)
                            else:
                                recommendedPaths = []
                            if len(recommendedPaths) <5:
                                recommendedPaths.append((node2_distance,new_path[len(new_path)-2],new_path,currentProba))
                            
                            if len(recommendedPaths) ==5:
                                maxPosibility = currentProba
                                recommendedPaths[len(recommendedPaths)-1] = (node2_distance,edge[0],new_path,currentProba)
                            recommendedPaths = sorted(recommendedPaths, key=lambda x: float(x[3]))
                            
                            

                            
                            shortestPath_dict.update({vertex:(node2_distance,new_path)})
                            
                            if node2_distance in list(distanceNodeList_dict.keys()):
                                nodeList1 = distanceNodeList_dict.get(node2_distance)
                            else:
                                nodeList1 = []
                            
                            if vertex not in nodeList1:
                                nodeList1.append(vertex)
                                path_dict.update({(node2_distance,vertex):new_path})
                                distanceNodeList_dict.update({node2_distance:nodeList1})
                            
        recommendedPaths_dict.update({vertex:recommendedPaths})
    
    return recommendedPaths_dict

def predictLinkLikeJodie(clf,previousPathList,destinationVertex,embeddings_dict,embeddingPassingType):
    
    pathsList = []
    for sourceVertexI in range(len(previousPathList)-2):
        newPathList = []
        sourceVertex = previousPathList[sourceVertexI]
        
        edge = (sourceVertex,destinationVertex)
        
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
        newPathList = previousPathList[:sourceVertexI+1]
        
        newPathList.append(destinationVertex)
        pathDistance = int(len(newPathList))-1
        pathPredictionTuple = (currentProba,pathDistance,newPathList)
        pathsList.append(pathPredictionTuple)
    
    pathList = sorted(pathsList, key=lambda x: float(x[0]),reverse=True)
    recommendedPath = pathList[0]
    return recommendedPath
        