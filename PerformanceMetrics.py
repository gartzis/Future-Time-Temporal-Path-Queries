import math



def computePrecision(truePossitiveList,retreivedList):
    precision = len(truePossitiveList)/ len(retreivedList)
    return precision


def computeRecall(truePossitiveList,FalseNegativeList):#Sensitivity | True Positive Rate | Recall
    positivesNum = len(truePossitiveList) + len(FalseNegativeList)
    recall = len(truePossitiveList)/positivesNum
    return recall

def computeFalseNegativeRate(truePossitiveList,FalseNegativeList):
    positivesNum = len(truePossitiveList) + len(FalseNegativeList)
    if (positivesNum>0):
        FNR = len(FalseNegativeList)/positivesNum
    else:
        FNR = math.inf
    return FNR

def computeTrueNegativeRate(falsePossitiveList,trueNegativeList):#Specificity | True Negative Rate
    negativesNum = len(falsePossitiveList) + len(trueNegativeList)
    if (negativesNum>0):
        TNR = len(trueNegativeList)/negativesNum
    else:
        TNR = math.inf
    return TNR
def computeFalsePositiveRate(falsePossitiveList,trueNegativeList):
    negativesNum = len(falsePossitiveList) + len(trueNegativeList)
    if (negativesNum>0):
        FPR = len(falsePossitiveList)/negativesNum
    else:
        FPR = math.inf
    return FPR

def computeFMeasure(precision,recall):
    F = 2* (precision*recall)/(precision+recall)
    return F