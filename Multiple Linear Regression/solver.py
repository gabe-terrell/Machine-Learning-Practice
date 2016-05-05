import sys
import pandas as pd
import statsmodels.api as sm
import numpy as np

def main():
    features, datasets = getDataInfo()
    X, y = organizeData(features, datasets)
    model = modelData(X, y)
    datasetsToPredict = int(input())
    predictData(model, datasetsToPredict)

def getDataInfo():
    info = input().split()
    return int(info[0]), int(info[1])

def organizeData (features, datasets):
    X = [[] for _ in range(features)]
    Y = []
    for _ in range(datasets):
        dataset = [float(i) for i in input().split()]
        for i, value in enumerate(dataset[:-1]):
            X[i].append(value)
        Y.append(dataset[-1])
    X = np.column_stack(X)
    X = sm.add_constant(X)
    return X, Y

def modelData(X, y):
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def predictData(model, n):
    datasets = []
    for _ in range(n):
        dataset = ([float(i) for i in input().split()])
        dataset.insert(0, 1.0) # constant
        datasets.append(dataset)
    for result in model.predict(datasets):
        print(result)

main()