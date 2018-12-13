import pandas as pd
import MLSpec as mlspec
import os, sys

if len(sys.argv) == 1:
    print("no dataset given")
    sys.exit()

name = sys.argv[1]
#name = "Apache"
dataset = "datasets/"+name+".csv"

if not os.path.isfile(dataset):
    print("Dataset "+name+" unknown")
    sys.exit()

print("Dataset "+name)
resultsPath = "results/"
learningType = "classification"
perf="perf"
'''
nbThresholds=20
nbFolds = 10
minSampleSize=10
maxSampleSize=100
paceSampleSize=2
'''
print("Configuring runs")
try:
    '''
    ml = mlspec.MLSpec(name,dataset,resultsPath, learningType, perf=perf, nbThresholds=nbThresholds, 
                       nbFolds=nbFolds, minSampleSize=minSampleSize, maxSampleSize=maxSampleSize, 
                       paceSampleSize=paceSampleSize)'''
    ml = mlspec.MLSpec(name,dataset,resultsPath, learningType, perf=perf)
except Exception as e:
    print(e)
print("Starting")

try:
    ml.start()
except Exception as e:
    print(e)
    print("Fails")

print("End")
