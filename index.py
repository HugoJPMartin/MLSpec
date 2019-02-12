import pandas as pd
import MLSpec as mlspec
import os, sys, json, traceback




def getPossibleConfigurations(hyperparamsList):
    configs = []
    
    #Check if there is still something to iterate
    if any(hyperparamsList):
        
        #Get the first key in the list of hyperparams
        key = next(iter(hyperparamsList))
        
        #Get all the possible configs without considering the current key
        nextPossibleConfigs = getPossibleConfigurations({k:v for k,v in hyperparamsList.items() if not k == key})
        
        #Combine every value of the current key to all known possible configurations
        for value in hyperparamsList[key]:
            
            if len(nextPossibleConfigs) > 0:
                for config in nextPossibleConfigs:
                    config[key] = value
                    configs.append(config)
            
            else:
                configs.append({key:value})
    
    return configs



if len(sys.argv) == 1:
    print("no dataset given")
    sys.exit()

name = sys.argv[1]
print(sys.argv)
#name = "Apache"
dataset = "datasets/"+name+".csv"

if not os.path.isfile(dataset):
    print("Dataset "+name+" unknown")
    sys.exit()

print("Dataset "+name)
#resultsPath = "results/"
learningType = "gbClassification"
perf="perf"

params = {
    "name":name,
    "dataset":dataset,
    "resultsPath":"results/",
    "learningType":"classification",
    "perf":"perf",
    "nbThresholds":20,
    "nbFolds":10,
    "minSampleSize":10,
    "maxSampleSize":None,
    "paceSampleSize":None
}

#List all possible hyperparams, in order to avoid error with scikit
possibleHyperparams = ["criterion","splitter","max_features","max_depth","min_samples_split","min_samples_leaf", "min_weight_fraction_leaf","max_leaf_nodes","class_weight","random_state","min_impurity_decrease","presort","loss","learning_rate","n_estimators","subsample"]

hyperparams = {}

hyperparamsList = {
    
}

#Get input params
for k,v in enumerate(sys.argv):
    if v.startswith("--"):
        key = v[2:]
        #json loads handle list and int from input string
        try:
            value = json.loads(sys.argv[k+1])
        except:
            value = sys.argv[k+1]
        
        if key in params:
            params[key] = value
        elif key in possibleHyperparams:
            if type(value) == list:
                hyperparamsList[key] = value
            else:
                hyperparams[key] = value

params["hyperparams"] = hyperparams

print("Configuring runs")

configs = getPossibleConfigurations(hyperparamsList)

if len(configs) > 0:
    for config in configs:
        
        for k,v in config.items():
            params["hyperparams"][k] = v

        try:
            '''
            ml = mlspec.MLSpec(name,dataset,resultsPath, learningType, perf=perf, nbThresholds=nbThresholds, 
                               nbFolds=nbFolds, minSampleSize=minSampleSize, maxSampleSize=maxSampleSize, 
                               paceSampleSize=paceSampleSize)
            ml = mlspec.MLSpec(name,dataset,resultsPath, learningType, perf=perf)'''
            ml = mlspec.MLSpec(**params)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        print("Starting")

        try:
            ml.start()
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Fails")
else:
    
        try:
            '''
            ml = mlspec.MLSpec(name,dataset,resultsPath, learningType, perf=perf, nbThresholds=nbThresholds, 
                               nbFolds=nbFolds, minSampleSize=minSampleSize, maxSampleSize=maxSampleSize, 
                               paceSampleSize=paceSampleSize)
            ml = mlspec.MLSpec(name,dataset,resultsPath, learningType, perf=perf)'''
            ml = mlspec.MLSpec(**params)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        print("Starting")

        try:
            ml.start()
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Fails")
print("End")

