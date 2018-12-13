import os
from os import listdir
from os.path import isfile, join
import json
from threading import Thread,BoundedSemaphore

import sys, os
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

threadLimiter = BoundedSemaphore(os.sysconf('SC_NPROCESSORS_ONLN'))

class MLSpec:
    
    def __init__(self, name=None, dataset=None, resultsPath=None, learningType=None, perf="perf", graphPath=None, nbThresholds=20, nbFolds = 10, minSampleSize=2, maxSampleSize=None, paceSampleSize=None, hyperparams=None, percentageThresholds=None):
        
        if not learningType in ["classification", "regression"] and not learningType.startswith("multiclass-") :
            raise Exception("Type must be classification, regression or multiclass")
        
        self.setDataset(dataset)
            
            
        #Create the folder for results if it does not exist:
        if not os.path.exists(resultsPath):
            try:
                os.makedirs(resultsPath)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                    
        if not graphPath == None:
            #Create the folder for graphs if it does not exist:
            if not os.path.exists(graphPath):
                try:
                    os.makedirs(graphPath)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
        
        self.name = name
        self.resultsPath = resultsPath
        self.perf = perf
        self.graphPath = graphPath
        self.nbThresholds = nbThresholds
        self.nbFolds = nbFolds
        self.minSampleSize = minSampleSize
        if maxSampleSize == None and not dataset is None:
            self.maxSampleSize = int(self.dataset.shape[0] * 0.9)
        elif not maxSampleSize == None:
            self.maxSampleSize = maxSampleSize
            
        if paceSampleSize == None:
            self.paceSampleSize = int(self.maxSampleSize/50)
        else:
            self.paceSampleSize = 1
        
        self.learningType = learningType
        
        self.percentageThresholds = percentageThresholds
        
        self.saveFile = None
        self.dfResults = None
        
        if int(self.minSampleSize) < 2:
            raise Exception('minSampleSize cannot be less than 2')
            
        if learningType == "classification" or learningType.startswith("multiclass-"):
            if hyperparams == None:
                self.hyperparams = {
                    "criterion":"gini",
                    "splitter":"best",
                    "max_features":None,
                    "max_depth":None,
                    "min_samples_split":2,
                    "min_samples_leaf":1,
                    "min_weight_fraction_leaf":0.,
                    "max_leaf_nodes":None,
                    "class_weight":None,
                    "random_state":None,
                    "min_impurity_decrease":1e-7,
                    "presort":False
                }
            else:
                self.hyperparams = hyperparams
        elif learningType == "regression":
            if hyperparams == None:
                self.hyperparams =  {
                    "criterion":"mse",
                    "splitter":"best",
                    "max_depth":None,
                    "min_samples_split":2,
                    "min_samples_leaf":1,
                    "min_weight_fraction_leaf":0.,
                    "max_features":None,
                    "random_state":None,
                    "max_leaf_nodes":None,
                    "min_impurity_decrease":1e-7,
                    "presort":False
                }
            else:
                self.hyperparams = hyperparams
            
    @classmethod
    def from_results(cls, resultsFile):
        if not os.path.exists(resultsFile):
            raise Exception("File not found")
        
        filename = resultsFile.split('/')[-1]
        name = '-'.join(filename.split('-')[:-1])
        
        resultsPath = '/'.join(resultsFile.split('/')[:-1])+"/"
        
        dfResults = pd.read_csv(resultsFile)
        
        dfList = pd.read_csv(resultsPath +"results-list.csv")
        
        try:
            row = dfList[dfList["results"]==resultsFile].iloc[0]
        except:
            raise Exception("File not saved in results list")
            
        cl = cls(name=name, resultsPath=resultsPath, learningType=row["learningType"], perf=row["perf"], nbThresholds=row["nbThresholds"], nbFolds = row["nbFolds"], minSampleSize=row["minSampleSize"], maxSampleSize=row["maxSampleSize"], paceSampleSize=row["paceSampleSize"], hyperparams=json.loads(row["hyperparams"]))
        
        cl._setResults(pd.read_csv(resultsFile))
        cl._setSaveFile(filename)
        
        return cl
        
        
        
        
        
    def _mlClassification(self):
        
        perf = self.perf
        d = self.dataset
        d = d.sort_values(by=perf) # Sort it by perf to get threshold values
        thresholds = [d[perf].iloc[i * d.shape[0]//self.nbThresholds] for i in range(1, self.nbThresholds)]
        
        if not self.percentageThresholds == None:
            thresholds=[d[perf].quantile(th) for th in self.percentageThresholds]

        res = []
        threads=[]
        #for sr in range(1,99):
        for sr in range(self.minSampleSize, self.maxSampleSize, self.paceSampleSize):
            for t in thresholds:
                
                threads.append( MLThread(target=self._mlClassificationThread, args=(d, perf, sr, t)) )
                
        for t in threads:
            t.start()
            
        for t in threads:
            res.append(t.join())
        
        self.dfResults = pd.DataFrame(res)
        
    def _mlClassificationThread(self, dataset, perf, sr, t):
        shuffle_split = StratifiedShuffleSplit(train_size=sr, n_splits=self.nbFolds)
        d = dataset.copy()
        try:
            d["label"] = 0
            d.loc[d[perf] > t, "label"] = 1

            TN = TP = FN = FP = 0 # Counters for classification results

            clean = d.drop(["perf"],axis=1,errors="ignore")

            c = tree.DecisionTreeClassifier(**self.hyperparams)

            try:
                for train_index, test_index in shuffle_split.split(clean,clean.label):
                    c.fit(clean.drop(["label"],axis=1).iloc[train_index], clean.label.iloc[train_index])
                    pred = c.predict(clean.drop(["label"],axis=1).iloc[test_index])

                    dfTemp = pd.DataFrame()
                    dfTemp["label"] = clean.label.iloc[test_index]
                    dfTemp["pred"] = pred

                    TN += dfTemp[(dfTemp.label == 0) & (dfTemp.pred == 0)].shape[0]
                    TP += dfTemp[(dfTemp.label == 1) & (dfTemp.pred == 1)].shape[0]
                    FN += dfTemp[(dfTemp.label == 1) & (dfTemp.pred == 0)].shape[0]
                    FP += dfTemp[(dfTemp.label == 0) & (dfTemp.pred == 1)].shape[0]

            except Exception as e:
                print(e)

            return {
                "sr":sr,
                "t":t,
                "TN":TN/self.nbFolds,
                "TP":TP/self.nbFolds,
                "FN":FN/self.nbFolds,
                "FP":FP/self.nbFolds,
            }
        except Exception as e:
            print(e)
        
        
    def _mlRegression(self):
        
        perf = self.perf
        d = self.dataset
        d = d.sort_values(by=perf) # Sort it by perf to get threshold values
        thresholds = [d[perf].iloc[i * d.shape[0]//self.nbThresholds] for i in range(1, self.nbThresholds)]
        
        if not self.percentageThresholds == None:
            thresholds=[d[perf].quantile(th) for th in self.percentageThresholds]

        res = {"sr":[],"t":[],"TN":[],"TP":[],"FN":[],"FP":[],"MSE":[],"MAE":[]}
        #for sr in range(1,99):
        for sr in range(self.minSampleSize, self.maxSampleSize, self.paceSampleSize):
            for t in thresholds:
                #print("Computing for sr=%d and t=%.3f..." % (sr, t))

                shuffle_split = StratifiedShuffleSplit(train_size=sr, n_splits=self.nbFolds)

                d["label"] = 0
                d.loc[d[perf] > t, "label"] = 1

                TN = TP = FN = FP = MAE = MSE = 0 # Counters for classification results

                c = tree.DecisionTreeRegressor(**self.hyperparams)
                
                try:
                    for train_index, test_index in shuffle_split.split(d,d.label):
                        c.fit(d.drop([perf,"label"],axis=1).iloc[train_index], d[perf].iloc[train_index])
                        pred = c.predict(d.drop([perf,"label"],axis=1).iloc[test_index])

                        dfTest = pd.DataFrame()
                        dfTest[perf] = d[perf].iloc[test_index]
                        dfTest["pred"] = pred
                        dfTest["label"] = d.label.iloc[test_index]
                        dfTest["label_pred"] = 0
                        dfTest.loc[dfTest["pred"] > t, "label_pred"] = 1

                        MSE += mse(dfTest[perf],dfTest["pred"])
                        MAE += mae(dfTest[perf],dfTest["pred"])
                        
                        TN += dfTest[(dfTest.label == 0) & (dfTest.label_pred == 0)].shape[0]
                        TP += dfTest[(dfTest.label == 1) & (dfTest.label_pred == 1)].shape[0]
                        FN += dfTest[(dfTest.label == 1) & (dfTest.label_pred == 0)].shape[0]
                        FP += dfTest[(dfTest.label == 0) & (dfTest.label_pred == 1)].shape[0]


                except Exception as e:
                    print(e)
                    break
                    break

                res["sr"].append(sr)
                res["t"].append(t)
                res["MSE"].append(MSE/self.nbFolds)
                res["MAE"].append(MAE/self.nbFolds)
                res["TN"].append(TN/self.nbFolds)
                res["TP"].append(TP/self.nbFolds)
                res["FN"].append(FN/self.nbFolds)
                res["FP"].append(FP/self.nbFolds)

                #break
            #break
        
        self.dfResults = pd.DataFrame(res)
        
        
    def _mlMultiClassification(self, nbClasses):
        
        perf = self.perf
        d = self.dataset
        d = d.sort_values(by=perf) # Sort it by perf to get threshold values
        thresholds = [d[perf].iloc[i * d.shape[0]//self.nbThresholds] for i in range(1, self.nbThresholds)]
        
        if not self.percentageThresholds == None:
            thresholds=[d[perf].quantile(th) for th in self.percentageThresholds]

        res = {"sr":[],"t":[],"TN":[],"TP":[],"FN":[],"FP":[]}
        #for sr in range(1,99):
        for sr in range(self.minSampleSize, self.maxSampleSize, self.paceSampleSize):
            for t in thresholds:
                #print("Computing for sr=%d and t=%.3f..." % (sr, t))

                shuffle_split = StratifiedShuffleSplit(train_size=sr, n_splits=self.nbFolds)
                
                d, ltClasses, gtClasses = self.multiclassSeparator(d, t, int(nbClasses))
                
                TN = TP = FN = FP = 0 # Counters for classification results

                clean = d.drop(["perf"],axis=1,errors="ignore")

                c = tree.DecisionTreeClassifier(**self.hyperparams)
                
                try:
                    for train_index, test_index in shuffle_split.split(clean,clean.label):
                        c.fit(clean.drop(["label"],axis=1).iloc[train_index], clean.label.iloc[train_index])
                        pred = c.predict(clean.drop(["label"],axis=1).iloc[test_index])

                        dfTest = pd.DataFrame()
                        dfTest["label"] = clean.label.iloc[test_index]
                        dfTest["pred"] = pred
                        TN += dfTest[(dfTest.label.isin(ltClasses)) & (dfTest.pred.isin(ltClasses))].shape[0]
                        TP += dfTest[(dfTest.label.isin(gtClasses)) & (dfTest.pred.isin(gtClasses))].shape[0]
                        FN += dfTest[(dfTest.label.isin(gtClasses)) & (dfTest.pred.isin(ltClasses))].shape[0]
                        FP += dfTest[(dfTest.label.isin(ltClasses)) & (dfTest.pred.isin(gtClasses))].shape[0]


                except Exception as e:
                    print(e)

                res["sr"].append(sr)
                res["t"].append(t)
                res["TN"].append(TN/self.nbFolds)
                res["TP"].append(TP/self.nbFolds)
                res["FN"].append(FN/self.nbFolds)
                res["FP"].append(FP/self.nbFolds)

                #break
            #break
        
        self.dfResults = pd.DataFrame(res)
        
        
        
    def _saveResults(self):
        newFilename = newVersionFilename(self.resultsPath,self.name)
        self.saveFile = newFilename
        
        self.dfResults.to_csv(newFilename+".csv", index=False)
        
        params = {}
        params["hyperparams"] = json.dumps(self.hyperparams)
        params['file']=self.name
        params['results']=newFilename+".csv"
        params["learningType"] = self.learningType
        params["minSampleSize"] = self.minSampleSize
        params["maxSampleSize"] = self.maxSampleSize
        params["paceSampleSize"] = self.paceSampleSize
        params["nbThresholds"] = self.nbThresholds
        params["nbFolds"] = self.nbFolds
        params["perf"] = self.perf
        
        dfParamsUsed = pd.DataFrame.from_dict([params])
        
        # If params list does not exists, create it
        if not os.path.exists(self.resultsPath+"results-list.csv"):
            dfParamsUsed.to_csv(self.resultsPath+"results-list.csv", index=False)
        # If the list already exists, add the params used
        else:
            paramList = pd.read_csv(self.resultsPath+"results-list.csv")
            frames = [paramList, dfParamsUsed]
            paramList = pd.concat(frames)
            pd.DataFrame(paramList).to_csv(self.resultsPath+"results-list.csv", index=False)     
    
    
    def getResults(self):
        return self.dfResults
    
    def _setResults(self, results):
        self.dfResults = results
        
    def _setSaveFile(self, saveFile):
        self.saveFile = saveFile
    
    def _setMetrics(self):
        if self.dfResults is None:
            raise Exception('There is no results')
        
        result = {}
        
        result["Accuracy"] = (self.dfResults["TP"].mean()+self.dfResults["TN"].mean())/(self.dfResults["TP"].mean()+self.dfResults["TN"].mean()+self.dfResults["FP"].mean()+self.dfResults["FN"].mean())
        result["Precision"] = (self.dfResults["TP"].mean())/(self.dfResults["TP"].mean()+self.dfResults["FP"].mean())
        result["Recall"] = (self.dfResults["TP"].mean())/(self.dfResults["TP"].mean()+self.dfResults["FN"].mean())
        result["Specificity"] = (self.dfResults["TN"].mean())/(self.dfResults["TN"].mean()+self.dfResults["FP"].mean())
        result["NPV"] = (self.dfResults["TN"].mean())/(self.dfResults["TN"].mean()+self.dfResults["FN"].mean())
        
        self.result = result
    
    def getMetrics(self):
        if self.dfResults is None:
            raise Exception('There is no results')
        return self.result
    
    def start(self):
        if self.learningType == "classification":
            self._mlClassification()
        elif self.learningType == "regression":
            self._mlRegression()
        elif self.learningType.startswith("multiclass-"):
            self._mlMultiClassification(self.learningType.split("-")[1])
        self._saveResults()
        self._setMetrics()
        
    def setDataset(self, dataset):
        
        #Dataset handling
        #If dataset is a string, consider it as a path to a csv
        if type(dataset) == str:
            self.dataset = pd.read_csv(dataset)
        #If already a dataframe, keep it
        elif isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
            
    def setPerf(self, perf):
        self.perf = perf
        
    def setGraphPath(self, graphPath):
        #Create the folder for graphs if it does not exist:
        if not os.path.exists(graphPath):
            try:
                os.makedirs(graphPath)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        self.graphPath = graphPath
        
    def drawHeatmaps(self):
        if self.dfResults is None:
            raise Exception('There is no results')
            
        if self.graphPath == None:
            raise Exception('There is no graphPath defined')
        cmd = 'Rscript ./heatmaps.R '+self.resultsPath+''+self.saveFile+' '+self.graphPath
        return os.system(cmd)
    
    def getLearningParams(self):
        return {
            "learningType":self.learningType,
            "nbThresholds":self.nbThresholds,
            "nbFolds":self.nbFolds,
            "minSampleSize":self.minSampleSize,
            "paceSampleSize":self.paceSampleSize,
            "maxSampleSize":self.maxSampleSize,
        }
    
    def getHyperparams(self):
        return self.hyperparams
        
        
    def multiclassSeparator(self, df, t, nbClasses):
        df1 = df[df[self.perf] < t]
        df1 = df1.copy()
        df2 = df[df[self.perf] >= t]
        df2 = df2.copy()
        
        labelClass = 0
        subClasses = int(nbClasses/2)
        
        ltClasses = []
        for i in range(0,subClasses):
            subT = df1[self.perf].quantile((1/subClasses) * i)
            df1.loc[df1[self.perf] >= subT, "label"] = str(labelClass)
            ltClasses.append(str(labelClass))
            labelClass += 1
        
        gtClasses = []
        for i in range(0,subClasses):
            subT = df2[self.perf].quantile((1/subClasses) * i)
            df2.loc[df2[self.perf] >= subT, "label"] = str(labelClass)
            gtClasses.append(str(labelClass))
            labelClass += 1
            
        df = pd.concat([df1,df2])
        
        return df, ltClasses, gtClasses

def newVersionFilename(path, filename):
    # Get all the files in the {path} directory starting with {filename}
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.startswith(filename+"-")]
    files.sort(reverse=True)
    # If no file yet
    if len(files)==0:
        return path+filename+"-"+str(1).zfill(4)
    # Split the last one
    splitted = files[0].split("-")
    # Get the last version
    num = int(splitted[-1].split(".")[0])
    # Return the full name with new version
    return path+filename+"-"+str(num+1).zfill(4)


class MLThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        
        threadLimiter.acquire()
        try:
            if self._target is not None:
                self._return = self._target(*self._args,
                                                **self._kwargs)
        finally:
            threadLimiter.release()
        
       
    def join(self):
        Thread.join(self)
        return self._return