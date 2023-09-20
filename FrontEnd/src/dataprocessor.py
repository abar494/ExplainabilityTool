import os
import shutil
import numpy as np
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from PIL import Image
import glob


class DataProcessor():
    def __init__(self, prefs, data_dict):
    
        self.prefs = prefs
        self.iclf, self.clfs = None, None
        self.x_train, self.x_test = data_dict["x_train"], data_dict["x_test"]
        self.y_train, self.y_test = data_dict["y_train"], data_dict["y_test"] 

        
        self.pipelineDict = self.initialisePipelineDict()

        self.hps = prefs["hps"]
        self.hps["probability"] = True
        self.clfType = prefs["clf"]
        self.var = prefs["var"]
        self.vals = prefs["vals"]

        self.classify()

    def classify(self):
        if len(np.unique(self.y_train)) < 2:
            print("There needs to be at least two classes in the target set")
            return

        self.pipes = [self.pipelineDict['Flatten Step']]

        self.clfs = [None for i in range(len(self.vals))]
        self.sample(120)
        print("please wait a little while the classifier is fit to the data")
        self.iclf = self.getPipeline(self.pipes)
        self.iclf.fit(self.x_train, self.y_train)
        print(self.iclf.score(self.x_test, self.y_test))
        self.pipes.pop()

        if self.var in self.hps.keys():
            self.ival = self.hps[self.var]
        else:
            self.ival = "Default"

        for i in range(len(self.vals)):
            self.hps[self.var] = self.vals[i]
            self.clfs[i] = self.getPipeline(self.pipes)
            self.clfs[i].fit(self.x_train, self.y_train)
            print(self.clfs[i].score(self.x_test, self.y_test))
            self.pipes.pop()
        print("done")


    def initialisePipelineDict(self):
        pipelineDict = {}
        pipelineDict['Flatten Step'] = ('Flatten Step',
                        self.PipeStep(lambda img_list: [img.ravel() for img in img_list]))
        pipelineDict['Make Gray'] = ('Make Gray',
                            self.PipeStep(lambda img_list: [rgb2gray(img) for img in img_list]))
        return pipelineDict

    def getPipeline(self, pipes):
        clfPipeName = self.clfType.lower()
        constructor = globals()[self.clfType]
        clfTuple = (clfPipeName, constructor(**self.hps))
        pipes.append(clfTuple)
        return Pipeline(pipes)

    def sample(self, numSamples):
        folder = "Datasets/sampledata/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        for i in range(min(len(self.x_test), numSamples)):
            im = Image.fromarray(self.x_test[i])
            im.save("Datasets/sampledata/"+str(i)+".png")

    class PipeStep(object):
        """
        Wrapper for turning functions into pipeline transforms (no-fitting)
        """
        def __init__(self, step_func):
            self._step_func=step_func
        def fit(self,*args):
            return self
        def transform(self,X):
            return self._step_func(X)
