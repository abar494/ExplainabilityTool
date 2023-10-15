import os
import shutil
import numpy as np
from skimage.color import rgb2gray
from sklearn.pipeline import Pipeline
from PIL import Image
import importlib


class DataProcessor():
    def __init__(self, prefs, data_dict):
    
        # initialise class variables
        self.prefs = prefs
        self.iclf, self.clfs = None, None
        self.x_train, self.x_test = data_dict["x_train"], data_dict["x_test"]
        self.y_train, self.y_test = data_dict["y_train"], data_dict["y_test"] 

        # initialise pipeline dictionary
        self.pipelineDict = self.initialisePipelineDict()

        self.hps = prefs["hps"]
        self.hps["probability"] = True
        self.clfType = prefs["clf"]
        self.var = prefs["var"]
        self.vals = prefs["vals"]

        # fit the classifier to the data
        self.classify()

    def classify(self):

        # check if there are at least two classes in the target set
        if len(np.unique(self.y_train)) < 2:
            print("There needs to be at least two classes in the target set")
            return

        # By default, the pipeline will only have the flatten step
        self.pipes = [self.pipelineDict['Flatten Step']]

        # instantiate set of empty classifiers for later fitting.
        if self.vals is None:
            self.clfs = None
        else:
            self.clfs = [None for i in range(len(self.vals))]

        # sample max 120 images from the test set
        self.sample(120)

        # Notify the user that the classifier is being fit to the data
        print("please wait a little while the classifier is fit to the data")
        
        # instantiate the initial classifier
        self.iclf = self.getPipeline(self.pipes)

        # fit the initial classifier to the data
        self.iclf.fit(self.x_train, self.y_train)
        print(self.iclf.score(self.x_test, self.y_test))

        # remove the initial classifier from the pipeline
        self.pipes.pop()

        # check if the variable to be tuned is in the hyperparameters dictionary
        
        if self.var is None:
            self.ival = "None"
        else:
            
            if self.var in self.hps.keys():
                self.ival = self.hps[self.var]
            else:
                # if not, set the initial value to the default value
                self.ival = "Default"

            # iterate through the values of the variable to be tuned,
            # creating a new classifier for each value.
            for i in range(len(self.vals)):
                # set the value of the variable to be tuned to the current value
                self.hps[self.var] = self.vals[i]
                # instantiate the classifier
                self.clfs[i] = self.getPipeline(self.pipes)
                # fit the classifier to the data
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
        # Split the clfType string into a class and module name
        splittedClf = self.clfType.split('.')
        class_name = splittedClf[-1]
        module_name = '.'.join(splittedClf[:-1])
        clfPipeName = self.clfType.lower()
        try:
            # Dynamically import the module based on the clfType string
            classifier_module = importlib.import_module(module_name)
            # Get the class from the module
            constructor = getattr(classifier_module, class_name)
            clfTuple = (clfPipeName, constructor(**self.hps))
            pipes.append(clfTuple)
            return Pipeline(pipes)

        except (ImportError, AttributeError):
            raise ValueError(f"Could not import the module or find the class for {self.clfType}")


    def sample(self, numSamples):
        # upload a set number of test images to the sampledata folder
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
