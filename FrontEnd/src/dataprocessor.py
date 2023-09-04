import glob
import os
import shutil
import numpy as np
from scipy.io import loadmat
from skimage.color import gray2rgb, rgb2gray
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage.transform import resize
import glob
# fromimport stack as npstack

class DataProcessor():
    def __init__(self, prefs):
    
        self.prefs = prefs
        self.x_test, self.y_test, self.iclf, self.clfs = None, None, None, None

        self.pipelineDict = self.initialisePipelineDict()

        self.hps = prefs["hps"]
        self.hps["probability"] = True
        self.clfType = prefs["clf"]
        self.var = prefs["var"]
        self.vals = prefs["vals"]

        self.catClassify()


    def enumerate(self, animal, animalList):
            for i in range(len(animalList)):
                if animal == animalList[i]:
                    return i

    def catClassify(self):
        fileList = glob.glob("Datasets/animals/cat/*")
        fileList = fileList[:100]       
        x, labels = [], []

        for fname in fileList:
            img = resize(np.array(Image.open(fname)), (100, 100))
            if img.shape != (100, 100, 3):
                img = gray2rgb(img)
            x.append(img)
            animal = fname.split("_")[0].split("/")[-1]
            labels.append(self.enumerate(animal, ['cats', 'dogs', 'panda']))
        data = (255*np.stack(x)).astype(np.uint8)
        self.pipes = [self.pipelineDict['Flatten Step']]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, np.ravel(labels), train_size=0.8)
        self.classify()

    def enumerate(self, animal, animalList):
        for i in range(len(animalList)):
            if animal == animalList[i]:
                return i


    def classify(self):
        if len(np.unique(self.y_train)) < 2:
            print("There needs to be at least two classes in the target set")
            return

        self.clfs = [None for i in range(len(self.vals))]
        self.sample(120)
        print("please wait a little while the classifier is fit to the data")
        self.iclf = self.getPipeline(self.pipes)
        self.iclf.fit(self.x_train, self.y_train)
        print(self.iclf.score(self.x_test, self.y_test))
        self.pipes.pop()

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
        
    
    # def defaultClassify(self):
    #     mnist = loadmat(r"Datasets/MNIST/mnist-original.mat")
    #     data, labels = mnist["data"].T, mnist["label"].T
    #     random.Random(6).shuffle(data)
    #     random.Random(6).shuffle(labels)
    #     data, labels = data[:8000], labels[:8000]
    #     data = np.stack([gray2rgb(iimg)
    #                         for iimg in data.reshape((-1, 28, 28))], 0).astype(np.uint8)
    #     labels = labels.astype(np.uint8)
    #     self.pipes = [self.pipelineDict['Make Gray'], self.pipelineDict['Flatten Step']]
    #     self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, np.ravel(labels), train_size=0.8)
    #     self.classify()
