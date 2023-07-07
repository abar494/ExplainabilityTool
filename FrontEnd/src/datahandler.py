import main
import classifierselect
import gallery
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget
import glob
import os
import shutil
import numpy as np
from scipy.io import loadmat
from skimage.color import gray2rgb, rgb2gray
import random
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage.transform import resize
import glob


class DataSelect(QWidget):
    def __init__(self, stack):
        super(DataSelect, self).__init__()
        loadUi('FrontEnd/UI/DataSelect.ui', self)
        self.sampleData.clicked.connect(lambda: main.transition(stack, DataViewer(stack)))
        self.loadData.clicked.connect(lambda: main.transition(stack, classifierselect.ClassifierSelect(stack)))
        self.back.clicked.connect(lambda: main.transition(stack, main.MainMenu(stack)))
        self.show()


class DataViewer(QWidget):
    def __init__(self, stack, x_test=None, y_test=None, clfs=None):
        super(DataViewer, self).__init__()
        loadUi('FrontEnd/UI/DataViewer.ui', self)
        self.x_test, self.y_test, self.clfs = x_test, y_test, clfs
        self.p = {
            "C": [1, 3, 5],
            "kernel": ['linear', 'poly', 'poly'],
            "gamma": [0.5, 2, 5]
        }
        self.pipelineDict = self.initialisePipelineDict()
        self.backButton.clicked.connect(
            lambda: main.transition(stack, main.MainMenu(stack)))
        self.classifyButton.clicked.connect(self.defaultClassify)
        self.explainButton.clicked.connect(
            lambda: main.transition(
            stack, gallery.Gallery(stack, self.x_test, self.y_test, self.clfs)))
        self.plotButton.clicked.connect(self.catClassify)
        
        self.show()

    def enumerate(self, animal, animalList):
        for i in range(len(animalList)):
            if animal == animalList[i]:
                return i

    def catClassify(self):
        fileList = glob.glob("Datasets/catdogpanda/*")
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

    def defaultClassify(self):
        mnist = loadmat(r"Datasets/MNIST/mnist-original.mat")
        data, labels = mnist["data"].T, mnist["label"].T
        random.Random(6).shuffle(data)
        random.Random(6).shuffle(labels)
        data, labels = data[:8000], labels[:8000]
        data = np.stack([gray2rgb(iimg)
                         for iimg in data.reshape((-1, 28, 28))], 0).astype(np.uint8)
        labels = labels.astype(np.uint8)
        self.pipes = [self.pipelineDict['Make Gray'], self.pipelineDict['Flatten Step']]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, np.ravel(labels), train_size=0.8)
        self.classify()

    def classify(self):
        if len(np.unique(self.y_train)) < 2:
            print("There needs to be at least two classes in the target set")
            return

        self.clfs = [[[None for i in range(3)] for i in range(3)] for i in range(3)]
        self.sample(120)
        print("please wait a little while the classifier is fit to the data")
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.clfs[i][j][k] = self.getPipeline(self.pipes, (i,j,k))
                    self.clfs[i][j][k].fit(self.x_train, self.y_train)
                    print(self.clfs[i][j][k].score(self.x_test, self.y_test))
                    self.pipes.pop()
        print("done")

    def initialisePipelineDict(self):
        pipelineDict = {}
        pipelineDict['Flatten Step'] = ('Flatten Step',
                        self.PipeStep(lambda img_list: [img.ravel() for img in img_list]))
        pipelineDict['Make Gray'] = ('Make Gray',
                         self.PipeStep(lambda img_list: [rgb2gray(img) for img in img_list]))
        return pipelineDict
    
    def getPipeline(self, pipes, coords):
        clfTuple = ('svc', SVC(probability=True, C=self.p["C"][coords[0]],
                                     kernel=self.p["kernel"][coords[1]],
                                     gamma=self.p["gamma"][coords[2]]))
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
