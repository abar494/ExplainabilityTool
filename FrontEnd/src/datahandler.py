import main
import classifierselect
import gallery
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget
import glob
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, stack, x_test=None, y_test=None, clf=None):
        super(DataViewer, self).__init__()
        loadUi('FrontEnd/UI/DataViewer.ui', self)
        self.clfs = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.backButton.clicked.connect(lambda: main.transition(stack, main.MainMenu(stack)))
        self.classifyButton.clicked.connect(self.defaultClassify)
        self.explainButton.clicked.connect(lambda: main.transition(stack, gallery.Gallery(stack, self.x_test, self.y_test, self.clfs)))
        self.plotButton.clicked.connect(self.catClassify)
        self.show()

    def enumerate(self, animal):
        if animal == 'cats':
            num = 0
        elif animal == 'dogs':
            num = 1
        else:
            num = 2
        return num

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
            labels.append(self.enumerate(animal))
        data = (255*np.stack(x)).astype(np.uint8)
        
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

        flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

        # clf = Pipeline([
        #     ('Flatten Image', flatten_step),
        #     ('svc', SVC(probability=True))]) # SVC uses default params found here https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

        self.clfs = [[[None for i in range(3)] for i in range(3)] for i in range(3)]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, np.ravel(labels), train_size=0.8)
        self.sample(120)
        # clf.fit(self.x_train, self.y_train)
        # clf.fit(self.x_train, self.y_train)
        # print("done")

        # print(labels[34])

        # print(clf.score(self.x_test, self.y_test))
        # pred = self.enumerate(clf.predict(self.x_test)[3])
        
        params = {
            "C": [1, 3, 5],
            "kernel": ['linear', 'poly', 'rbf'],
            "gamma": [0.5, 2, 5]
        }
        print("please wait a little while the classifier is fit to the data")

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.clfs[i][j][k] = Pipeline([
                        # ('Make Gray', makegray_step),
                        ('Flatten Image', flatten_step),
                        ('svc', SVC(probability=True, C=params["C"][i],
                                     kernel=params["kernel"][j],
                                     gamma=params["gamma"][k]))]) # SVC uses default params found here https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

                    self.clfs[i][j][k].fit(self.x_train, self.y_train)
                    print(self.clfs[i][j][k].score(self.x_test, self.y_test))
                    # prediction = np.zeros((784, 3))
                    
                    # predictions = self.clfs[i][j].predict(self.x_test)

                # img = self.x_test[3]
                # for i in range(len(img)):
                #     for j in range(len(img)):
                #         for k in range(3):
                #             prediction[(len(img)-1)*i + j][k] = img[i][j][k]

                #         prediction.append(element)
                # prediction = np.array(prediction).T

                # print(prediction.shape)

                # print(self.clfs[i][j].predict(prediction))
                # print(self.y_test[3])

        print("done")



        # self.explain(self.x_test[3], pred, clf)

        # print(clf.predict(self.x_test)[3])
        # print(self.y_test[3])






        # print(self.x_test[0])
        # print(clf.predict(self.x_test[0]))
        # print(self.y_test[0])



    # def explain(self, x, pred, clf):
    #     explainer = lime_image.LimeImageExplainer(verbose = False)
    #     segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    #     explanation = explainer.explain_instance(x, 
    #                                     classifier_fn = clf.predict_proba, 
    #                                     top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)

    #     temp, mask = explanation.get_image_and_mask(pred, positive_only=True, num_features=10, hide_rest=True, min_weight = 0.01)
    #     fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
    #     # ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
    #     ax1.imshow(mark_boundaries(temp, mask))
    #     # ax1.set_title('Positive Regions for {}'.format(y_test[id]))
    #     temp, mask = explanation.get_image_and_mask(pred, positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
    #     ax2.imshow(mark_boundaries(temp, mask))
    #     # ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
    #     # ax2.set_title('Positive/Negative Regions for {}'.format(y_test[id]))
    #     ax1.axis('off')
    #     ax2.axis('off')
    #     plt.show()




    def defaultClassify(self):
        mnist = loadmat(r"Datasets/MNIST/mnist-original.mat")
        data, labels = mnist["data"].T, mnist["label"].T
        random.Random(6).shuffle(data)
        random.Random(6).shuffle(labels)
        data, labels = data[:8000], labels[:8000]
        data = np.stack([gray2rgb(iimg)
                        for iimg in data.reshape((-1, 28, 28))], 0).astype(np.uint8)
        
        labels = labels.astype(np.uint8)
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
            
        makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
        flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

        self.clfs = [[[None for i in range(3)] for i in range(3)] for i in range(3)]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, np.ravel(labels), train_size=0.8)

        # pred = self.enumerate(clf.predict(self.x_test)[3])
        if len(np.unique(self.y_train)) < 2:
            print("There needs to be at least two classes in the target set")
            return

        self.sample(120)
        params = {
            "C": [1, 3, 5],
            "kernel": ['rbf', 'rbf', 'rbf'],
            "gamma": [0.5, 2, 3]
        }
        print("please wait a little while the classifier is fit to the data")


        num = 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.clfs[i][j][k] = Pipeline([
                        ('Make Gray', makegray_step),
                        ('Flatten Image', flatten_step),
                        ('svc', SVC(probability=True, C=params["C"][i],
                                     kernel=params["kernel"][j],
                                     gamma=params["gamma"][k]))]) # SVC uses default params found here https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

                    # self.clfs[i][j][k] = num
                    # num += 1
                    # print(self.clfs[i][j][k])

                    self.clfs[i][j][k].fit(self.x_train, self.y_train)
                    # print(self.clfs[i][j][k].score(self.x_test, self.y_test))
                    print(num)
                    num+=1
                    # prediction = np.zeros((784, 3))
                    
                    # predictions = self.clfs[i][j].predict(self.x_test)

        print("done")


        # self.clfs = [
        #     [None, None, None],
        #     [None, None, None],
        #     [None, None, None]
        # ]
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, np.ravel(label), train_size=0.65)
        # params = {
        #     "C": [1, 3, 5],
        #     "kernel": ['linear', 'poly', 'rbf']
        # }
        # print("please wait a little while the classifier is fit to the data")

        # for i in range(3):
        #     for j in range(3):
        #         self.clfs[i][j] = Pipeline([
        #             ('Make Gray', makegray_step),
        #             ('Flatten Image', flatten_step),
        #             ('svc', SVC(probability=True, C=params["C"][i], kernel=params["kernel"][j]))]) # SVC uses default params found here https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

        #         self.clfs[i][j].fit(self.x_train, self.y_train)
        #         print(self.clfs[i][j].score(self.x_test, self.y_test))
        #         # prediction = np.zeros((784, 3))
                
        #         # predictions = self.clfs[i][j].predict(self.x_test)
        #         print("debug")

                # img = self.x_test[3]
                # for i in range(len(img)):
                #     for j in range(len(img)):
                #         for k in range(3):
                #             prediction[(len(img)-1)*i + j][k] = img[i][j][k]

                #         prediction.append(element)
                # prediction = np.array(prediction).T

                # print(prediction.shape)

                # print(self.clfs[i][j].predict(prediction))
                # print(self.y_test[3])


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


    def plot(self):
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(self.x_test[0], interpolation="none") 
        ax1.set_title(f"Digit: {self.y_test[0]}" )
        plt.show()

        print(self.x_test[0])
        print(self.y_test[0])