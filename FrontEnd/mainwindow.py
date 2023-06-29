# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from PyQt5.uic import loadUi
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# since the ImageExplanation(image, segments) wants 3D numpy arrays (colour images) for segments
from skimage.color import gray2rgb, rgb2gray, label2rgb
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import random
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def transition(context):
    widget.addWidget(context)
    nextIndex = widget.currentIndex()
    widget.removeWidget(widget.currentWidget())
    widget.setCurrentIndex(nextIndex)

class MainMenu(QMainWindow):
    def __init__(self, parent=None):
        #Sets labels etc
        super(MainMenu, self).__init__()
        loadUi('FrontEnd/UI/MainMenu.ui', self)
        self.title = QLabel('P4P XAI Tool', self)
        
        self.title.setGeometry(50,50,1000,75)
        self.title.setFont(QFont("Comic Sans MS", 40))
        self.title.move(100, 30)

        #This initialises the ui
        self.initUI()

    def initUI(self):
        #labels
        self.lbl = QLabel('', self)
        self.lbl.move(75, 200)
        self.defaultCLF.clicked.connect(lambda: transition(DataViewer()))
        self.trainCLF.clicked.connect(lambda: transition(DataSelect()))
        self.exit.clicked.connect(self.closeApp)
        self.show()

    def closeApp(self):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:        QApplication.quit()
        else:                               QMessageBox.Close

class DataSelect(QWidget):
    def __init__(self):
        super(DataSelect, self).__init__()
        loadUi('FrontEnd/UI/DataSelect.ui', self)
        self.pipe = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.sampleData.clicked.connect(lambda: transition(DataViewer()))
        self.back.clicked.connect(lambda: transition(MainMenu()))
        self.show()

class DataViewer(QWidget):
    def __init__(self):
        super(DataViewer, self).__init__()
        loadUi('FrontEnd/UI/DataViewer.ui', self)
        self.pipe = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.backButton.clicked.connect(lambda: transition(MainMenu()))
        self.classifyButton.clicked.connect(self.defaultClassify)
        self.explainButton.clicked.connect(self.explain)
        self.plotButton.clicked.connect(self.plot)
        self.show()

    def defaultClassify(self):
        mnist = loadmat(r"Datasets/MNIST/mnist-original.mat")
        data, label = mnist["data"].T, mnist["label"].T
        random.Random(6).shuffle(data)
        random.Random(6).shuffle(label)
        data, label = data[:15000], label[:15000]
        data = np.stack([gray2rgb(iimg)
                        for iimg in data.reshape((-1, 28, 28))], 0).astype(np.uint8)
        label = label.astype(np.uint8)
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
        self.pipe = Pipeline([
            ('Make Gray', makegray_step),
            ('Flatten Image', flatten_step),
            ('svc', SVC(probability=True))]) # SVC uses default params found here https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, np.ravel(label), train_size=0.65)
        print("please wait a little while the classifier is fit to the data")
        self.pipe.fit(self.x_train, self.y_train)
        print(self.pipe.score(self.x_test, self.y_test))
        print("done")

    def plot(self):
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(self.x_test[0], interpolation="none") 
        ax1.set_title(f"Digit: {self.y_test[0]}" )
        plt.show()

        print(self.x_test[0])
        print(self.y_test[0])

    def explain(self):
        explainer = lime_image.LimeImageExplainer(verbose = False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

        explanation = explainer.explain_instance(self.x_test[0], 
                                        classifier_fn = self.pipe.predict_proba, 
                                        top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)

        temp, mask = explanation.get_image_and_mask(self.y_test[0], positive_only=True, num_features=10, hide_rest=False, min_weight = 0.01)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
        ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
        ax1.set_title('Positive Regions for {}'.format(self.y_test[0]))
        temp, mask = explanation.get_image_and_mask(self.y_test[0], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
        ax2.set_title('Positive/Negative Regions for {}'.format(self.y_test[0]))
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    mainMenu = MainMenu()
    widget.addWidget(mainMenu)
    widget.setFixedHeight(500)
    widget.setFixedWidth(600)
    widget.show()
    sys.exit(app.exec_())
