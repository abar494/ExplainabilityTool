# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QDrag
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QFont, QPixmap, QPainter
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
from PIL import Image

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

        self.defaultCLF.clicked.connect(lambda: transition(DataViewer()))
        self.loadCLF.clicked.connect(lambda: transition(Gallery()))
        self.trainCLF.clicked.connect(lambda: transition(DataSelect()))
        self.exit.clicked.connect(self.closeApp)
        self.show()

    def closeApp(self):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:        QApplication.quit()
        else:                               QMessageBox.Close


class DragButton(QPushButton):

    id = None

    # def __init__(self, id):
    #     super(DragButton, self).__init__()
    #     self.id

    def mouseMoveEvent(self, e):

        if e.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)

            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)
    
    def getID(self):
        return self.id


class ClassifierSelect(QWidget):

    def getButtonID(self):
        id = self.sender().getID()
        pos = self.sender().geometry()        
        print("You have clicked button " + str(id))
        print("at position (" + str(pos.x())+", "+str(pos.y())+")")

    def __init__(self):
        super(ClassifierSelect, self).__init__()
        self.setAcceptDrops(True)
        back = QPushButton("Howdy", self)
        back.clicked.connect(lambda: transition(DataSelect()))

        self.blayout = QHBoxLayout()
        for l in ['A', 'B', 'C', 'D']:
            btn = DragButton(l)
            btn.id = l
            btn.clicked.connect(lambda: self.getButtonID())
            self.blayout.addWidget(btn)

        self.setLayout(self.blayout)
        self.show() 

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        pos = e.pos()
        widget = e.source()
        widget.move(pos)
        
        # print(self.blayout)
        # print(self.blayout.count())
        # for n in range(self.blayout.count()):
        #     # Get the widget at each index in turn.
        #     w = self.blayout.itemAt(n).widget()
        #     if pos.x() < w.x() + w.size().width() // 2:
        #         # We didn't drag past this widget.
        #         # insert to the left of it.
        #         self.blayout.insertWidget(n-1, widget)
        #         break

        # e.accept()



class DataSelect(QWidget):
    def __init__(self):
        super(DataSelect, self).__init__()
        loadUi('FrontEnd/UI/DataSelect.ui', self)
        self.pipe = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.sampleData.clicked.connect(lambda: transition(DataViewer()))
        self.loadData.clicked.connect(lambda: transition(ClassifierSelect()))
        self.back.clicked.connect(lambda: transition(MainMenu()))
        self.show()

class DataViewer(QWidget):
    def __init__(self, x_test=None, y_test=None, clf=None):
        super(DataViewer, self).__init__()
        loadUi('FrontEnd/UI/DataViewer.ui', self)
        self.pipe = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.backButton.clicked.connect(lambda: transition(MainMenu()))
        self.classifyButton.clicked.connect(self.defaultClassify)
        self.explainButton.clicked.connect(lambda: transition(Gallery(self.x_test, self.y_test, self.pipe)))
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

        for i in range(120):
            im = Image.fromarray(self.x_test[i])
            im.save("Datasets/sampledata/"+str(i)+".png")

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


class PicButton(QAbstractButton):
    def __init__(self, pixmap, id=0, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap
        self.id = id

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(event.rect(), self.pixmap)

    def sizeHint(self):
        return self.pixmap.size()
    
    def getID(self):
        return self.id
    

class Gallery(QWidget):

    def __init__(self, x_test, y_test, clf):
        #Sets labels etc
        super(Gallery, self).__init__()
        # loadUi('FrontEnd/UI/Gallery.ui', self)

        back = QPushButton("Back", self)
        back.clicked.connect(lambda: transition(DataViewer(x_test, y_test, clf)))

        gridLayout = QGridLayout()
        groupBox = QGroupBox()

        for i in range(120):
            button = PicButton(QPixmap("Datasets/sampledata/" + str(i) + ".png"))
            button.id = i
            button.clicked.connect(lambda: self.explain(x_test, y_test, clf))
            # button = QPushButton(str(i), self)
            gridLayout.addWidget(button, (i)//3, (i)%3)
        
        groupBox.setLayout(gridLayout)

        scroll = QScrollArea()
        scroll.setWidget(groupBox)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(400)

        layout = QVBoxLayout()
        layout.addWidget(scroll)

        self.setLayout(layout)


        # self.defaultCLF.clicked.connect(lambda: transition(DataViewer()))
        # self.trainCLF.clicked.connect(lambda: transition(DataSelect()))
        # self.exit.clicked.connect(self.closeApp)
        self.show()


    def explain(self, x_test, y_test, clf):
        id = self.sender().getID()
        explainer = lime_image.LimeImageExplainer(verbose = False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

        explanation = explainer.explain_instance(x_test[id], 
                                        classifier_fn = clf.predict_proba, 
                                        top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)

        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=True, num_features=10, hide_rest=False, min_weight = 0.01)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
        ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
        ax1.set_title('Positive Regions for {}'.format(y_test[id]))
        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
        ax2.set_title('Positive/Negative Regions for {}'.format(y_test[id]))
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
