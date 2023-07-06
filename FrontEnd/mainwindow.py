# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QDrag
from PyQt5.QtCore import Qt, QMimeData, QPoint
from PyQt5.QtGui import QFont, QPixmap, QPainter
from PyQt5.uic import loadUi
import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# since the ImageExplanation(image, segments) wants 3D numpy arrays (colour images) for segments
from skimage.color import gray2rgb, rgb2gray, label2rgb
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import random
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage.transform import resize
import glob


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


# class DragButton(QPushButton):

#     id = None

#     # def __init__(self, id):
#     #     super(DragButton, self).__init__()
#     #     self.id

#     def mouseMoveEvent(self, e):

#         if e.buttons() == Qt.LeftButton:
#             drag = QDrag(self)
#             mime = QMimeData()
#             drag.setMimeData(mime)

#             pixmap = QPixmap(self.size())
#             self.render(pixmap)
#             drag.setPixmap(pixmap)

#             drag.exec_(Qt.MoveAction)
    
#     def getID(self):
#         return self.id










class DraggableLabel(QLabel):

    id = None

    def __init__(self, name, coords, mainWindow):
        super(DraggableLabel, self).__init__(name, mainWindow)
        self.move(coords[0], coords[1])
        self.hasLanded = True
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()
        self.initialPos = self.mapFromGlobal(self.cursor().pos())
        print(self.initialPos)
        # print("Local position:", local_pos)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mimedata = QMimeData()
        mimedata.setText(self.text())
        drag.setMimeData(mimedata)
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())
        drag.exec_(Qt.CopyAction | Qt.MoveAction)

    def getID(self):
        return self.id


class DropZone(QLabel):
    def __init__(self, bing, mainWindow):
        super(DropZone, self).__init__(str(bing), mainWindow)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        pos = event.pos()
        geom = self.geometry()
        point = QPoint(geom.x(), geom.y())
        print(pos)
        widget = event.source()
        widget.move(pos - widget.initialPos + point)
        print("landed")
        event.acceptProposedAction()
        widget.hasLanded = True

class ClassifierSelect(QWidget):

    def getButtonID(self):
        id = self.sender().getID()
        pos = self.sender().geometry()        
        print("You have clicked button " + str(id))
        print("at position (" + str(pos.x())+", "+str(pos.y())+")")

    def __init__(self):
        super().__init__()
        label = QLabel("--------------------------------------------------------------------------------------------------------", self)
        label2 = QLabel("--------------------------------------------------------------------------------------------------------", self)
        label3 = QLabel("|", self)
        label.adjustSize()
        label.move(0, 50)
        label2.move(0, 200)
        label3.move(300, 50)
        dropTo = DropZone("", self)
        dropTo.setGeometry(300, 50, 300,150)
        self.setAcceptDrops(True)
        label_to_drag = DraggableLabel("drag this", (100, 475), self) 
        # label_to_drag.move(40, 40)
        hasLanded = QPushButton("Landed", self)
        hasLanded.move(0, 475)
        hasLanded.clicked.connect(lambda: print(label_to_drag.hasLanded))
        back = QPushButton("Back", self)
        back.clicked.connect(lambda: transition(DataSelect()))
        self.show()
    
    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        pos = e.pos()
        print(pos)
        widget = e.source()
        widget.move(pos - widget.initialPos)
        widget.hasLanded = False
        e.acceptProposedAction()













# class ClassifierSelect(QWidget):

#     def getButtonID(self):
#         id = self.sender().getID()
#         pos = self.sender().geometry()        
#         print("You have clicked button " + str(id))
#         print("at position (" + str(pos.x())+", "+str(pos.y())+")")








    # def __init__(self):
    #     super(ClassifierSelect, self).__init__()
    #     loadUi('FrontEnd/UI/Gallery.ui', self)
    #     self.setAcceptDrops(True)
    #     # back = QPushButton("Howdy", self)
    #     self.back.clicked.connect(lambda: transition(DataSelect()))

    #     self.blayout = QHBoxLayout()
    #     for l in ['A', 'B', 'C', 'D']:
    #         btn = DragButton(l)
    #         btn.id = l
    #         btn.clicked.connect(lambda: self.getButtonID())
    #         self.blayout.addWidget(btn)

    #     self.setLayout(self.blayout)
    #     self.show() 

    # def dragEnterEvent(self, e):
    #     e.accept()

    # def dropEvent(self, e):
    #     pos = e.pos()
    #     widget = e.source()
    #     widget.move(pos)

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
        self.sampleData.clicked.connect(lambda: transition(DataViewer()))
        self.loadData.clicked.connect(lambda: transition(ClassifierSelect()))
        self.back.clicked.connect(lambda: transition(MainMenu()))
        self.show()

class DataViewer(QWidget):
    def __init__(self, x_test=None, y_test=None, clf=None):
        super(DataViewer, self).__init__()
        loadUi('FrontEnd/UI/DataViewer.ui', self)
        self.clfs = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.backButton.clicked.connect(lambda: transition(MainMenu()))
        self.classifyButton.clicked.connect(self.defaultClassify)
        self.explainButton.clicked.connect(lambda: transition(Gallery(self.x_test, self.y_test, self.clfs)))
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

    def __init__(self, x_test, y_test, clfs):
        #Sets labels etc
        super(Gallery, self).__init__()
        back = QPushButton("Back", self)
        back.clicked.connect(lambda: transition(DataViewer(x_test, y_test, clfs)))

        gridLayout = QGridLayout()
        groupBox = QGroupBox()

        for i in range(min(len(x_test), 120)):
            button = PicButton(QPixmap("Datasets/sampledata/" + str(i) + ".png"))
            button.id = i
            button.clicked.connect(lambda: self.explainerTransition(x_test, y_test, clfs))
            # button.clicked.connect(lambda: self.explain(x_test, y_test, clf))
            # button = QPushButton(str(i), self)
            gridLayout.addWidget(button, (i)//3, (i)%3)
        
        groupBox.setLayout(gridLayout)

        scroll = QScrollArea()
        scroll.setWidget(groupBox)
        scroll.setWidgetResizable(False)
        scroll.setFixedHeight(400)

        layout = QVBoxLayout()
        layout.addWidget(scroll)

        self.setLayout(layout)


        # self.defaultCLF.clicked.connect(lambda: transition(DataViewer()))
        # self.trainCLF.clicked.connect(lambda: transition(DataSelect()))
        # self.exit.clicked.connect(self.closeApp)
        self.show()

    def explainerTransition(self, x_test, y_test, clfs):
        id = self.sender().getID()
        transition(Explainer(id, x_test, y_test, clfs))


class Explainer(QWidget):

    def __init__(self, id, x_test, y_test, clfs):
        #Sets labels etc
        super(Explainer, self).__init__()
        loadUi('FrontEnd/UI/Explainer.ui', self)
        back = QPushButton("Back", self)
        back.clicked.connect(lambda: transition(Gallery(x_test, y_test, clfs)))
        # self.go.clicked.connect(lambda: self.explain(id, x_test, y_test,
        #                                               clfs[self.slider1.value()][self.slider2.value()]))

        self.go.clicked.connect(lambda: self.printThenExplain(id, x_test, y_test, clfs))

        # self.defaultCLF.clicked.connect(lambda: transition(DataViewer()))
        # self.trainCLF.clicked.connect(lambda: transition(DataSelect()))
        # self.exit.clicked.connect(self.closeApp)
        self.show()

    
    def printThenExplain(self, id, x_test, y_test, clfs):
        # print(self.slider1.value())
        # print(self.slider2.value())
        # print(clfs[self.slider1.value()][self.slider2.value()])
        self.explain(id, x_test, y_test, clfs[self.slider1.value()][self.slider2.value()][self.slider3.value()])

    def explain(self, id, x_test, y_test, clf):
        explainer = lime_image.LimeImageExplainer(verbose = False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

        explanation = explainer.explain_instance(x_test[id], 
                                        classifier_fn = clf.predict_proba, 
                                        top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)

        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=True, num_features=5, hide_rest=True, min_weight = 0.01)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
        # ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
        ax1.imshow(mark_boundaries(temp, mask))
        # ax1.set_title('Positive Regions for {}'.format(y_test[id]))
        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        ax2.imshow(mark_boundaries(temp, mask))
        # ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
        # ax2.set_title('Positive/Negative Regions for {}'.format(y_test[id]))
        ax1.axis('off')
        ax2.axis('off')
        plt.show()


if __name__ == "__main__":
    if not os.path.isdir("Datasets/sampledata"):
        os.mkdir("Datasets/sampledata")
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    mainMenu = MainMenu()
    widget.addWidget(mainMenu)
    widget.setFixedHeight(500)
    widget.setFixedWidth(600)
    widget.show()
    ret = app.exec_()
    shutil.rmtree('Datasets/sampledata')
    sys.exit(ret)



