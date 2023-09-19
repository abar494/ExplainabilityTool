from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QGroupBox, QScrollArea, QVBoxLayout
from PyQt5.QtWidgets import QAbstractButton, QLabel
import main
import classifierselect
import builtloader
import explainer
import matplotlib.pyplot as plt
from lime.wrappers.scikit_image import SegmentationAlgorithm
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QFont
from lime import lime_image
# from keras.models import load_model
from skimage.segmentation import mark_boundaries

class Gallery(QWidget):

    def backTransition(self):
        if self.fromClf:
            main.transition(
                self.stack, classifierselect.ClassifierSelect(self.stack, self.modelData))
        else:
            main.transition(
                self.stack, builtloader.ImageLoader(self.stack))

    def __init__(self, stack, modelData, fromClf):
        #Sets labels etc
        super(Gallery, self).__init__()

        self.stack = stack
        self.fromClf = fromClf

        self.modelData = modelData
        self.back = QPushButton("Back", self)
        self.back.clicked.connect(lambda: self.backTransition())
        
        gridLayout = QGridLayout()
        groupBox = QGroupBox()
        for i in range(min(len(self.modelData["x_test"]), 120)):
            button = self.PicButton(
                QPixmap("Datasets/sampledata/" + str(i) + ".png"), i)
            button.id = i
            button.clicked.connect(
                lambda: self.explain(stack, self.modelData))
            gridLayout.addWidget(button, (i)//3, (i)%3)

        groupBox.setLayout(gridLayout)
        scroll = QScrollArea()
        scroll.setWidget(groupBox)
        scroll.setWidgetResizable(False)
        scroll.setFixedHeight(400)
        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)


        self.title = QLabel("Gallery", self)
        self.title.setGeometry(460, 90, 181, 61)
        font = QFont()
        font.setPointSize(50)
        font.setBold(True)
        self.title.setFont(font)


        self.description = QLabel("Choose an Image to test the model on!", self)
        self.description.setGeometry(410, 220, 271, 111)
        font = QFont()
        font.setPointSize(30)
        self.description.setFont(font)
        self.description.setAlignment(Qt.AlignCenter)
        self.description.setWordWrap(True)

        self.show()

    def explain(self, stack, modelData):
        id = self.sender().id
        if self.fromClf:
            main.transition(
                stack, explainer.Explainer(stack, id, modelData))
        else:
            print(id)
            print(modelData["iclf"])
            print(modelData["y_test"][id])
            plt.imshow(modelData["x_test"][id])
            plt.show()

        # else:
        #     print(modelData["iclf"])
        #     model = load_model(modelData["iclf"], compile=False)

        #     explainer = lime_image.LimeImageExplainer(verbose = False)
        #     segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

        #     explanation = explainer.explain_instance(
        #         self.modelData["x_test"][id], 
        #         classifier_fn = model.predict_proba, 
        #         top_labels=10,
        #         hide_color=0,
        #         num_samples=2000,
        #         segmentation_fn=segmenter
        #     )
            
        #     temp, mask = explanation.get_image_and_mask(
        #         self.modelData["y_test"][id],
        #         positive_only=False,
        #         num_features=10,
        #         hide_rest=False,
        #         min_weight = 0.01
        #     )

        #     plt.imshow(mark_boundaries(temp, mask))
        #     plt.show()



    class PicButton(QAbstractButton):
        def __init__(self, pixmap, id, parent=None):
            super(Gallery.PicButton, self).__init__(parent)
            self.pixmap = pixmap
            self.id = id

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.drawPixmap(event.rect(), self.pixmap)

        def sizeHint(self):
            return self.pixmap.size()