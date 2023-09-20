import main
import gallery
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, \
    QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QRadioButton
from PyQt5.QtGui import QFont, QPixmap
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

class Explainer(QWidget):


    def __init__(self, stack, id, modelData, fromClf):
        super(Explainer, self).__init__()

        x_test, y_test, iclf, clfs = modelData["x_test"], modelData["y_test"], modelData["iclf"], modelData["clfs"] 
        back = QPushButton("Back", self)
        # self.hpSlider = self.LabeledSlider(minimax=(0, len(modelData["vals"])-1), labels=modelData["vals"], parent=self)
        self.go = QPushButton("Go", self)
        self.go.setGeometry(580, 400, 132, 32)
        self.go.hide()
        # self.hpSlider.move(30, 60)

        self.modelData = modelData
        self.varLabel = QLabel("Variable " + modelData["var"] + " values:", self)
        font = QFont()
        font.setPointSize(15)
        self.varLabel.setGeometry(30, 310, 130, 61)
        self.varLabel.setWordWrap(True)
        self.varLabel.setFont(font)

        self.sliderLayout = QWidget(self)
        self.sliderLayout.setGeometry(140, 310, 421, 62)
        self.sliderLayout.setObjectName("sliderLayout")
        self.sliderBox = QHBoxLayout(self.sliderLayout)
        self.sliderBox.setContentsMargins(0, 0, 0, 0)
        self.sliderBox.setObjectName("sliderBox")

        self.currentClf = None
        #Iter 1:
        i = 0
        for val in modelData["vals"]:
            self.Vbox = QVBoxLayout()
            self.Vbox.setObjectName("Vbox")

            Hbox = QHBoxLayout()
            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            Hbox.addItem(spacerItem)
            self.hpValue = QLabel(self.sliderLayout)
            self.hpValue.setObjectName("Name: " + str(i))
            self.hpValue.setText(str(val))
            Hbox.addWidget(self.hpValue)
            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            Hbox.addItem(spacerItem)
            self.Vbox.addLayout(Hbox)

            Hbox = QHBoxLayout()
            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            Hbox.addItem(spacerItem)
            self.valButton = QRadioButton(self.sliderLayout)
            self.valButton.setText("")
            self.valButton.setObjectName("Button: " + str(i))
            self.valButton.clicked.connect(lambda: self.setCurrentClf())

            Hbox.addWidget(self.valButton)
            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            Hbox.addItem(spacerItem)
            self.Vbox.addLayout(Hbox)

            self.sliderBox.addLayout(self.Vbox)
            i += 1


        self.title = QLabel("Predict and Explain", self)
        self.title.setGeometry(150, 20, 451, 61)
        font = QFont()
        font.setPointSize(50)
        font.setBold(True)
        self.title.setFont(font)

        pixmap = QPixmap("Datasets/sampledata/" + str(id) + ".png")
        pixmap = pixmap.scaled(210, 210)
        self.image = QLabel(self)
        self.image.move(250, 85)
        self.image.setPixmap(pixmap)


        back.clicked.connect(lambda: main.transition(stack, gallery.Gallery(stack, modelData, fromClf)))
        self.go.clicked.connect(lambda: self.explain(id, x_test, y_test, iclf, clfs[self.currentClf]))
        self.show()
    
    def setCurrentClf(self):
        valObjectName = self.sender().objectName()
        valObjectId = int(valObjectName.split(" ")[1])
        self.currentClf = valObjectId
        self.go.show()

    def explain(self, id, x_test, y_test, iclf, vclf):
        explainer = lime_image.LimeImageExplainer(verbose = False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(x_test[id], 
                                classifier_fn = iclf.predict_proba, 
                                top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)
        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
        ax1.imshow(mark_boundaries(temp, mask))
        # explainer = lime_image.LimeImageExplainer(verbose = False)
        # segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(x_test[id], 
                                        classifier_fn = vclf.predict_proba, 
                                        top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)
        # temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=True, num_features=5, hide_rest=True, min_weight = 0.01)
        # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
        # ax1.imshow(mark_boundaries(temp, mask))
        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        ax2.imshow(mark_boundaries(temp, mask))
        ax1.axis('off')
        ax1.set_title("Explanation for model with initial " + str(self.modelData["var"]) + ": " + str(self.modelData["ival"]))
        ax2.axis('off')
        ax2.set_title("Explanation after varying " + str(self.modelData["var"]) + " to " + str(self.modelData["vals"][self.currentClf]))
        plt.show()
    