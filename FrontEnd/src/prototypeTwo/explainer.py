import main
import gallery
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, \
    QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QRadioButton
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

class Explainer(QWidget):

    def __init__(self, stack, id, modelData, fromClf):
        super(Explainer, self).__init__()

        # initialise class variables
        x_test, y_test, iclf, self.clfs, label_map = modelData["x_test"], modelData["y_test"], modelData["iclf"], modelData["clfs"], modelData["label_map"]
        self.currentClf = None
        self.modelData = modelData
        back = QPushButton("Back", self)
        
        self.go = QPushButton("Go", self)
        self.go.setGeometry(580, 400, 132, 32)

        if self.clfs is not None:
            self.go.hide()

        # Instantiate layout and add widgets
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

            #Iter 1:
            i = 0
            # Create array of identifiable and interactible radio buttons,
            # corresponding to each classifier with different hyperparameter value
            for val in modelData["vals"]:
                self.Vbox = QVBoxLayout()
                self.Vbox.setObjectName("Vbox")

                Hbox = QHBoxLayout()
                spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                Hbox.addItem(spacerItem)
                self.hpValue = QLabel(self.sliderLayout)
                self.hpValue.setObjectName("Name: " + str(i))
                self.hpValue.setText(str(val))
                Hbox.addWidget(self.hpValue)
                spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                Hbox.addItem(spacerItem)
                self.Vbox.addLayout(Hbox)

                Hbox = QHBoxLayout()
                spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                Hbox.addItem(spacerItem)
                self.valButton = QRadioButton(self.sliderLayout)
                self.valButton.setText("")
                self.valButton.setObjectName("Button: " + str(i))
                self.valButton.clicked.connect(lambda: self.setCurrentClf())

                Hbox.addWidget(self.valButton)
                spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                Hbox.addItem(spacerItem)
                self.Vbox.addLayout(Hbox)

                self.sliderBox.addLayout(self.Vbox)
                i += 1

        else:
            self.noVariableLabel = QLabel("Click the go button to generate your explanation!", self)
            self.noVariableLabel.setGeometry(220, 330,281, 41)
            self.noVariableLabel.setWordWrap(True)
            self.noVariableLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = QFont()
            font.setPointSize(20)
            font.setBold(True)
            self.noVariableLabel.setFont(font)


        self.title = QLabel("Predict and Explain", self)
        self.title.setGeometry(150, 20, 451, 61)
        font = QFont()
        font.setPointSize(50)
        font.setBold(True)
        self.title.setFont(font)
        
        # Add gallery image
        pixmap = QPixmap("Datasets/sampledata/" + str(id) + ".png")
        pixmap = pixmap.scaled(210, 210)
        self.image = QLabel(self)
        self.image.move(250, 85)
        self.image.setPixmap(pixmap)

        # Add back button
        back.clicked.connect(lambda: main.transition(stack, gallery.Gallery(stack, modelData, fromClf)))
        
        # Trigger model classification and LIME explanation
        self.go.clicked.connect(lambda: self.varChecker(id, x_test, y_test, iclf, label_map))
       

        self.show()
    
    # Sets the current classifier to be explained (based on radio button selection corresponding to hyperparameter value)
    def setCurrentClf(self):
        valObjectName = self.sender().objectName()
        valObjectId = int(valObjectName.split(" ")[1])
        self.currentClf = valObjectId
        self.go.show()


    def varChecker(self, id, x_test, y_test, iclf, label_map):
        if self.currentClf is None:
            self.explain(id, x_test, y_test, iclf, None, label_map)
        else:
            self.explain(id, x_test, y_test, iclf, self.clfs[self.currentClf], label_map)


    # Method which triggers the LIME explanation
    def explain(self, id, x_test, y_test, iclf, vclf, label_map):
        # Apply LIME integration to the classifier

        print(y_test[id])

        explainer = lime_image.LimeImageExplainer(verbose = False)

        # LIME notably takes in its own hyperparameters which are set here
        # This can pose a limitation to the user, as they may not be able to
        # change these hyperparameters, however this current set of parameters seeks to
        # demonstrate LIME's capabilities in an educational context
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(x_test[id], 
                                classifier_fn = iclf.predict_proba, 
                                top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)
        
        # Create two plots, one for the initial classifier, and one for the classifier with the hyperparameter value changed
        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        
        if vclf is None:
            plt.imshow(mark_boundaries(temp, mask))
            plt.axis('off')
            plt.title("Model explanation,\nPredicted " + str(label_map[iclf.predict(x_test[id].reshape(1, -1))[0]]) + ": " +
                       ("Correct" if iclf.predict(x_test[id].reshape(1, -1))[0] == y_test[id] else "Incorrect"))
            plt.show()
        else:
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
            regPrediction = iclf.predict(x_test[id].reshape(1, -1))[0]
            regCorrect = "Correct" if regPrediction == y_test[id] else "Incorrect"
            ax1.set_title("Model explanation, " + str(self.modelData["var"]) + ": " + str(self.modelData["ival"]) + "\nPredicted " + str(label_map[regPrediction]) + ": " + regCorrect)
            ax2.axis('off')
            varyPrediction = vclf.predict(x_test[id].reshape(1, -1))[0]
            varyCorrect = "Correct" if varyPrediction == y_test[id] else "Incorrect"
            ax2.set_title("Explanation after varying " + str(self.modelData["var"]) + " to " + str(self.modelData["vals"][self.currentClf]) + "\nPredicted " + str(label_map[varyPrediction]) + ": " + varyCorrect)
            plt.show()