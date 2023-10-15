from PyQt6.QtWidgets import QWidget, QPushButton, QGridLayout, QGroupBox, QScrollArea, QVBoxLayout
from PyQt6.QtWidgets import QAbstractButton, QLabel
import main
import classifierselect
import builtloader
import explainer
import matplotlib.pyplot as plt
import tensorflow as tf
from lime.wrappers.scikit_image import SegmentationAlgorithm
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPainter, QFont
from lime import lime_image
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from skimage.segmentation import slic, mark_boundaries, quickshift
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image, ImageOps  # Install pillow instead of PIL
# from keras.models import load_model
from skimage.segmentation import mark_boundaries

class Gallery(QWidget):

    def backTransition(self):
        if self.fromClf:
            modelData = {
                "x_train": self.modelData["x_train"],
                "y_train": self.modelData["y_train"],
                "x_test": self.modelData["x_test"],
                "y_test": self.modelData["y_test"],
                "label_map": self.modelData["label_map"] 
            }

            main.transition(
                self.stack, classifierselect.ClassifierSelect(self.stack, modelData))
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
            pixmap = QPixmap("Datasets/sampledata/" + str(i) + ".png")
            pixmap = pixmap.scaled(100, 100)
            button = self.PicButton(
                pixmap, i)
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
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setWordWrap(True)

        self.show()

    def explain(self, stack, modelData):
        id = self.sender().id
        if self.fromClf:
            main.transition(
                stack, explainer.Explainer(stack, id, modelData, self.fromClf))
        else:

            self.updatedModel = tf.keras.models.load_model(self.modelData["iclf"])

            class_names = open("KerasModels/labels.txt", "r").readlines()

            # Create the array of the right shape to feed into the keras model
            # The 'length' or number of images you can put into the array is
            # determined by the first position in the shape tuple, in this case 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Replace this with the path to your image
            image = Image.open("Datasets/sampledata/"+str(id)+".png").convert("RGB")

            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = self.updatedModel.predict(data)
            # prediction = self.loaded_model.predict(data)

            class_name = class_names[np.argmax(prediction)].split(" ")[1]

            confidence_score = prediction[0][np.argmax(prediction)]

            # Print prediction and confidence score

            print("Class:", class_name[2:], end="")
            print("Confidence Score:", confidence_score)

            ximage = self.modelData["x_test"][id]
            ximage = cv2.resize(ximage, (224, 224)) # This resize must be based on the model's first layer input size

            # Reshape the image to match the input shape of the model
            ximage = ximage.reshape(-1, 224, 224, 3)

            # segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=300, ratio=0.1)

            limeExplain = lime_image.LimeImageExplainer(verbose=False)

            validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

            validation_generator = validation_datagen.flow(
                x = ximage,
                batch_size=1
            )

            explanation = limeExplain.explain_instance(
                validation_generator[0][0], 
                classifier_fn=self.updatedModel.predict,
                top_labels=10,
                hide_color=0,
                num_samples=1000,
                random_seed=1
            )

            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                num_features=10,  # Try reducing this
                hide_rest=False,
                min_weight=0.01  # Try increasing this
            )

            plt.imshow(mark_boundaries(temp, mask))
            plt.axis('off')

            truncated_confidence = round(confidence_score*100, 2)


            plt.title(f"Predicted: {class_name}, {truncated_confidence}% confidence")
            # plt.set_title(f"Predicted: {class_name}, {truncated_confidence}% confidence")

            plt.show()

            print("TESTING ", explanation.top_labels[0])
            # print(self.loaded_model.predict(image))
            print(f"Image Clicked! Index: {id}")




            # print("ID is " + id)
            # print(modelData["iclf"])
            # print(modelData["y_test"][id])
            # plt.imshow(modelData["x_test"][id])
            # plt.show()

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