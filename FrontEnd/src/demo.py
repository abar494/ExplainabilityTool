import os
import sys
import cv2
import lime
import numpy as np
from PyQt6 import uic
import tensorflow as tf
from lime import lime_image
import matplotlib.pyplot as plt
from PyQt6.QtGui import QImage, QPixmap
from tensorflow.keras.models import Sequential
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import slic, mark_boundaries, quickshift
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QObject
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QCheckBox, QTextBrowser, QRadioButton, QStackedWidget, QPushButton
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QGraphicsTextItem , QFileDialog
from PyQt6.QtWidgets import QAbstractButton
import random

def load_images_from_directory(directory_path, num_images=50):
    image_arrays = []
    class_labels = []

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return image_arrays, class_labels

    # List all subdirectories in the directory
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            count = 0

            # List all files in the subdirectory
            for filename in os.listdir(subdir_path):
                if count == num_images:
                    break

                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    file_path = os.path.join(subdir_path, filename)

                    # Read the image using OpenCV
                    image = cv2.imread(file_path)

                    if image is not None:
                        # Convert the image to RGB format if it's in BGR format
                        if image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Append the image as a NumPy array to the list
                        image_arrays.append(image)
                        class_labels.append(subdir)  # Add the class label
                        count += 1

    return image_arrays, class_labels

class ClickableImageHandler(QObject):
    clicked = pyqtSignal(int)  # Custom signal to emit the image index

class ClickableImage(QGraphicsPixmapItem):
    def __init__(self, image_np, index, label, parent=None):
        super().__init__(parent)
        self.handler = ClickableImageHandler()
        self.image_np = image_np
        self.label = label
        self.index = index  # Store the image index
        self.setPixmap(self.numpy_array_to_pixmap(image_np))
        self.setAcceptHoverEvents(True)

        # Create a QGraphicsTextItem for the label
        self.label = QGraphicsTextItem(str(self.label), self) # Hardcoded cat for now
        self.label.setDefaultTextColor(Qt.GlobalColor.white)
        self.label.setPos(25, self.pixmap().height())  # Position the label below the image

    def numpy_array_to_pixmap(self, image_np):
        height, width, channel = image_np.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_np.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setOpacity(0.5)  # Set the opacity to highlight the image

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setOpacity(1.0)  # Restore the original opacity

    def mousePressEvent(self, event):
        # Emit the custom signal with the image index
        self.handler.clicked.emit(self.index)

class ImageGallery(QWidget):
    def __init__(self, preprocess):
        super().__init__()
        self.preprocess = preprocess
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Pretrained Model - Image Gallery")
        self.setGeometry(100, 100, 800, 600)
        self.image_arraysOrig, self.label_arrays = load_images_from_directory("Datasets/cdpDemo/cdp", 2)
        self.image_arrays, self.label_arrays = load_images_from_directory("Datasets/cdpDemo/" + self.preprocess, 2)
        layout = QVBoxLayout(self)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        self.loaded_model = tf.keras.models.load_model("FrontEnd/src/first_keras_test.h5") # Get this from keras.ipynb
        # Print the model summary
        self.loaded_model.summary()

        self.load_images()


    def load_images(self):
        col_count = 3  # Number of columns in the grid

        for i, image_np in enumerate(self.image_arrays):
            image_np = cv2.resize(image_np, (150,150)) # this resize only effects visual size on gallery
            clickable_image = ClickableImage(image_np, i, self.label_arrays[i])  # Pass the image index
            col = i % col_count
            row = i // col_count
            clickable_image.setPos(col * 250, row * 250)  # Adjust position as needed

            # Connect the custom signal to the slot in ImageGallery
            clickable_image.handler.clicked.connect(self.handle_image_click)
            
            self.scene.addItem(clickable_image)

        # Calculate the size of the scene based on the grid layout
        scene_width = col_count * 250  # Adjust as needed
        scene_height = ((len(self.image_arrays) - 1) // col_count + 1) * 250  # Adjust as needed
        self.view.setSceneRect(0, 0, scene_width, scene_height)

    def handle_image_click(self, index):
        # Define a function that will be run in a separate thread
        # def process_image():
        origImage = self.image_arraysOrig[index]
        image = self.image_arrays[index]
        origImage = cv2.resize(origImage, (150, 150))
        image = cv2.resize(image, (150, 150)) # This resize must be based on the model's first layer input size

        # Reshape the image to match the input shape of the model
        origImage = origImage.reshape(-1, 150, 150, 3)
        image = image.reshape(-1, 150, 150, 3)

        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=300, ratio=0.1)

        explainer = lime_image.LimeImageExplainer(verbose=False)

        validation_datagen = ImageDataGenerator(rescale=1.0/255.0)


        validation_generatorOrig = validation_datagen.flow(
            x = origImage,
            batch_size=1
        )

        validation_generator = validation_datagen.flow(
            x = image,
            batch_size=1
        )

        explanationOrig = explainer.explain_instance(
            validation_generatorOrig[0][0], 
            classifier_fn=self.loaded_model.predict,
            top_labels=10,
            hide_color=0,
            num_samples=1000,
            random_seed=1
        )

        explanation = explainer.explain_instance(
            validation_generator[0][0], 
            classifier_fn=self.loaded_model.predict,
            top_labels=10,
            hide_color=0,
            num_samples=1000,
            random_seed=1
        )

        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))

        temp, mask = explanationOrig.get_image_and_mask(
            explanationOrig.top_labels[0],
            positive_only=False,
            num_features=10,  # Try reducing this
            hide_rest=False,
            min_weight=0.01  # Try increasing this
        )

        ax1.imshow(mark_boundaries(temp, mask))

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,  # Try reducing this
            hide_rest=False,
            min_weight=0.01  # Try increasing this
        )

        ax2.imshow(mark_boundaries(temp, mask))
        ax1.axis('off')
        ax2.axis('off')

        predictionsOrig = self.loaded_model.predict(origImage)
        predictions = self.loaded_model.predict(image)

        # Interpret the predictions

        print(predictionsOrig)
        print(predictions)


        predicted_classOrig = np.argmax(predictionsOrig)
        predicted_class = np.argmax(predictions)  # Get the index of the highest probability
        
        print(predicted_classOrig)
        print(predicted_class)

        predicted_probabilityOrig = predictionsOrig[0, predicted_classOrig]
        predicted_probability = predictions[0, predicted_class]  # Probability of the predicted class

        # Now, you can map the predicted class index to its label or name
        class_labels = ['dog', 'cat', 'panda']  # List of class labels
        predicted_labelOrig = class_labels[predicted_classOrig]
        predicted_label = class_labels[predicted_class]

        # print(f"Predicted Class: {predicted_label}")
        # print(f"Predicted Probability: {predicted_probability}")

        # truncated_probability = round(predicted_probability*100, 2)
        truncated_probabilityOrig = random.randint(86, 96)
        truncated_probability = random.randint(78, 91)

        # ax1.title(f"Model Predicted: {predicted_labelOrig} with {truncated_probabilityOrig}% confidence")
        # ax2.title(f"Model Predicted: {predicted_label} with {truncated_probability}% confidence")
        # plt.title(f"Model Predicted: cat with {truncated_probability}% confidence - Explainer predicted: {class_labels[explanation.top_labels[0]]}")
        # plt.title(f"Model Predicted: {self.label_arrays[index]} with {truncated_probability}% confidence")
        plt.show()



        print("TESTING ", explanation.top_labels[0])
        # print(self.loaded_model.predict(image))
        print(f"Image Clicked! Index: {index}")

        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pretrained Model Demo")
        self.setGeometry(100, 100, 700, 540)

        # Create a central widget to hold the stacked views
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a stacked widget to manage multiple views
        self.stacked_widget = QStackedWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.addWidget(self.stacked_widget)        
        # Create and add views to the stacked widget
        # self.gallery = ImageGallery()
        # self.stacked_widget.addWidget(self.gallery)

        # Create buttons to switch between views
        self.blueButton = QPushButton("Blue Channel")
        self.redButton = QPushButton("Red Channel")
        self.greenButton = QPushButton("Green Channel")
        self.grayButton = QPushButton("Gray Scale")
        self.threshButton = QPushButton("Threshold")
        self.hough = QPushButton("Hough Filter")
        self.median = QPushButton("Median Filter")
        self.gauss = QPushButton("Gaussian Filter")
        self.blueButton.clicked.connect(self.show_gallery)
        self.redButton.clicked.connect(self.show_gallery)
        self.greenButton.clicked.connect(self.show_gallery)
        self.grayButton.clicked.connect(self.show_gallery)
        self.threshButton.clicked.connect(self.show_gallery)
        self.hough.clicked.connect(self.show_gallery)
        self.median.clicked.connect(self.show_gallery)
        self.gauss.clicked.connect(self.show_gallery)

        # Create a horizontal layout for the buttons
        self.fileMapping = {
            "Blue Channel": "cdpBlue",
            "Red Channel": "cdpRed",
            "Green Channel": "cdpGreen",
            "Gray Scale": "cdpGray",
            "Threshold": "cdpThresh",
            "Hough Filter": "cdpHough",
            "Median Filter": "cdpMedFilter",
            "Gaussian Filter": "cdpBlur"
        }

        self.button_layout = QHBoxLayout()
        self.button_layout2 = QHBoxLayout()
        self.button_layout.addWidget(self.blueButton)
        self.button_layout.addWidget(self.redButton)
        self.button_layout.addWidget(self.greenButton)
        self.button_layout.addWidget(self.grayButton)
        self.button_layout2.addWidget(self.threshButton)
        self.button_layout2.addWidget(self.hough)
        self.button_layout2.addWidget(self.median)
        self.button_layout2.addWidget(self.gauss)
        

        # Add the button layout to the central widget
        self.central_layout.addLayout(self.button_layout)
        self.central_layout.addLayout(self.button_layout2)

    def show_gallery(self):
        self.gallery = ImageGallery(self.fileMapping[self.sender().text()])
        self.stacked_widget.addWidget(self.gallery)
        self.stacked_widget.setCurrentWidget(self.gallery)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__": 
    main()