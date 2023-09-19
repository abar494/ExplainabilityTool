import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QStackedWidget, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5 import uic
import imageloader
import main
import classifierselect as clfSel


class Noise(QWidget):
    def __init__(self, stack, model_data=None):
        super().__init__()
        uic.loadUi("FrontEnd/UI/AddNoise.ui", self)  # Load your UI file
        self.stack = stack
        self.data_dict = model_data
        self.blur_level = 0
        self.initial_state()
        self.connect_all()

        # Example image
        # self.image = cv2.imread("/home/caleb/Desktop/p4p/ExplainabilityTool/Datasets/animals/cats_00197.jpg")
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = self.data_dict["x_train"][0]
        self.image_copy = self.image.copy()
        # Get the QGraphicsView widget from the UI
        self.graphicsView = self.findChild(QGraphicsView, "graphicsView")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)  # Set the scene for the QGraphicsView

        self.display_image(self.image)

    # Displays image to the graphics view.
    def display_image(self, image):
        if image is not None:
            view_width = self.graphicsView.width()
            view_height = self.graphicsView.height()

            # Calculate the aspect ratio of the image
            aspect_ratio = image.shape[1] / image.shape[0]

            # Calculate the maximum width and height that fit within the view
            max_width = view_width
            max_height = view_height

            if view_width / aspect_ratio > view_height:
                max_width = view_height * aspect_ratio
            else:
                max_height = view_width / aspect_ratio

            # Resize the image to fit within the calculated dimensions
            resized_image = cv2.resize(image, (int(max_width), int(max_height)))

            # Convert the resized image to QImage and display it in a QGraphicsPixmapItem
            q_image = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0],
                             resized_image.shape[1] * 3, QImage.Format_RGB888)  # Use RGB888 format
            pixmap = QPixmap.fromImage(q_image)

            # Clear the scene and add the pixmap item
            self.scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)

    def initial_state(self):
        self.continue1.setEnabled(False)

    def connect_all(self):
        self.continue1.clicked.connect(self.apply_blur_to_selection)
        self.noiseSlider.valueChanged.connect(self.apply_blur_to_example)
        self.confirm.clicked.connect(self.enable_continue)
        self.back.clicked.connect(lambda: main.transition(self.stack, imageloader.ImageLoader(self.stack)))
    
    # Enable or disable the continue button based on confirm selection checkbox.    
    def enable_continue(self): 
        if self.confirm.isChecked():
            self.continue1.setEnabled(True)
        else:
            self.continue1.setEnabled(False)

    # Continue button pressed - apply noise.
    def apply_blur_to_selection(self):

        if self.confirm.isChecked() and self.blur_level > 0:
            # Don't apply any noise, send data through to classifer select
            if self.neither.isChecked():
                print("Don't apply blur to any set")
                # Send data to classifer select

            elif self.both.isChecked():
                print("Apply blur to both training and testing sets")
                self.apply_blur_to_dataset("x_train")
                self.apply_blur_to_dataset("x_test")
                # Send data to classifer select
            elif self.train.isChecked():
                print("Apply blur to training set only")
                self.apply_blur_to_dataset("x_train")
                # Send data to classifer select

            elif self.test.isChecked():
                print("Apply blur to testing set only")
                self.apply_blur_to_dataset("x_test")
                # Send data to classifer select
           
        print("and continue")
        print("Blur level:" + str(self.blur_level/10))
        main.transition(self.stack, clfSel.ClassifierSelect(self.stack, self.data_dict))
    
    
    # Based on the key input string, apply noise to dataset. 
    def apply_blur_to_dataset(self, key):
        if self.data_dict[key] is not None:
            # Apply noise to all images in the dataset
            for index, image in enumerate(self.data_dict[key]): 
                self.data_dict[key][index] = cv2.GaussianBlur(image, (0, 0), self.blur_level/10)
    
    # Display 
    def apply_blur_to_example(self):
        self.blur_level = self.noiseSlider.value()
        print(self.blur_level)
        
        if self.image is not None and self.blur_level > 0:
            blurred_image = cv2.GaussianBlur(self.image, (0, 0), self.blur_level/10)
            self.display_image(blurred_image)


    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    check = Noise(widget)
    check.show()
    sys.exit(app.exec())
    