import os
import sys
import cv2
import lime
import numpy as np
import tensorflow as tf
from lime import lime_image
import matplotlib.pyplot as plt
from PyQt6.QtGui import QImage, QPixmap
from tensorflow.keras.models import Sequential
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QObject
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import slic, mark_boundaries, quickshift
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMainWindow, QVBoxLayout, QWidget, QGraphicsTextItem
# default shows 50 images only.
def load_images_from_directory(directory_path, num_images=50):
    image_arrays = []
    count = 0
    
    # List all files in the directory
    for filename in os.listdir(directory_path):
        if count == num_images:
            break
        
        if filename.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(directory_path, filename)
            
            # Read the image using OpenCV
            image = cv2.imread(file_path)
            
            if image is not None:
                # Convert the image to RGB format if it's in BGR format
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Append the image as a NumPy array to the list
                image_arrays.append(image)
                count += 1
    
    return image_arrays

class ClickableImageHandler(QObject):
    clicked = pyqtSignal(int)  # Custom signal to emit the image index

class ClickableImage(QGraphicsPixmapItem):
    def __init__(self, image_np, index, parent=None):
        super().__init__(parent)
        self.handler = ClickableImageHandler()
        self.image_np = image_np
        self.index = index  # Store the image index
        self.setPixmap(self.numpy_array_to_pixmap(image_np))
        self.setAcceptHoverEvents(True)

        # Create a QGraphicsTextItem for the label
        self.label = QGraphicsTextItem("cat", self)
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

class ImageGallery(QMainWindow):
    def __init__(self, image_arrays):
        super().__init__()
        self.image_arrays = image_arrays
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Pretrained Model - Image Gallery")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        self.loaded_model = tf.keras.models.load_model("/home/caleb/Desktop/p4p/ExplainabilityTool/testing/first_keras_test.h5") # Get this from keras.ipynb
        # Print the model summary
        self.loaded_model.summary()


        self.load_images()




    def load_images(self):
        row_count = 3  # Number of rows in the grid
        col_count = 3  # Number of columns in the grid

        for i, image_np in enumerate(self.image_arrays):
            image_np = cv2.resize(image_np, (150,150)) # this resize only effects visual size on gallery
            clickable_image = ClickableImage(image_np, i)  # Pass the image index
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

    # Slot to handle image click and receive the image index
    def handle_image_click(self, index):
        image = self.image_arrays[index]
        image = cv2.resize(image, (150, 150)) # This resize must be based on the model's first layer input size

        # Reshape the image to match the input shape of the model
        image = image.reshape(-1, 150, 150, 3)

        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=300, ratio=0.1)

        # Verify image[0] is whole image
        # plt.imshow(image[0])
        # plt.show()

        explainer = lime_image.LimeImageExplainer(verbose=False)


        validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

        validation_generator = validation_datagen.flow(
        x = image,
        batch_size=1
        )



        explanation = explainer.explain_instance(
            validation_generator[0][0], # it is image[0] because image shape is (-1, 150, 150 3), getting image[0] ignores batch data (or whatever)
            classifier_fn=self.loaded_model.predict,
            top_labels=10,
            hide_color=0,
            num_samples=2000
            # segmentation_fn= segmenter  
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,  # Try reducing this
            hide_rest=False,
            min_weight=0.001  # Try increasing this
        )

        plt.imshow(mark_boundaries(temp, mask))
        plt.show()

        print(self.loaded_model.predict(image))
        print(f"Image Clicked! Index: {index}")

    def predict_fn(images):
        return self.loaded_model.predict(images)
        

def main():
    # only load cat for prediction testing. don't care about other classes rn
    image_directory = "/home/caleb/Desktop/p4p/ExplainabilityTool/Datasets/animals/cat"
    image_arrays = load_images_from_directory(image_directory)

    app = QApplication(sys.argv)
    gallery = ImageGallery(image_arrays)
    gallery.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
