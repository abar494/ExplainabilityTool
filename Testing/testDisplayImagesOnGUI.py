from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QScrollArea
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
import numpy as np
import sys
# # Assuming you have a numpy array called 'image_data'
# # with shape (height, width, channels)
from scipy.io import loadmat
from skimage.color import gray2rgb, rgb2gray, label2rgb
import matplotlib.pyplot as plt
mnist = loadmat(
    r"C:\Users\space\OneDrive\Desktop\p4p\ExplainabilityTool\Datasets\MNIST\mnist-original.mat")
data = mnist["data"].T
label = mnist["label"].T
data = np.stack([gray2rgb(iimg)
                 for iimg in data.reshape((-1, 28, 28))], 0).astype(np.uint8)
label = label.astype(np.uint8)

# # Convert the numpy array to QImage
image_data, height, width, channels = np.shape(data)
print(image_data, height, width, channels)
bytes_per_line = channels * width

# Assuming you have a list of image data called 'image_data'
# image_data = [data[0], data[1], data[2], data[3]]  # Replace with your image data
image_data = data[0:5]
app = QApplication([])

# Create the main window
window = QMainWindow()
window.setWindowTitle('Image Gallery')

# Create a central widget for the window
central_widget = QWidget()
window.setCentralWidget(central_widget)

# Create a layout for the central widget
grid_layout = QGridLayout(central_widget)
grid_layout.setAlignment(Qt.AlignTop)

# Create a scroll area widget
scroll_area = QScrollArea()
scroll_area.setWidgetResizable(True)
scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
scroll_widget = QWidget()
scroll_area.setWidget(scroll_widget)

# Set the scroll area as the central widget's layout
grid_layout.addWidget(scroll_area)

# Create a layout for the scroll widget
scroll_layout = QGridLayout(scroll_widget)

# Iterate over the image data and create QLabel widgets to display the images
for row, image_data in enumerate(image_data):
    # Convert the image data to QPixmap
    q_image = QImage(data[0], width, height,
                     bytes_per_line, QImage.Format_RGB888)
# q_image = QImage(data[0], width, height,
#                  bytes_per_line, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(q_image)

    # Create a QLabel and set the pixmap as its content
    label = QLabel()
    label.setPixmap(pixmap.scaledToWidth(200))  # Adjust the width as desired

    # Add the label to the scroll layout
    scroll_layout.addWidget(label, row // 3, row % 3)

# Show the main window
window.show()

# Run the application event loop
app.exec_()
