import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import numpy as np
from scipy.io import loadmat
from skimage.color import gray2rgb, rgb2gray, label2rgb
import matplotlib.pyplot as plt

# Create a subclass of QWidget to define your custom window


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle('Test GUI')

       # Set the background color of the window
        self.setStyleSheet("background-color: #2E8B57;")

        self.button = QPushButton('Click me!', self)
        self.button.setGeometry(100, 80, 100, 30)
        self.button.clicked.connect(self.updateButton)
        self.button = QPushButton('Click me!', self)
        self.button.setGeometry(100, 80, 100, 30)
        self.show()

    def updateButton(self):
        self.button.setText('Button Clicked!')
        self.setStyleSheet("background-color: #FF5733;")


if __name__ == '__main__':
    app = QApplication(sys.argv)  # Create an application object
    window = MyWindow()  # Create an instance of your custom window
    sys.exit(app.exec_())  # Start the application event loop
