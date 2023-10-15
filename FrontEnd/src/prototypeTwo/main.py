import os
import shutil
import sys
import createloader
import builtloader
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QPushButton, QLabel, QMainWindow, QStackedWidget, QMessageBox


# Method which allows for transition between different views of Prototype 2
def transition(stack, context):
    stack.addWidget(context)
    next_index = stack.currentIndex()
    stack.removeWidget(stack.currentWidget())
    stack.setCurrentIndex(next_index)


# Main Menu for Prototype 2, allows basic selection between built-in models and custom models
class MainMenu(QMainWindow):
    def __init__(self, stack, parent=None):
        super().__init__(parent)
        
        #Title
        self.title = QLabel("ML-XplainEd", self)
        self.title.setGeometry(270, 40, 301, 61)
        font = QFont()
        font.setPointSize(30)
        font.setBold(True)
        self.title.setFont(font)

        #Description
        self.description = QLabel("An interactive tool designed to aid you in learning " + \
            "key machine learning concepts!", self)
        self.description.setGeometry(130, 130, 461, 71)
        font = QFont()
        font.setPointSize(18)
        font.setBold(False)
        self.description.setFont(font)
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setWordWrap(True)

        font = QFont()
        font.setPointSize(12)

        self.builtIn = QPushButton("Built-in Models", self)
        self.builtIn.setGeometry(295, 230, 131, 32)
        self.builtIn.setFont(font)

        self.build = QPushButton("Create a Model", self)
        self.build.setGeometry(295, 280, 131, 32)
        self.build.setFont(font)

        self.exit = QPushButton("Exit", self)
        self.exit.setGeometry(295, 330, 131, 32)
        self.exit.setFont(font)

        self.build.clicked.connect(
            lambda: transition(stack, createloader.ImageLoader(stack)))
        self.builtIn.clicked.connect(
            lambda: transition(stack, builtloader.ImageLoader(stack)))
        self.exit.clicked.connect(self.close_app)
        
        self.show()

    # Method which allows for the closing of the application
    def close_app(self):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            QApplication.quit()
        else:
            QMessageBox.Close


# Main method which runs the application
if __name__ == "__main__":

    # Create a sampledata folder to store later test images
    if not os.path.isdir("Datasets/sampledata"):
        os.mkdir("Datasets/sampledata")

    # Create the application
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    mainMenu = MainMenu(widget)
    widget.addWidget(mainMenu)
    widget.resize(733, 464)

    # Read the style.qss file
    with open('FrontEnd/UI/style.qss', 'r') as f:
        style_sheet = f.read()

    widget.setStyleSheet(style_sheet)
    widget.show()
    ret = app.exec()
    shutil.rmtree('Datasets/sampledata')
    sys.exit(ret)
