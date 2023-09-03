import os
import shutil
import sys
import classifierselect
import imageloader
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QHBoxLayout, QFormLayout, \
    QPushButton, QLabel, QSpacerItem, QSizePolicy, QMainWindow, QStackedWidget, QMessageBox


def transition(stack, context):
    stack.addWidget(context)
    next_index = stack.currentIndex()
    stack.removeWidget(stack.currentWidget())
    stack.setCurrentIndex(next_index)


class MainMenu(QMainWindow):
    def __init__(self, stack, parent=None):
        # Sets labels etc
        super(MainMenu, self).__init__()


        #Title
        self.title = QLabel("ML-XplainEd", self)
        self.title.setGeometry(225, 40, 301, 61)
        font = QFont()
        font.setPointSize(50)
        font.setBold(True)
        self.title.setFont(font)

        #Description
        self.description = QLabel("An interactive tool designed to aid you in learning " + \
            "key machine learning concepts!", self)
        self.description.setGeometry(130, 130, 461, 71)
        font = QFont()
        font.setPointSize(22)
        font.setBold(False)
        self.description.setFont(font)
        self.description.setAlignment(Qt.AlignCenter)
        self.description.setWordWrap(True)

        font = QFont()
        font.setPointSize(16)

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
            lambda: transition(stack, classifierselect.ClassifierSelect(stack)))
        self.builtIn.clicked.connect(
            lambda: transition(stack, imageloader.ImageLoader(stack)))
        self.exit.clicked.connect(
            self.close_app)
        
        self.show()

    def close_app(self):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:        QApplication.quit()
        else:                               QMessageBox.Close


    def print_size(self):
        # Get the current window size
        window_size = self.size()
        print("Window size:", window_size.width(), "x", window_size.height())

if __name__ == "__main__":
    if not os.path.isdir("Datasets/sampledata"):
        os.mkdir("Datasets/sampledata")
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    mainMenu = MainMenu(widget)
    widget.addWidget(mainMenu)
    widget.resize(733, 464)
    # widget.setFixedHeight(500)
    # widget.setFixedWidth(600)
    widget.show()
    ret = app.exec_()
    shutil.rmtree('Datasets/sampledata')
    sys.exit(ret)
