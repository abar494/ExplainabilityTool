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

        cw = QWidget(self)
        gl = QGridLayout(cw)
        hl = QHBoxLayout()
        si = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hl.addItem(si)
        fl = QFormLayout()
        defaultCLF = QPushButton("Use a default classifier", cw)
        fl.setWidget(2, QFormLayout.FieldRole, defaultCLF)
        # loadCLF = QPushButton("Load a classifier", cw)
        # fl.setWidget(3, QFormLayout.FieldRole, loadCLF)
        trainCLF = QPushButton("Train a new classifier", cw)
        fl.setWidget(3, QFormLayout.FieldRole, trainCLF)
        exit = QPushButton("Exit", cw)
        fl.setWidget(5, QFormLayout.FieldRole, exit)
        title = QLabel("P4P XAI Tool", cw)
        font = QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(40)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        fl.setWidget(0, QFormLayout.FieldRole, title)
        si = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)
        fl.setItem(1, QFormLayout.FieldRole, si)
        hl.addLayout(fl)
        si = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hl.addItem(si)
        gl.addLayout(hl, 2, 0, 1, 1)
        si = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        gl.addItem(si, 5, 0, 1, 1)
        si = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)
        gl.addItem(si, 1, 0, 1, 1)
        self.setCentralWidget(cw)


        self.sizebutton = QPushButton("Print Window Size", self)
        self.sizebutton.clicked.connect(self.print_size)

        trainCLF.clicked.connect(
            lambda: transition(stack, classifierselect.ClassifierSelect(stack)))
        defaultCLF.clicked.connect(
            lambda: transition(stack, imageloader.ImageLoader(stack)))
        exit.clicked.connect(
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
