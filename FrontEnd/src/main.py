import os
import shutil
import sys
import datahandler
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from PyQt5.uic import loadUi

def transition(stack, context):
    stack.addWidget(context)
    nextIndex = stack.currentIndex()
    stack.removeWidget(stack.currentWidget())
    stack.setCurrentIndex(nextIndex)

class MainMenu(QMainWindow):
    def __init__(self, stack, parent=None):
        #Sets labels etc
        super(MainMenu, self).__init__()
        loadUi('FrontEnd/UI/MainMenu.ui', self)

        self.defaultCLF.clicked.connect(lambda: transition(stack, datahandler.DataViewer(stack)))
        self.trainCLF.clicked.connect(lambda: transition(stack, datahandler.DataSelect(stack)))
        self.exit.clicked.connect(self.closeApp)
        self.show()

    def closeApp(self):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:        QApplication.quit()
        else:                               QMessageBox.Close

if __name__ == "__main__":
    if not os.path.isdir("Datasets/sampledata"):
        os.mkdir("Datasets/sampledata")
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    mainMenu = MainMenu(widget)
    widget.addWidget(mainMenu)
    widget.setFixedHeight(500)
    widget.setFixedWidth(600)
    widget.show()
    ret = app.exec_()
    shutil.rmtree('Datasets/sampledata')
    sys.exit(ret)