from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QGroupBox, QScrollArea, QVBoxLayout
from PyQt5.QtWidgets import QAbstractButton
import main
import datahandler
import explainer
from PyQt5.QtGui import QPixmap, QPainter


class Gallery(QWidget):
    def __init__(self, stack, x_test, y_test, iclf, clfs, values):
        #Sets labels etc
        super(Gallery, self).__init__()

        back = QPushButton("Back", self)
        back.clicked.connect(
            lambda: main.transition(
            stack, datahandler.DataViewer(stack, x_test, y_test, iclf, clfs, values)
            ))
        
        gridLayout = QGridLayout()
        groupBox = QGroupBox()
        for i in range(min(len(x_test), 120)):
            button = self.PicButton(
                QPixmap("Datasets/sampledata/" + str(i) + ".png"), i)
            button.id = i
            button.clicked.connect(
                lambda: self.explainerTransition(stack, x_test, y_test, iclf, clfs, values))
            gridLayout.addWidget(button, (i)//3, (i)%3)

        groupBox.setLayout(gridLayout)
        scroll = QScrollArea()
        scroll.setWidget(groupBox)
        scroll.setWidgetResizable(False)
        scroll.setFixedHeight(400)
        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.show()

    def explainerTransition(self, stack, x_test, y_test, iclf, clfs, values):
        id = self.sender().id
        main.transition(
            stack, explainer.Explainer(stack, id, x_test, y_test, iclf, clfs, values))


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