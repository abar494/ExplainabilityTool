from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QGroupBox, QScrollArea, QVBoxLayout
from PyQt5.QtWidgets import QAbstractButton, QLabel
import main
import classifierselect
import explainer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QFont


class Gallery(QWidget):
    def __init__(self, stack, modelData, data_dict):
        #Sets labels etc
        super(Gallery, self).__init__()

        print(data_dict["x_test"].shape)

        self.modelData = modelData
        back = QPushButton("Back", self)
        back.clicked.connect(
            lambda: main.transition(
            stack, classifierselect.ClassifierSelect(stack), data_dict=None
            ))
        
        gridLayout = QGridLayout()
        groupBox = QGroupBox()
        for i in range(min(len(self.modelData["x_test"]), 120)):
            button = self.PicButton(
                QPixmap("Datasets/sampledata/" + str(i) + ".png"), i)
            button.id = i
            button.clicked.connect(
                lambda: self.explainerTransition(stack, self.modelData))
            gridLayout.addWidget(button, (i)//3, (i)%3)

        groupBox.setLayout(gridLayout)
        scroll = QScrollArea()
        scroll.setWidget(groupBox)
        scroll.setWidgetResizable(False)
        scroll.setFixedHeight(400)
        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)


        self.title = QLabel("Gallery", self)
        self.title.setGeometry(460, 90, 181, 61)
        font = QFont()
        font.setPointSize(50)
        font.setBold(True)
        self.title.setFont(font)


        self.description = QLabel("Choose an Image to test the model on!", self)
        self.description.setGeometry(410, 220, 271, 111)
        font = QFont()
        font.setPointSize(30)
        self.description.setFont(font)
        self.description.setAlignment(Qt.AlignCenter)
        self.description.setWordWrap(True)



        self.show()

    def explainerTransition(self, stack, modelData):
        id = self.sender().id
        main.transition(
            stack, explainer.Explainer(stack, id, modelData))


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