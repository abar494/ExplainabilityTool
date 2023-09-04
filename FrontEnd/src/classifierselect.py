from PyQt5.QtWidgets import QWidget, QComboBox, QLineEdit, QApplication, QLabel, \
      QPushButton, QSpacerItem, QHBoxLayout, QSizePolicy, QGroupBox, QGridLayout
import main
import gallery
import dataprocessor
from PyQt5.QtCore import Qt, QMimeData, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QDrag
import json
import imageloader_v2 as img_loader


class ClassifierSelect(QWidget):

    def __init__(self, stack):
        super().__init__()
        f = open('FrontEnd/src/ClassifierPrefs.json')
        self.params = {}
        self.stack = stack
        self.hps = {}
        self.data = None

        data = json.load(f)
        self.layoutWidget= QWidget(self)
        self.layoutWidget.setGeometry(20, 10, 551, 41)
        self.layout = QHBoxLayout(self.layoutWidget)

        self.layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        for clf in data.keys():
            button = QPushButton(clf, self.layoutWidget)
            button.clicked.connect(
                lambda: self.clfSelect(data)
            )
            self.layout.addWidget(button)
        self.layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        back = QPushButton("Back", self)
        back.clicked.connect(lambda: main.transition(stack, main.img_loader(stack)))
        self.show()

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        pos = e.pos()
        print(pos)
        widget = e.source()
        widget.move(pos - widget.initialPos)
        widget.hasLanded = False
        e.acceptProposedAction()

    def clfSelect(self, data):
        self.clf = self.sender().text()
        self.hps = data[self.clf]
        self.layoutWidget.hide()

        self.groupBox = QGroupBox(self)
        self.groupBox.setGeometry(80, 70, 401, 221)
        self.groupBox.setObjectName("groupBox")

        self.grid = QGridLayout(self.groupBox)
        self.grid.setObjectName("grid")

        i = 0
        varhpLabel = QLabel("Select Variable Hyperparameter", self)
        varhpLabel.move(90, 340)
        varhpLabel.show()
        self.varComboBox = QComboBox(self)
        self.varComboBox.move(90, 370)
        for hp in self.hps.keys():
            self.varComboBox.addItem(hp)
            label = QLabel(hp, self.groupBox)
            self.grid.addWidget(label, 0, i)
            comboBox = QComboBox(self.groupBox)
            comboBox.addItem("Default")
            comboBox.setObjectName(hp)
            for val in self.hps[hp]:
                comboBox.addItem(val)
                comboBox.last = "default"
                comboBox.currentTextChanged.connect(self.comboChanged)
            self.grid.addWidget(comboBox, 1, i)
            i+=1
        self.varComboBox.show()
        self.groupBox.show()
        self.selectedButton = QPushButton("Accept", self)
        self.selectedButton.move(490, 280)
        self.selectedButton.clicked.connect(lambda: self.modelSelected())
        self.selectedButton.show()

    
    def comboChanged(self):
        box = self.sender()
        hpName = box.objectName()
        hpVal = box.currentText()
        self.params[hpName] = hpVal
        child = self.findChild(QLineEdit, hpName)
        if hpVal == "Custom" and (child is None):
            p = box.pos() + QPoint(80, 120)
            line = QLineEdit(self)
            line.setPlaceholderText("Enter Value")
            line.setObjectName(hpName)
            line.move(p)
            line.show()
        elif hpVal == "Custom":
            child.show()
        else:
            if box.last == "Custom":
                child.setText("")
                child.hide()
            if hpVal == "Default":
                self.params.pop(hpName)
        box.last = hpVal

    def modelSelected(self):
        for hpName in self.params.keys():
            if str(self.params[hpName]).isdigit():
                self.params[hpName] = int(self.params[hpName])
            if self.params[hpName] == "Custom":
                child = self.findChild(QLineEdit, hpName)
                if child.text() == "":
                    self.params.pop(hpName)
                else:
                    self.params[hpName] = float(child.text())
        vals = self.hps[self.varComboBox.currentText()]
        if "Custom" in vals:
            vals.remove("Custom")
            vals = [int(val) for val in vals]
        prefs = {
            "hps": self.params,
            "clf": self.clf,
            "var": self.varComboBox.currentText(),
            "vals": vals
        }

        dp = dataprocessor.DataProcessor(prefs)
        modelData = {
            "x_test": dp.x_test,
            "y_test": dp.y_test,
            "iclf": dp.iclf,
            "clfs": dp.clfs,
            "vals": dp.vals
        }
        main.transition(self.stack, gallery.Gallery(self.stack, modelData, self.data))

        # main.transition(self.stack, dataprocessor.DataProcessor(self.stack, prefs))
        





    class DraggableLabel(QLabel):
        def __init__(self, name, coords, mainWindow):
            super(ClassifierSelect.DraggableLabel, self).__init__(name, mainWindow)
            self.move(coords[0], coords[1])
            self.hasLanded = True

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.drag_start_position = event.pos()
            self.initialPos = self.mapFromGlobal(self.cursor().pos())
            print(self.initialPos)

        def mouseMoveEvent(self, event):
            if not (event.buttons() & Qt.LeftButton):
                return
            if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
                return
            drag = QDrag(self)
            mimedata = QMimeData()
            mimedata.setText(self.text())
            drag.setMimeData(mimedata)
            pixmap = QPixmap(self.size())
            painter = QPainter(pixmap)
            painter.drawPixmap(self.rect(), self.grab())
            painter.end()
            drag.setPixmap(pixmap)
            drag.setHotSpot(event.pos())
            drag.exec_(Qt.CopyAction | Qt.MoveAction)

    
    class DropZone(QLabel):
        def __init__(self, bing, mainWindow):
            super(ClassifierSelect.DropZone, self).__init__(str(bing), mainWindow)
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event):
            if event.mimeData().hasText():
                event.acceptProposedAction()

        def dropEvent(self, event):
            pos = event.pos()
            geom = self.geometry()
            point = QPoint(geom.x(), geom.y())
            print(pos)
            widget = event.source()
            widget.move(pos - widget.initialPos + point)
            print("landed")
            event.acceptProposedAction()
            widget.hasLanded = True
