from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton
import main
import datahandler
from PyQt5.QtCore import Qt, QMimeData, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QDrag


class DraggableLabel(QLabel):

    id = None

    def __init__(self, name, coords, mainWindow):
        super(DraggableLabel, self).__init__(name, mainWindow)
        self.move(coords[0], coords[1])
        self.hasLanded = True
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()
        self.initialPos = self.mapFromGlobal(self.cursor().pos())
        print(self.initialPos)
        # print("Local position:", local_pos)

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

    def getID(self):
        return self.id


class DropZone(QLabel):
    def __init__(self, bing, mainWindow):
        super(DropZone, self).__init__(str(bing), mainWindow)
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

class ClassifierSelect(QWidget):

    def getButtonID(self):
        id = self.sender().getID()
        pos = self.sender().geometry()        
        print("You have clicked button " + str(id))
        print("at position (" + str(pos.x())+", "+str(pos.y())+")")

    def __init__(self, stack):
        super().__init__()
        label = QLabel("--------------------------------------------------------------------------------------------------------", self)
        label2 = QLabel("--------------------------------------------------------------------------------------------------------", self)
        label3 = QLabel("|", self)
        label.adjustSize()
        label.move(0, 50)
        label2.move(0, 200)
        label3.move(300, 50)
        dropTo = DropZone("", self)
        dropTo.setGeometry(300, 50, 300,150)
        self.setAcceptDrops(True)
        label_to_drag = DraggableLabel("drag this", (100, 475), self) 
        # label_to_drag.move(40, 40)
        hasLanded = QPushButton("Landed", self)
        hasLanded.move(0, 475)
        hasLanded.clicked.connect(lambda: print(label_to_drag.hasLanded))
        back = QPushButton("Back", self)
        back.clicked.connect(lambda: main.transition(stack, datahandler.DataSelect(stack)))
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