import sys

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QCursor
from PyQt5.QtCore import QMimeData, Qt

class DraggableLabel(QLabel):
    def __init__(self, name, coords, mainWindow):
        super(DraggableLabel, self).__init__(name, mainWindow)
        self.move(coords[0], coords[1])
        self.monka = True
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()
        self.initialPos = self.mapFromGlobal(self.cursor().pos())
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

class DropLabel(QLabel):
    def __init__(self, bing, mainWindow):
        super(DropLabel, self).__init__(str(bing), mainWindow)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        pos = event.pos()
        widget = event.source()
        widget.move(pos - widget.initialPos)
        # newNum = int(self.text())+1
        # # text = event.mimeData().text()
        # self.setText(str(newNum))
        print("landed")
        event.acceptProposedAction()
        widget.monka = True

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        label = QLabel("--------------------------------------------------------------------------------------------------------", self)
        label.adjustSize()
        label.move(0, 150)
        label = DropLabel("", self)
        label.setGeometry(0, 0, 600,150)
        label2 = DropLabel("", self)
        label2.move(300, 0)
        self.setAcceptDrops(True)
        label_to_drag = DraggableLabel("drag this", (100, 475), self) 
        # label_to_drag.move(40, 40)
        monka = QPushButton("monka", self)
        monka.move(0, 475)
        monka.clicked.connect(lambda: print(label_to_drag.monka))
        self.show()
    
    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        pos = e.pos()
        widget = e.source()
        widget.move(pos - widget.initialPos)
        widget.monka = False
        e.acceptProposedAction()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.setFixedHeight(500)
    w.setFixedWidth(600)
    w.show()
    sys.exit(app.exec_())