import sys 
from PyQt5.QtCore import Qt, QMimeData 
from PyQt5.QtGui import QDrag 
from PyQt5.QtWidgets import QApplication, QLabel, QWidget 
 
class DragDropWidget(QWidget): 
    def __init__(self): 
        super().__init__() 
 
        # Create a label to display the file path 
        self.label = QLabel(self) 
        self.label.setText("Drop a file here") 
        self.label.setAlignment(Qt.AlignCenter) 
 
        # Set the widget to accept drops 
        self.setAcceptDrops(True) 
 
    def dragEnterEvent(self, event): 
        # Check if the dragged data is a file 
        if event.mimeData().hasUrls(): 
            # Accept the drop if the file is a valid type 
            event.accept() 
        else: 
            # Reject the drop if the file is not a valid type 
            event.ignore() 
 
    def dropEvent(self, event): 
        # Get the file path from the dropped data 
        file_path = event.mimeData().urls()[0].toLocalFile() 
 
        # Update the label to display the file path 
        self.label.setText(file_path) 
        self.label.adjustSize()
 
if __name__ == '__main__': 
    app = QApplication(sys.argv) 
    widget = DragDropWidget() 
    widget.show() 
    sys.exit(app.exec_()) 