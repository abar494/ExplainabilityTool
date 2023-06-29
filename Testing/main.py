import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQuick import QQuickView

if __name__ == "__main__":
    app = QApplication(sys.argv)

    view = QQuickView()
    view.setSource(QUrl('welcome.qml'))
    view.setResizeMode(QQuickView.SizeRootObjectToView)

    new_view = QQuickView()
    new_view.setSource(QUrl('nextScreen.qml'))
    new_view.setResizeMode(QQuickView.SizeRootObjectToView)

    def switch_to_new_window():
        view.close()
        new_view.show()

    view.show()

    # Access the QML root object to establish the signal-slot connection
    root = view.rootObject()
    root.buttonClicked.connect(switch_to_new_window)

    sys.exit(app.exec_())
