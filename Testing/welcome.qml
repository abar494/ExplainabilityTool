import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    width: 400
    height: 300

    signal buttonClicked // Define a custom signal

    Button {
        text: "Welcome!"
        anchors.centerIn: parent
        onClicked: buttonClicked() // Emit the custom signal
    }
}
