import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 800
    height: 600
    title: "Screen Manager"

    // Available screens
    property var screens: [
        { name: "Welcome", color: "lightblue" },
        { name: "Tutorial", color: "lightgreen" },
        { name: "Settings", color: "lightpink" }
    ]

    // Current active screen index
    property int currentScreenIndex: 0

    Rectangle {
        width: parent.width
        height: parent.height
        color: screens[currentScreenIndex].color
        radius: 24

        Text {
            text: screens[currentScreenIndex].name
            font.pixelSize: 24
            anchors.centerIn: parent
        }




        Button {
            text: "Button 1"
            anchors.right: parent.rights
            anchors.verticalCenter: parent.verticalCenter
        }

        Button {
            text: "Next Screen"
             anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            onClicked: {
                // Change to the next screen
                currentScreenIndex = (currentScreenIndex + 1) % screens.length
            }
        }

        }
 
}

   
