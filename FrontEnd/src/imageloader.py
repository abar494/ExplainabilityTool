import os
import sys
import matplotlib.pyplot as plt  # Plotting Images
from PIL import Image
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
import main
import classifierselect
from PyQt5 import uic
from dataset import Dataset, save_dataset_to_file

# Temp dark stylesheet
dark_stylesheet = """
    /* Set the background color of the application */
    QApplication { background-color: #333333; }

    /* Set the text color for all widgets */
    QWidget { color: #FFFFFF; background-color: #333333 }

    /* Set the background and text color for buttons */
    QPushButton {
        background-color: #555555;
        color: #FFFFFF;
        border: none;
        padding: 5px;
        border-radius: 2.5px;
    }

    /* Set the background color of buttons when hovered */
    QPushButton:hover {
        background-color: #888888;
    }

    /* Set the background color of buttons when pressed */
    QPushButton:pressed {
        background-color: #333333;
    }

    QPushButton:disabled {
        background-color: #444444;
        color: #888888;
    }

    /* Set the background color of disabled spin boxes */
    QSpinBox:disabled {
        background-color: #444444;
        color: #888888;
    }

    QSpinBox:disabled::up-button {
        border: 1px solid #999999; /* Border color for up arrow when disabled */
    }

    QSpinBox:disabled::down-button {
        border: 1px solid #999999; /* Border color for down arrow when disabled */
    }

    /* Set the color of disabled QLabel text */
    QLabel:disabled {
        color: #888888;
    }
    QSlider {
        background-color: #555555;
        height: 8px;
    }

    QSlider::groove:horizontal {
        background-color: #888888;
        height: 8px;
    }

    QSlider::handle:horizontal {
        background-color: #FFFFFF;
        width: 12px;
        margin: -2px 0;
        border-radius: 6px;
    }

    QSlider::sub-page:horizontal {
        background-color: #FFFFFF;
        height: 8px;
    }

    QSlider::add-page:horizontal {
        background-color: #444444;
        height: 8px;
    }

    QSlider:disabled {
        background-color: #444444;
    }

    QSlider::groove:disabled {
        background-color: #555555;
    }

    QSlider::handle:disabled {
        background-color: #888888;
    }

    QSlider::sub-page:disabled {
        background-color: #888888;
    }

    QSlider::add-page:disabled {
        background-color: #444444;
    }

    /* Set the background and text color for line edit */
    QLineEdit {
        background-color: #555555;
        color: #FFFFFF;
        border: 1px solid #888888;
        padding: 5px;
    }
    
    /* Set the background color of line edit when focused */
    QLineEdit:focus {
        background-color: #777777;
        border: 1px solid #FFFFFF;
    }

    QCheckBox::disabled {
        color: #888888
    }


    QCheckBox::indicator {
        width: 10px;
        height: 10px;
        border: 2px solid #888888;
        background-color: #222222;
    }

    QCheckBox::indicator:unchecked {
        border: 2px solid #888888;
        background-color: #222222;
    }

    QCheckBox::indicator:checked {
        background-color: #888888;
    }

    QCheckBox::indicator:hover {
        border: 2px solid #aaaaaa;
    }

    QCheckBox::indicator:checked:hover {
        border: 2px solid #888888;
    }

    QCheckBox::indicator:unchecked:hover {
        border: 2px solid #888888;
    }

"""

        # border: 2px solid #00ff00;


class ImageLoader(QWidget):
    def __init__(self, stack=None):
        super().__init__()
        uic.loadUi("FrontEnd/UI/ImageLoader.ui", self)
        
        self.stack = stack
        
        self.init_button_connects()
        self.init_initial_values()
        self.init_sliders()
        self.init_buttons_and_layouts()
        # Initially most buttons are disabled.

        # Set initial tool tips
        self.dataDir.setToolTip("Folder Directory: None")
        self.dataFile.setToolTip("File Directory: None")

    # Connect buttons to their functions
    def init_button_connects(self):
        self.selectData.clicked.connect(self.select_file)
        self.selectDir.clicked.connect(self.select_folder)
        self.resetData.clicked.connect(self.reset_selection)
        self.resetSpins.clicked.connect(self.reset_parameters)
        # Temporary function of the BACK button prints all the class attributes.
        self.back.clicked.connect(self.print_attributes) 
        self.maxImages.valueChanged.connect(self.update_spins)
        self.resizeX.valueChanged.connect(self.update_spins)
        self.resizeY.valueChanged.connect(self.update_spins)
        # Connect Train/Test Split sliders and spinboxes to their function
        self.trainSlider.valueChanged.connect(self.update_sliders)
        self.trainSpin.valueChanged.connect(self.update_sliders)
        self.confirmSelection.stateChanged.connect(self.confirm_selection)
        self.saveFile.clicked.connect(self.save_file)

        #     lambda: self.save_file(self.fileName))

    # Set attributes to their defaults
    def init_initial_values(self):
        self.max_images_value = 100
        self.resize_x_value = 500
        self.resize_y_value = 500
        self.image_count = 0
        self.largest_size = (0, 0)
        self.smallest_size = (0, 0)
        self.subdirectories = None
        self.folder_directory = None
        self.file_directory = None
        self.folder_name = None
        self.file_name = None
        self.default_max_images = 0
        self.default_max_resize = 0
        self.num_classes = 0
        self.save_file_name = None

    # Set sliders to their defaults
    def init_sliders(self):
        self.default_train_split = self.train_test_split = 70 # This value sets the default on the GUI
        self.trainSlider.setValue(self.default_train_split)
        self.testSlider.setValue(100 - self.default_train_split)
        # Disable Test text, spinbox, and slider. Should always be disabled.
        self.testSlider.setEnabled(False)
        self.testSpin.setEnabled(False)
        self.testText.setEnabled(False)

    # Initialise all buttons and layouts
    def init_buttons_and_layouts(self):
        self.selectDir.setEnabled(True)
        self.selectData.setEnabled(True)
        self.resetData.setEnabled(False)
        self.resetSpins.setEnabled(False)
        self.resetData.setEnabled(False)
        self.confirmSelection.setEnabled(False)
        self.dataFile.setText("Selected File: None")
        self.dataDir.setText("Selected Folder: None")
        self.enable_parameter_layout(False)
        self.maxImages.setMaximum(100000)
        self.saveFile.setEnabled(False)
        self.fileName.setEnabled(False)
        # Initially disable information layout
        self.enable_layout(False, self.findChild(QVBoxLayout, "infoLayout"))
        # self.enable_layout(False, self.findChild(QVBoxLayout, "saveToFile"))
        self.confirmSelection.setEnabled(False)

    # Upate Train and Test split sliders.
    def update_sliders(self):
        # Check sender
        if self.sender() == self.trainSlider:
            train_test_split = self.trainSlider.value()
            complement = 100 - train_test_split
            self.trainSpin.setValue(train_test_split)
        else:
            train_test_split = self.trainSpin.value()
            complement = 100 - train_test_split
            self.trainSlider.setValue(train_test_split)

        self.testSlider.setValue(complement)
        self.testSpin.setValue(complement)
        self.train_test_split = train_test_split
        self.check_enable_params()
        # print(self.train_test_split)

    # Update attributes with current spinbox values, and enable clearing of spinboxes.
    def update_spins(self):
        self.max_images_value = self.maxImages.value()
        self.resize_x_value = self.resizeX.value()
        self.resize_y_value = self.resizeY.value()
        print(self.max_images_value, self.resize_x_value, self.resize_y_value)
        # If any of the spinboxes have a none default value, enable the clear button
        self.check_enable_params()

    # If params aren't default, enable param reset.
    def check_enable_params(self):
        if (
            self.max_images_value != self.default_max_images
            or self.resize_x_value != self.default_max_resize
            or self.resize_y_value != self.default_max_resize
            or self.train_test_split != self.default_train_split
        ):
            self.resetSpins.setEnabled(True)
        else:
            self.resetSpins.setEnabled(False)

    # Set spinbox values to default and disable clear button.

    # Reset all params.
    def reset_parameters(self):
        # Setting these values to default also automatically updates the corresponding class attributes (via ValueChanged()).
        self.maxImages.setValue(self.default_max_images)
        self.resizeX.setValue(self.default_max_resize)
        self.resizeY.setValue(self.default_max_resize)
        self.trainSpin.setValue(self.default_train_split)
        self.resetSpins.setEnabled(False)  # Disable Reset after pressing.

    # Clear selected files, re-enable and re-disable appropriate buttons and text
    def reset_selection(self):
        self.init_initial_values()
        self.init_sliders()
        self.init_buttons_and_layouts()
        self.reset_parameters()
        self.update_info()
        self.confirmSelection.setChecked(False)

    # Select the directory containing the dataset. Enable appropriate Pushbuttons and SpinBoxes
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_name = os.path.basename(folder_path)
            self.folder_directory = folder_path
            self.dataDir.setText("Selected Folder: " + self.folder_name)
            self.dataDir.setToolTip("Folder Directory: " + str(self.folder_directory))

            self.selectData.setEnabled(False)
            self.dataFile.setText("")
            self.confirmSelection.setEnabled(True)
            self.resetData.setEnabled(True)
            self.enable_parameter_layout(True)
            self.get_folder_info()
            self.update_info()

    # Get info about chosen folder.
    def get_folder_info(self):
        folder_name = self.folder_name
        folder_directory = self.folder_directory
        subdirectories = 0
        image_count = 0
        largest_size = (0, 0)
        # Initial smallest size should be very high
        smallest_size = (float("inf"), float("inf"))
        try:
            # Check if the folder exists
            if os.path.exists(folder_directory):
                # Get the list of subdirectories and files within the folder
                entries = os.scandir(folder_directory)

                for entry in entries:
                    if entry.is_dir():
                        subdirectories += 1
                        # Count the number of images within each subdirectory
                        subdirectory_path = os.path.join(folder_directory, entry.name)
                        images = [
                            name
                            for name in os.listdir(subdirectory_path)
                            if os.path.isfile(os.path.join(subdirectory_path, name))
                        ]

                        for image_name in images:
                            image_path = os.path.join(subdirectory_path, image_name)
                            image = Image.open(image_path)
                            width, height = image.size

                            if width * height > largest_size[0] * largest_size[1]:
                                largest_size = (width, height)

                            if width * height < smallest_size[0] * smallest_size[1]:
                                smallest_size = (width, height)

                            image_count += 1
                            image.close()

                print("Folder Information:")
                print(f"Name: {folder_name}")
                print(f"Directory: {folder_directory}")
                print(f"Number of Subdirectories: {subdirectories}")
                print(f"Number of Images: {image_count}")
                print(f"Largest Image Size: {largest_size}")
                print(f"Smallest Image Size: {smallest_size}")

                self.subdirectories = subdirectories
                self.image_count = image_count
                self.largest_size = largest_size
                self.smallest_size = smallest_size

                self.enable_layout(True, self.findChild(QVBoxLayout, "infoLayout"))
                self.update_info()
        except Exception as e:
            print("Error occured: ", e)

    # Select a .pkl (pickle) file
    def select_file(self):
        # Opens file explorer
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("All files (*.pkl *.mat *.joblib)")

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            # If the User does select a file (restricted to selecting one file only)
            if selected_files:
                file_path = selected_files[0]
                self.file_directory = file_path
                file_name = os.path.basename(self.file_directory)
                self.file_name = file_name
                self.dataFile.setText("Selected File: " + self.file_name)
                self.dataFile.setToolTip("File Directory: " + str(self.file_directory))

                self.selectDir.setEnabled(False)  # Select Folder button
                self.dataDir.setText("")  # Remove text
                self.confirmSelection.setEnabled(True)
                # Enable reset to clear file selection
                self.resetData.setEnabled(True)
                # if a file has been selected, enable use of spinboxes etc
                self.enable_parameter_layout(True)
                self.get_file_info()
                self.test.print_dataset_distribution()

    # TODO: Implement function to display information about the selected file
    # - Include .mat handling.
    # Helper function to display information about the selected file
    def get_file_info(self):
        self.test = Dataset()
        self.test.load_dataset_from_file(self.file_directory)
        self.image_count = self.test.num_images
        self.num_classes = len(self.test.label)
        self.update_info()
        # self.print_dataset_attributes(self.test)
        # print("Instance Attributes:")
        # test.plot_image(0, 0)
        if self.test.target_size is not None:
            self.resizeX.setValue(self.test.target_size[0])
            self.resizeY.setValue(self.test.target_size[1])
        else:
            self.resizeX.setValue(500)
            self.resizeY.setValue(500)
        self.maxImages.setValue(self.image_count // len(self.test.label))
        # self.enable_parameter_layout(True)
        self.maxImages.setMaximum(self.image_count // len(self.test.label))

        self.enable_layout(True, self.findChild(QVBoxLayout, "infoLayout"))

    def print_dataset_attributes(self, dataset):

        for attr_name, attr_value in vars(dataset).items():
            if (
                not attr_name.startswith("__")
                and not callable(attr_value)
                and not isinstance(attr_value, (QWidget, QVBoxLayout, QHBoxLayout))
            ):
                if attr_name != "data":
                    print(f"- {attr_name}: {attr_value}")

    

    # Update dataset info with appropriate information.
    def update_info(self):
        layout = self.findChild(QVBoxLayout, "datasetInfo")

        info_list = [
            self.folder_directory
            if self.file_directory is None
            else self.file_directory,
            self.folder_name if self.file_name is None else self.file_name,
            self.subdirectories,
            self.image_count,
            self.num_classes if self.folder_name is None else self.subdirectories,
            self.largest_size,
            self.smallest_size,
            (self.resize_x_value, self.resize_y_value),
            str(self.train_test_split) + str("%"),
        ]

        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget():
                item.widget().setText(str(info_list[index]))
        # self.test = Dataset()
        # self.test.num_images = self.image_count
        # self.test.train_test_split = self.train_test_split
        # # self.test.

    # 
    def confirm_selection(self):
        checked = not self.confirmSelection.isChecked() 
        self.update_info()
        self.enable_parameter_layout(checked)
        self.enable_layout(checked, self.findChild(QVBoxLayout, "infoLayout"))
        self.resetSpins.setEnabled(checked)

        self.enable_layout(not checked, self.findChild(QVBoxLayout, "saveToFile"))

    # Helper function to disable and enable items in a given layout.
    def enable_layout(self, enable=True, layout=None):
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget():
                item.widget().setEnabled(enable)
            elif item.layout():
                self.enable_layout(enable, item.layout())

    # Enable or Disable parameterLayout
    def enable_parameter_layout(self, enable=True):
        # Disable/Enable SpinBoxes and Text
        layout = self.findChild(QHBoxLayout, "parameterLayout")
        self.enable_layout(enable, layout)
        self.trainText.setEnabled(enable)
        # These items should always be off
        self.testSpin.setEnabled(False)
        self.testSlider.setEnabled(False)
        self.testText.setEnabled(False)

    #     # Connect the textChanged signal of the line edit to a custom slot
    #     line_edit.textChanged.connect(self.handle_text_changed)

    def save_file(self):
        self.save_file_name = self.fileName.text()
        test = Dataset()

        print("self.file_directory", self.file_directory)
        print("self.folder_directory", self.folder_directory)

        # Directory to load from, pass max number of images if not 0, resize X,Y values, and train/test split size
        dir = self.folder_directory
        if self.max_images_value != 0:
            maxImg = self.max_images_value
        else:
            maxImg = None
        if self.resize_x_value != 0 and self.resize_y_value != 0:
            resize = (self.resize_x_value, self.resize_y_value)
        else:
            resize = None
        split = self.train_test_split
        x_train, x_test, y_train, y_test = test.load_dataset_from_dir(dir, maxImg, resize, split)
        modelData = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        main.transition(self.stack, classifierselect.ClassifierSelect(self.stack, modelData))
        # test.load_dataset_from_dir(self.folder_directory, self.max_images_value if self.max_images_value != 0 else None, (self.resize_x_value, self.resize_y_value) if self.resize_x_value and self.resize_y_value != 0 else None, self.train_test_split)
        # elif self.file_name != None:
        #     test.load_dataset_from_file(self.file_name)

        # # Directory to save to(TODO: Change later), and file name to save as.
        # print("Printing attributes of dataset")
        # self.print_dataset_attributes(test)

        # save_dataset_to_file("Datasets/pickled",self.save_file_name,test)
        

    # Debugging method
    def print_attributes(self):
        print("Instance Attributes:")
        for attr_name, attr_value in vars(self).items():
            if (
                not attr_name.startswith("__")
                and not callable(attr_value)
                and not isinstance(attr_value, (QWidget, QVBoxLayout, QHBoxLayout))
            ):
                print(f"- {attr_name}: {attr_value}")

    # Ctrl+w shortcut to close window for Windows
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Initialize dark theme
    app.setStyleSheet(dark_stylesheet)
    check = ImageLoader()
    check.show()
    sys.exit(app.exec())

   