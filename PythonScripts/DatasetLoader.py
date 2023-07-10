import pickle
import os
from PIL import Image
import numpy as np
# Convert gray images to RGB (increase number of channels)
from skimage.color import gray2rgb
import matplotlib.pyplot as plt  # Plotting Images
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5 import uic
import sys

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

"""


class Dataset():
    """
    Dataset loader. Loads a given dataset from its root directory.
    It expects a root directory and several subdirectories. The subdirectories
    should correspond to the image classes (e.g., folder name: cats, should only contain images of cats)

    This class contains numpy arrays containing the image and label data.
    By default, it loads dataset from a directory.
    """

    def __init__(self, root_dir=None, limit=None, target_size=None):
        self.root = root_dir
        self.limit = limit
        self.target_size = target_size
        self.num_images = 0
        self.train_test_split = 0

        if root_dir is not None:
            self.data, self.label = self.load_dataset_from_dir(
                root_dir=self.root, limit=self.limit, target_size=self.target_size)
        else:
            print(
                "Warning: No loaded dataset. root_dir is None. All attributes are None or empty.")
            self.data = np.array([])
            self.label = np.array([])

    def get_num_images(self):
        self.num_images = len(np.concatenate(self.data))
        return self.num_images

    def load_dataset_from_dir(self, root_dir, limit=None, target_size=None):
        """
        Load desired dataset from a directory.
        root_dir: root directory of the desired dataset
        limit: maximum number of images per class, can be left as None
        target_size: desired image resize, can be left as None
        """

        dataset = []
        class_labels = []
        image_formats = ['.jpg', '.jpeg', '.png', '.jfif']

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_labels.append(class_name)
            images = []
            loaded_images = 0  # Track the number of images loaded for the current class

            for file_name in os.listdir(class_dir):
                if loaded_images == limit:
                    break  # Reached the limit for the current class

                file_path = os.path.join(class_dir, file_name)
                if not os.path.isfile(file_path):
                    continue

                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in image_formats:
                    continue

                try:
                    image = Image.open(file_path)
                    if target_size is not None:
                        image = image.resize(target_size)

                    # Convert to uint8 data type
                    image_array = np.array(image, dtype=np.uint8)
                    # gray images don't have 3 channels.
                    if len(image_array.shape) != 3:
                        image_array = gray2rgb(image_array)
                    images.append(image_array)
                    loaded_images += 1  # Increment the count of loaded images
                except Exception as e:
                    print(f"Error loading image: {file_path} ({e})")

            dataset.append(images)

        # Convert the dataset and labels to numpy arrays
        self.data = np.array(dataset)
        self.label = np.array(class_labels)

        return self.data, self.label

    # TODO: Implement .mat handling
    def load_dataset_from_file(self, file_dir):
        """
        Load desired dataset from a pickle (.pkl) file.

        Attributes:
        file_dir - Path to desired .pkl file.
        """
        with open(file_dir, 'rb') as file:
            loaded_dataset = pickle.load(file)

        # Copies attributes from loaded Dataset object into the Dataset object calling this function.
        for attr_name in loaded_dataset.__dict__:
            setattr(self, attr_name, getattr(loaded_dataset, attr_name))
        self.get_num_images()

    def save_dataset_to_file(self, file_dir, file_name="dataset"):
        # Open a file in binary mode
        with open(file_dir+file_name+'.pkl', 'wb') as file:
            # Dump the object to the file
            pickle.dump(self, file)
            print(
                f"File saved to directory: {str(file_dir+file_name+'.pkl')} ")

    def plot_image(self, image_num=0, class_label=0):
        # plot the image with matplotlib using based on input params.
        plt.imshow(self.data[class_label][image_num])
        plt.title(self.label[class_label])
        plt.axis('off')
        plt.show()


class DataLoader(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("DatasetLoader.ui", self)
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
        self.back.clicked.connect(self.print_attributes)  # temp usage of back
        self.maxImages.valueChanged.connect(self.update_spins)
        self.resizeX.valueChanged.connect(self.update_spins)
        self.resizeY.valueChanged.connect(self.update_spins)
        # Connect Train/Test Split sliders and spinboxes to their function
        self.trainSlider.valueChanged.connect(self.update_sliders)
        self.trainSpin.valueChanged.connect(self.update_sliders)
        # self.saveFile.clicked.connect(
        #     lambda: self.save_file(self.fileName))

    # Set attributes to their defaults
    def init_initial_values(self):
        self.max_images_value = 0
        self.resize_x_value = None
        self.resize_y_value = None
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
        self.default_train_split = self.train_test_split = 75
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
        self.loadData.setEnabled(False)
        self.dataFile.setText("Selected File: None")
        self.dataDir.setText("Selected Folder: None")
        self.enable_parameter_layout(False)
        self.maxImages.setMaximum(100000)
        self.saveFile.setEnabled(False)
        self.fileName.setEnabled(False)
        # Initially disable information layout
        self.enable_layout(False, self.findChild(QVBoxLayout, "infoLayout"))

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
        if (self.max_images_value != self.default_max_images or
            self.resize_x_value != self.default_max_resize or
            self.resize_y_value != self.default_max_resize or
                self.train_test_split != self.default_train_split):

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

    # Select the directory containing the dataset. Enable appropriate Pushbuttons and SpinBoxes
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, 'Select Folder')
        if folder_path:
            self.folder_name = os.path.basename(folder_path)
            self.folder_directory = folder_path
            self.dataDir.setText("Selected Folder: " + self.folder_name)
            self.dataDir.setToolTip("Folder Directory: " +
                                    str(self.folder_directory))

            self.selectData.setEnabled(False)
            self.dataFile.setText("")
            self.loadData.setEnabled(True)
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
        smallest_size = (float('inf'), float('inf'))
        try:
            # Check if the folder exists
            if os.path.exists(folder_directory):
                # Get the list of subdirectories and files within the folder
                entries = os.scandir(folder_directory)

                for entry in entries:
                    if entry.is_dir():
                        subdirectories += 1
                        # Count the number of images within each subdirectory
                        subdirectory_path = os.path.join(
                            folder_directory, entry.name)
                        images = [name for name in os.listdir(subdirectory_path) if os.path.isfile(
                            os.path.join(subdirectory_path, name))]

                        for image_name in images:
                            image_path = os.path.join(
                                subdirectory_path, image_name)
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

                self.enable_layout(True, self.findChild(
                    QVBoxLayout, "infoLayout"))
                self.update_info()
        except Exception as e:
            print("Error occured: ", e)

    # Select a .pkl (pickle) file
    def select_file(self):
        # Opens file explorer
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("All files (*.pkl *.mat)")

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            # If the User does select a file (restricted to selecting one file only)
            if selected_files:
                file_path = selected_files[0]
                self.file_directory = file_path
                file_name = os.path.basename(self.file_directory)
                self.file_name = file_name
                self.dataFile.setText("Selected File: " + self.file_name)
                self.dataFile.setToolTip("File Directory: " +
                                         str(self.file_directory))

                self.selectDir.setEnabled(False)  # Select Folder button
                self.dataDir.setText("")  # Remove text
                self.loadData.setEnabled(True)
                # Enable reset to clear file selection
                self.resetData.setEnabled(True)
                # if a file has been selected, enable use of spinboxes etc
                self.enable_parameter_layout(True)
                self.get_file_info()

    # TODO: Implement function to display information about the selected file
    # - Include .mat handling.
    # Helper function to display information about the selected file
    def get_file_info(self):

        test = Dataset()
        test.load_dataset_from_file(self.file_directory)
        self.image_count = test.num_images
        self.num_classes = len(test.label)
        self.update_info()
        print("Instance Attributes:")
        # test.plot_image(0, 0)
        self.resizeX.setValue(test.target_size[0])
        self.resizeY.setValue(test.target_size[1])
        self.maxImages.setValue(self.image_count//len(test.label))
        # self.enable_parameter_layout(True)
        self.maxImages.setMaximum(self.image_count//len(test.label))
        # self.
        self.enable_layout(True, self.findChild(QVBoxLayout, "infoLayout"))

        for attr_name, attr_value in vars(test).items():
            if not attr_name.startswith('__') and not callable(attr_value) and not isinstance(attr_value, (QWidget, QVBoxLayout, QHBoxLayout)):
                if attr_name != "data":
                    print(f"- {attr_name}: {attr_value}")

    # Update dataset info with appropriate information.
    def update_info(self):
        layout = self.findChild(QVBoxLayout, "datasetInfo")

        info_list = [self.folder_directory if self.file_directory is None else self.file_directory,
                     self.folder_name if self.file_name is None else self.file_name,
                     self.subdirectories,
                     self.image_count,
                     self.num_classes if self.folder_name is None else self.subdirectories,
                     self.largest_size,
                     self.smallest_size,
                     (self.resize_x_value, self.resize_y_value),
                     str(self.train_test_split) + str("%")]

        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget():
                item.widget().setText(str(info_list[index]))

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
        test = Dataset()

        # test.save_dataset_to_file(self.save_file_name
        # )
        # self.save_file_name = self.
        # print()

    # Debugging method
    def print_attributes(self):
        print("Instance Attributes:")
        for attr_name, attr_value in vars(self).items():
            if not attr_name.startswith('__') and not callable(attr_value) and not isinstance(attr_value, (QWidget, QVBoxLayout, QHBoxLayout)):
                print(f"- {attr_name}: {attr_value}")

    # Ctrl+w shortcut to close window for Windows
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Initialize dark theme
    app.setStyleSheet(dark_stylesheet)
    check = DataLoader()
    check.show()
    sys.exit(app.exec())

    # Example Using Load from Directory
    root = r"C:\Users\space\OneDrive\Desktop\p4p\Caleb\Datasets\leaves\leaves"
    # limit = 200  # Set the desired limit for the subset
    # target_size = (300, 300)  # Set the desired target size for resizing
    # Load from directory
    dataset1 = Dataset(root, limit, target_size)
    # __init__ function calls load_dataset_from_dir if root is provided.
    dataset1.plot_image(0, 2)
    # Path to save to
    saveto = r"C:\Users\space\OneDrive\Desktop\p4p\Caleb\\"
    savetofile = "leaves"
    dataset1.save_dataset_to_file(saveto, savetofile)

    print("Labels =", dataset1.label)
    print("Number of total images", dataset1.get_num_images())

    # Example Using Load from File
    saved = r"C:\Users\space\OneDrive\Desktop\p4p\Caleb\leaves.pkl"  # Path to file
    dataset2 = Dataset()
    dataset2.load_dataset_from_file(saved)

    print("Labels =", dataset2.label)
    print("Number of total images", dataset2.get_num_images())

    # Plotting the image
    dataset2.plot_image(0, 2)
