import pickle
import os
from PIL import Image
import numpy as np
# Convert gray images to RGB (increase number of channels)
from skimage.color import gray2rgb
import matplotlib.pyplot as plt  # Plotting Images
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtGui import QKeySequence
import sys
import inspect

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
        # Connect buttons to their functions
        self.selectData.clicked.connect(self.select_pkl_file)
        self.selectDir.clicked.connect(self.select_folder)
        self.resetData.clicked.connect(self.clear_selection)
        self.resetSpins.clicked.connect(self.clear_nums)
        self.maxImages.valueChanged.connect(self.update_and_enable_clear_spins)
        self.resizeX.valueChanged.connect(self.update_and_enable_clear_spins)
        self.resizeY.valueChanged.connect(self.update_and_enable_clear_spins)
        # Connect Train/Test Split sliders and spinboxes to their function
        self.trainSlider.valueChanged.connect(self.update_sliders)
        self.trainSpin.valueChanged.connect(self.update_sliders)
        # Ensuring Test text, spinbox, and slider are disabled (Should always be disabled).
        self.testSlider.setEnabled(False)
        self.testSpin.setEnabled(False)
        self.testText.setEnabled(False)
        # Initially None or Zero
        self.max_images_value = 0
        self.resize_x_value = 0
        self.resize_y_value = 0
        self.train_test_split = 0.6  # training data e.g. 60%  (0.6).
        self.folder_name = None
        self.folder_directory = None
        self.file_name = None
        self.file_directory = None
        # Initially most buttons are disabled.
        self.loadData.setEnabled(False)
        self.resetData.setEnabled(False)
        self.resetSpins.setEnabled(False)
        self.enable_spin_boxes_sliders_and_text(False)
        # Set initial tool tips
        self.dataDir.setToolTip("Folder Directory: None")
        self.dataFile.setToolTip("File Directory: None")
        # Initially disable information layout
        self.layout = self.findChild(QVBoxLayout, "infoLayout")
        self.enable_layout(False, self.layout)

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
        self.train_test_split = train_test_split / 100

        # If not default train/test split, enable reset
        if (self.train_test_split != 0.6):
            self.resetSpins.setEnabled(True)
        else:
            self.resetSpins.setEnabled(False)
        # print(self.train_test_split)

    # Update attributes with current spinbox values, and enable clearing of spinboxes.
    def update_and_enable_clear_spins(self):
        self.max_images_value = self.maxImages.value()
        self.resize_x_value = self.resizeX.value()
        self.resize_y_value = self.resizeY.value()
        print(self.max_images_value, self.resize_x_value, self.resize_y_value)
        # If any of the spinboxes have a none default value, enable the clear button
        if (self.max_images_value != 0 or self.resize_x_value != 0 or self.resize_y_value != 0):
            self.resetSpins.setEnabled(True)
        else:
            self.resetSpins.setEnabled(False)

    # Set spinbox values to 0 and disable clear button.
    def clear_nums(self):
        # Setting these values to 0 also automatically updates the corresponding class attributes (via ValueChanged()).
        self.maxImages.setValue(0)
        self.resizeX.setValue(0)
        self.resizeY.setValue(0)
        self.trainSpin.setValue(60)
        self.resetSpins.setEnabled(False)  # Disable Reset after pressing.

    # Clear selected files, re-enable and re-disable appropriate buttons and text
    def clear_selection(self):
        self.selectDir.setEnabled(True)
        self.selectData.setEnabled(True)
        self.loadData.setEnabled(False)
        self.dataFile.setText("Selected File: None")
        self.dataDir.setText("Selected Folder: None")
        self.folder_name = None
        self.file_name = None
        self.resetData.setEnabled(False)
        self.enable_spin_boxes_sliders_and_text(False)
        self.clear_nums()

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
        # Ensures that everything is dis/enabled incase user cancels folder selection.
        if self.folder_name != None:
            self.selectData.setEnabled(False)
            self.dataFile.setText("")
            self.loadData.setEnabled(True)
            self.resetData.setEnabled(True)
            self.enable_spin_boxes_sliders_and_text(True)

    # TODO: Implement helper function to display information about the selected folder/directory
    # Helper function to display information about the selected folder/directory
    def get_folder_info(self):
        print("Testing")

    # TODO: Implement function to display information about the selected file
    # Helper function to display information about the selected file
    def get_file_info(self):
        pass

    # Disable or enable any layout.
    def enable_layout(self, enable=True, layout=None):
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget():
                item.widget().setEnabled(enable)
            elif item.layout():
                self.enable_layout(enable, item.layout())

    # Allows user to select a pickle file (.pkl) OR a MATLAB file (.mat)
    def select_pkl_file(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("All files (*.pkl *.mat)")

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                self.file_directory = file_path
                file_name = os.path.basename(self.file_directory)
                self.file_name = file_name
                self.dataFile.setText("Selected File: " + self.file_name)
                self.dataFile.setToolTip("File Directory: " +
                                         str(self.file_directory))
        # Ensures that everything is dis/enabled incase user cancels file selection.
        if self.file_name != None:
            self.selectDir.setEnabled(False)
            self.dataDir.setText("")
            self.loadData.setEnabled(True)
            self.resetData.setEnabled(True)
            # if a file has been selected, disable use of spinboxes
            self.enable_spin_boxes_sliders_and_text(True)

    #
    def enable_spin_boxes_sliders_and_text(self, enable=True):
        # Disable/Enable SpinBoxes
        layout = self.findChild(QHBoxLayout, "parameterLayout")
        self.enable_layout(enable, layout)
        self.testSpin.setEnabled(False)
        self.testSlider.setEnabled(False)
        self.testText.setEnabled(False)
        self.trainText.setEnabled(False)
        # self.maxImages.setEnabled(enable)
        # self.resizeX.setEnabled(enable)
        # self.resizeY.setEnabled(enable)
        # # Disable/Enable Text (change colours)
        # self.maxText.setEnabled(enable)
        # self.resizeXText.setEnabled(enable)
        # self.resizeYText.setEnabled(enable)

        # # Check which method is calling this method.
        # # caller = inspect.currentframe().f_back.f_code.co_name
        # # if caller == "select_pkl_file":
        # #     enable = not enable
        # # Disable/Enable Train/Test Sliders, SpinBoxes, and Text
        # self.trainSpin.setEnabled(enable)
        # self.trainSlider.setEnabled(enable)
        # self.trainTestText.setEnabled(enable)
        # self.trainText.setEnabled(enable)

    # Ctrl+w shortcut to close window for Windows

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Initialize the dark theme
    app.setStyleSheet(dark_stylesheet)
    check = DataLoader()
    check.show()
    sys.exit(app.exec())

    # Example Using Load from Directory
    root = r"C:\Users\space\OneDrive\Desktop\p4p\Caleb\Datasets\leaves\leaves"
    limit = 200  # Set the desired limit for the subset
    target_size = (300, 300)  # Set the desired target size for resizing
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
