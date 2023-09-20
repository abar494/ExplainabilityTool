import os
import sys
import matplotlib.pyplot as plt  # Plotting Images
from PIL import Image
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, uic
import shutil
import random
import numpy as np
import cv2
import main as main
import gallery
# from dataset import Dataset, save_dataset_to_file

class ImageLoader(QWidget):
    def __init__(self, stack): # TODO: Add stack here later
        super().__init__()
        uic.loadUi("FrontEnd/UI/builtLoader.ui", self)
        self.initial_state()
        self.connect_all()
        self.stack = stack
        self.modelData = {
            "x_test": None,
            "y_test": None,
            "iclf": None,
            "clfs": None,
            "vals": None,
            "var": None
        }


    # Sets the initial state of the application and values
    def initial_state(self):
        # Set initial values to default
        self.folder_directory = None
        self.folder_name = None
        self.total_classes = None #0
        self.total_images = None # 0
        self.largest_image = None #(0, 0)
        self.smallest_image = None #(float("inf"), float("inf"))
        self.max_images = 0
        self.resize_xy = 0
        self.dataSelected = False
        self.modelSelected = False
        # User defined parameters
        self.resizeXY.setValue(0)
        self.maxImages.setValue(0)
        # self.reset_params()

        # Set initial layouts and buttons to disabled
        self.enable_layout(False, self.datasetDetails)
        self.enable_layout(False, self.datasetParams)
        self.resetData.setEnabled(False)
        self.continueNext.setEnabled(False)

    # Connect all buttons/sliders/spinboxes to their respective functions
    def connect_all(self):
        self.selectDir.clicked.connect(self.select_folder)
        self.selectMod.clicked.connect(self.select_model)
        self.resetData.clicked.connect(self.reset_data)
        self.resizeXY.valueChanged.connect(self.update_spins)
        self.maxImages.valueChanged.connect(self.update_spins)
        self.confirmSelection.clicked.connect(self.confirm_selection)
        self.resetParams.clicked.connect(self.reset_params)
        self.continueNext.clicked.connect(self.load_data_continue)
        self.back.clicked.connect(lambda: main.transition(self.stack, main.MainMenu(self.stack)))
        # self.resetParams.clicked.connect(self.test_buttons)

    def load_data_continue(self):
        self.load_dataset_from_dir()

        zipped = list(zip(self.x_test, self.y_test))
        random.shuffle(zipped)

        self.x_test, self.y_test = zip(*zipped)

        self.modelData = {
            "x_test": self.x_test,
            "y_test": self.y_test,
            "iclf": self.model_path,
            "clfs": None,
            "vals": None,
            "var": None
        }

        self.sample(120)

        main.transition(self.stack, gallery.Gallery(self.stack, self.modelData, False))


    def sample(self, numSamples):
        folder = "Datasets/sampledata/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        for i in range(min(len(self.x_test), numSamples)):
            im = Image.fromarray(self.x_test[i])
            im.save("Datasets/sampledata/"+str(i)+".png")


    def update_spins(self):
        
        if self.sender() == self.resizeXY:
            self.resize_xy = self.resizeXY.value()
        else:
            self.max_images = self.maxImages.value()

        
        self.resetParams.setEnabled(not self.params_default())       

    # If confirm selection is checked, enable continue button
    def confirm_selection(self):
        # Enable buttons and layouts.
        checked = self.confirmSelection.isChecked() 
        

        if checked == True: #and self.params_default() == False:
            # self.confirmSelection.setEnabled(not checked)
            self.continueNext.setEnabled(True)
            self.enable_layout(False, self.datasetParams)
            self.enable_layout(False, self.datasetDetails)
            self.confirmSelection.setEnabled(True) # This is needed to enable the checkbox because it is inside the datasetParams layout above

        else:
            self.continueNext.setEnabled(False)
            self.enable_layout(True, self.datasetParams)
            self.enable_layout(True, self.datasetDetails)
            # self.confirmSelection.setEnabled(True) # This already gets enabled



        # print(self.confirmSelection.isChecked())
        # self.enable_layout(not checked, self.datasetDetails)
        # self.enable_layout(not checked, self.datasetParams)
        # self.continueNext.setEnabled(not checked)

    # Returns False if any of the parameters are NOT default values
    def params_default(self):
        
        if( (self.maxImages.value()) != 0 or
            (self.resizeXY.value() != 0)
           ):
            return False
        else:
            return True

    # Resets everything their initial state
    def reset_data(self):
        self.initial_state()
        self.total_classes = "X"
        self.total_images = "X"
        self.largest_image = "X"
        self.smallest_image = "X"
        self.folder_directory = ""
        self.folder_name = ""
        # Update information
        self.reset_params()
        self.update_folder_info()
        self.update_info()

    # Reset all parameters to their default values
    def reset_params(self):
        self.maxImages.setValue(0)
        self.resizeXY.setValue(0)
        # Disable reset param button
        self.resetParams.setEnabled(False)

    # Update folder into and tool tips
    def update_data_info(self):
        self.selectedDataset.setText("Selected Dataset: " + self.folder_name)

    def update_model_info(self):
        self.selectedModel.setText("Selected Model: " + self.model_name)

    # Select folder containing dataset
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        # This handles situations where the user cancels the file dialog.
        if folder_path:
            self.folder_name = os.path.basename(folder_path)
            self.folder_directory = folder_path
            self.update_data_info()
            self.get_dataset_info()
            self.update_info()
            # Enable buttons and layouts.
            self.dataSelected = True
            if self.dataSelected and self.modelSelected:
                self.enable_layout(True, self.datasetDetails)
                self.enable_layout(True, self.datasetParams)
                self.resetData.setEnabled(True)

        # Select folder containing dataset
    def select_model(self):

        # options = QFileDialog.Options()  # Ensure that the selected file is read-only
        # options |= QFileDialog.ExistingFile  # Ensure that the selected file already exists
        # model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.h5 *.H5);;All Files (*)", options=options)
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        # model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.h5);;All Files (*)", options=options)
        if model_path:
            # Check if the selected file has the .h5 extension
            if model_path.endswith(".h5"):
                # Store the selected model file path
                self.model_name = os.path.basename(model_path)
                self.model_path = model_path
                self.update_model_info()
                # Enable buttons and layouts.
                self.modelSelected = True
                if self.dataSelected and self.modelSelected:
                    self.enable_layout(True, self.datasetDetails)
                    self.enable_layout(True, self.datasetParams)
                    self.resetData.setEnabled(True)
            else:
                # Show an error message if the selected file is not an .h5 file
                QMessageBox.critical(self, "Invalid File", "Please select a valid .h5 model file.")



        # model_path = QFileDialog.getExistingDirectory(self, "Select Model")
        # # This handles situations where the user cancels the file dialog.
        # if model_path:
        #     self.model_name = os.path.basename(model_path)
        #     self.model_path = model_path
        #     self.update_model_info()
        #     # Enable buttons and layouts.
        #     self.modelSelected = True
        #     if self.dataSelected and self.modelSelected:
        #         self.enable_layout(True, self.datasetDetails)
        #         self.enable_layout(True, self.datasetParams)
        #         self.resetData.setEnabled(True)


    # Helper function to disable and enable items in a given layout. Ideal for parent layouts
    def enable_layout(self, enable=True, layout=None):
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget():
                item.widget().setEnabled(enable)
            elif item.layout():
                self.enable_layout(enable, item.layout())

 # Get info about chosen folder.
    def get_dataset_info(self):
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

                self.total_images = image_count
                self.largest_image = largest_size
                self.smallest_image = smallest_size
                self.total_classes = subdirectories

        except Exception as e:
            print("Error occured: ", e)

    # Updates the details of the dataset info layout
    def update_info(self):
        layout = self.findChild(QVBoxLayout, "details")

        info_list = [
            self.total_classes,
            self.total_images,
            self.largest_image,
            self.smallest_image,
        ]
        # Iterate through the layout and update the text of each label
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget():
                item.widget().setText(str(info_list[index]))

    # Ctrl+w shortcut to close window for Windows/Linux
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_W:
            self.close()

    # Loads images from the selected root directory into a dictionary 
    # The dictionary contains numpy arrays for the train and testing data,
    # as well as the corresponding (numerical) labels for each image 
    def load_dataset_from_dir(self):
        dataset = []
        # Could be refactored, but this is fine for now
        class_labels = []  # List to store the corresponding labels for each image
        image_formats = [".jpg", ".jpeg", ".png", ".jfif"] # Add more image formats here if needed

        root_dir = self.folder_directory
        limit = self.max_images
        target_size = (self.resize_xy, self.resize_xy)

        # Iterate through each subdirectory in the root directory
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            images = [] # List to store the images for the current class
            num_loaded_images = 0  # Track the number of images loaded for the current class

            for file_name in os.listdir(class_dir):
                if num_loaded_images == limit:
                    break  # Reached the limit for the current class

                # Get the full path of the image file
                file_path = os.path.join(class_dir, file_name)
                if not os.path.isfile(file_path):
                    continue
                
                # Check if the file is an image
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in image_formats:
                    continue

                try:
                    # Load the image using OpenCV. Changed from PIL to OpenCV for future image processing
                    image = cv2.imread(file_path)
                    # As OpenCV uses BGR, convert from BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # All images need to be resized to the same size
                    # This is a required by numpy arrays, otherwise it will throw errors
                    # about inconsistent array shapes and inhomogeneous arrays
                    if target_size is None:
                        target_size = (500,500) # Note: OpenCV uses (width, height) instead of (height, width)
                    else:    
                        image = cv2.resize(image_rgb, target_size)

                    images.append(image)
                    num_loaded_images += 1  # Increment the count of loaded images
                    class_labels.append(class_name)  # Add the label for the current image

                except Exception as e:
                    print(f"Error loading image: {file_path} ({e})")

            # Add the images for the current class to the dataset
            dataset.extend(images)

        # Convert the dataset and labels to numpy arrays
        dataset_array = np.array(dataset)
        class_labels_array = np.array(class_labels)

        self.root = root_dir
        self.limit = limit
        self.target_size = target_size
        self.num_images = num_loaded_images
        self.data = dataset_array
        self.label = np.array(class_labels_array)

        # Get a list of class names from subdirectories    
        class_list = os.listdir(root_dir)

        self.num_class_labels = []
        for i in range(len(self.label)):
            self.num_class_labels.append(self.enumerate(self.label[i], class_list))

        self.x_test, self.y_test = self.data, self.num_class_labels
        # self.x_train, self.x_test, self.y_train , self.y_test = train_test_split_sklearn(self.data, self.num_class_labels, test_size=1)  
        # Dictionary to store the data for use in proceeding pages

    
    def enumerate(self, animal, animalList):
        for i in range(len(animalList)):
            if animal == animalList[i]:
                return i

     