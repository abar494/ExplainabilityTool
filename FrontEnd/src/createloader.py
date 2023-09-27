import os
from PIL import Image
from PyQt6.QtWidgets import QFileDialog, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6 import uic
import numpy as np
import cv2
import main as main
import noise
from sklearn.model_selection import train_test_split as train_test_split_sklearn
# from dataset import Dataset, save_dataset_to_file

class ImageLoader(QWidget):
    def __init__(self, stack): # TODO: Add stack here later
        super().__init__()
        uic.loadUi("FrontEnd/UI/createLoader.ui", self)
        self.default_split = 60 # This is the default split for the train/test sliders
        # Initialise initial states
        self.initial_state()
        self.connect_all()
        self.stack = stack
        self.data_dict = {"x_train": None,
                          "x_test": None,
                          "y_train": None,
                          "y_test": None}


    # Sets the initial state of the application and values
    def initial_state(self):
        # Set initial values to default
        self.folder_directory = None
        self.folder_name = None
        self.total_classes = None #0
        self.total_images = None # 0
        self.largest_image = None #(0, 0)
        self.smallest_image = None #(float("inf"), float("inf"))
        self.max_images = 100
        self.resize_xy = 100
        # User defined parameters
        # Set initial values for train/test sliders and spinboxes
        self.train_size = self.default_split # Default Train value
        self.test_size = 100 - self.train_size
        self.trainSlider.setValue(self.train_size)
        self.testSlider.setValue(self.test_size) # Should always reflect the complement of the train slider
        self.trainSpin.setValue(self.train_size)
        self.testSpin.setValue(self.test_size)
        self.resizeXY.setValue(0)
        self.maxImages.setValue(0)
        self.reset_params()

        # Set initial layouts and buttons to disabled
        self.enable_layout(False, self.datasetDetails)
        self.enable_layout(False, self.datasetParams)
        self.resetData.setEnabled(False)
        self.continueNext.setEnabled(False)

    # Connect all buttons/sliders/spinboxes to their respective functions
    def connect_all(self):
        self.selectDir.clicked.connect(self.select_folder)
        self.resetData.clicked.connect(self.reset_data)
        self.trainSlider.valueChanged.connect(self.update_sliders)
        self.trainSpin.valueChanged.connect(self.update_sliders) # update_sliders() handles both sliders and spin boxes
        self.resizeXY.valueChanged.connect(self.update_spins)
        self.maxImages.valueChanged.connect(self.update_spins)
        self.confirmSelection.clicked.connect(self.confirm_selection)
        self.resetParams.clicked.connect(self.reset_params)
        self.continueNext.clicked.connect(self.load_data_continue)
        self.back.clicked.connect(lambda: main.transition(self.stack, main.MainMenu(self.stack)))

    def load_data_continue(self):
        self.load_dataset_from_dir()

        main.transition(self.stack, noise.Noise(self.stack, self.data_dict))


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
            self.disable_test()
        else:
            self.continueNext.setEnabled(False)
            self.enable_layout(True, self.datasetParams)
            self.enable_layout(True, self.datasetDetails)
            # self.confirmSelection.setEnabled(True) # This already gets enabled
            self.disable_test()

    # Returns False if any of the parameters are NOT default values
    def params_default(self):
        
        if( (self.maxImages.value()) != 0 or
            (self.resizeXY.value() != 0) or
            (self.trainSlider.value() != self.default_split)
           ):
            return False
        else:
            return True

    # Disable test slider, text, and spinbox
    def disable_test(self):
        self.testSlider.setEnabled(False)
        self.testSpin.setEnabled(False)
        self.testText.setEnabled(False)

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
        self.maxImages.setValue(100)
        self.resizeXY.setValue(100)
        self.trainSlider.setValue(self.default_split) 
        # Disable reset param button
        self.resetParams.setEnabled(False)
        self.disable_test()

    # Update folder into and tool tips
    def update_folder_info(self):
        self.selectedDataset.setText("Selected Dataset: " + self.folder_name)
        self.selectedDataset.setToolTip("Dataset Directory: " + str(self.folder_directory))
        self.selectDir.setToolTip("Dataset Directory: " + str(self.folder_directory))

    # Select folder containing dataset
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        # This handles situations where the user cancels the file dialog.
        if folder_path:
            self.folder_name = os.path.basename(folder_path)
            self.folder_directory = folder_path
            self.update_folder_info()
            self.get_dataset_info()
            self.update_info()
            # Enable buttons and layouts.
            self.enable_layout(True, self.datasetDetails)
            self.enable_layout(True, self.datasetParams)
            self.resetData.setEnabled(True)
            self.disable_test() # "Test" text, slider, and spinbox should always be disabled

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
        self.resetParams.setEnabled(self.params_default())

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
        train_test_split = self.train_size

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
                        target_size = (100,100) # Note: OpenCV uses (width, height) instead of (height, width)
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
        # self.train_test_split = train_test_split
        self.data = dataset_array
        self.label = np.array(class_labels_array)
        test_size = (100-self.train_size)/100 

        # Get a list of class names from subdirectories    
        class_list = os.listdir(root_dir)

        self.num_class_labels = []
        for i in range(len(self.label)):
            self.num_class_labels.append(self.enumerate(self.label[i], class_list))


        self.x_train, self.x_test, self.y_train , self.y_test = train_test_split_sklearn(self.data, self.num_class_labels, test_size=test_size)  
        # Dictionary to store the data for use in proceeding pages
        self.data_dict = { "x_train": self.x_train,
                            "x_test": self.x_test,
                            "y_train": self.y_train,
                            "y_test": self.y_test  }
    
    def enumerate(self, label, label_list):
        for i in range(len(label_list)):
            if label == label_list[i]:
                return i
