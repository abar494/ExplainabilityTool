import numpy as np
import os
import cv2
import joblib
import PIL.Image as Image
import pickle
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split as train_test_split_sklearn

class Dataset:
    """
    Dataset loader. Loads a given dataset from its root directory.
    It expects a root directory and several subdirectories. The subdirectories
    should correspond to the image classes (e.g., folder name: cats, should only contain images of cats)

    This class contains numpy arrays containing the image and label data.
    """

    def __init__(self, root_dir=None, limit=None, target_size=None, train_test_split = None):
        self.root = root_dir
        self.limit = limit
        self.target_size = target_size
        self.num_images = 0
        self.train_test_split = train_test_split
        self.data = np.array([])
        self.label = np.array([])
        # Train/test split
        self.x_train = np.array([]) # Images for training
        self.y_train = np.array([]) # Labels for training
        self.x_test = np.array([])
        self.y_test = np.array([])


    def get_num_images(self):
        self.num_images = len(np.concatenate(self.data))
        return self.num_images
    
    # Loads a number of images from a directory, resizes them to a given size and returns them as a numpy array
    # WARNING: You may run out of memeory if you try to load too many images at once. 
    # 8GB of RAM may not be enough if you have other applications running in the background using resources
    def load_dataset_from_dir(self, root_dir, limit=None, target_size=None, train_test_split=None):
        dataset = []
        class_labels = []  # List to store the corresponding labels for each image
        image_formats = [".jpg", ".jpeg", ".png", ".jfif"] # Add more image formats here if needed

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
                    # About inconsistent array shapes and inhomogeneous arrays
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
        self.train_test_split = train_test_split
        self.data = dataset_array
        self.label = np.array(class_labels_array)
        test_size = (100-self.train_test_split)/100 
        self.x_train, self.x_test, self.y_train , self.y_test = train_test_split_sklearn(self.data, self.label, test_size=test_size)  

        return dataset_array, class_labels_array

    # TODO: Implement .mat handling
    def load_dataset_from_file(self, file_dir):
        """
        Load desired dataset from a pickle (.pkl) file.

        Attributes:
        file_dir - Path to desired .pkl file.
        """
        with open(file_dir, "rb") as file:
           # loaded_dataset = pickle.load(file)
            loaded_dataset = joblib.load(file)

        # Copies attributes from the joblib file to this class
        self.root = loaded_dataset.root
        self.limit = loaded_dataset.limit
        self.target_size = loaded_dataset.target_size
        self.num_images = loaded_dataset.num_images
        self.train_test_split = loaded_dataset.train_test_split
        self.data = loaded_dataset.data
        self.label = loaded_dataset.label
        self.x_train = loaded_dataset.x_train
        self.x_test = loaded_dataset.x_test
        self.y_train = loaded_dataset.y_train
        self.y_test = loaded_dataset.y_test


    def plot_image(self, image_num=0, class_label=0):
        # plot the image with matplotlib using based on input params.
        plt.imshow(self.data[class_label][image_num])
        plt.title(self.label[class_label])
        plt.axis("off")
        plt.show()

    # Saves the split dataset to a .joblib file
    def save_train_test_split(self, file_dir, file_name="dataset"):
        
        train_test = [self.x_train, self.y_train, self.x_test, self.y_test]

        file_name = str('/' + file_name)
        try:
                with open(file_dir + file_name + ".joblib", "wb") as file:
                    # Dump the object to the file   
                        # pickle.dump(dataset, file)
                        joblib.dump(train_test, file)

                print(f"File {file_name} saved to directory: {str(file_dir+file_name+'.joblib')} ")

        except Exception as e:
                print(f"Error while saving {file_name}.joblib: {e}")

    # returns the split dataset in a list
    def get_train_test_split(self):
        return [self.x_train, self.y_train, self.x_test, self.y_test]    

    def print_dataset_distribution(self):
         # Calculate the class distribution in the train dataset
        unique_classes_train, train_class_counts = np.unique(self.y_train, return_counts=True)

        # Calculate the class distribution in the test dataset
        unique_classes_test, test_class_counts = np.unique(self.y_test, return_counts=True)

        print("Train Dataset Class Distribution:")
        for cls, count in zip(unique_classes_train, train_class_counts):
            print(f"Class: {cls}, Count: {count}")

        print("\nTest Dataset Class Distribution:")
        for cls, count in zip(unique_classes_test, test_class_counts):
            print(f"Class: {cls}, Count: {count}")


# Saves the entire Dataset object to a .joblib file
def save_dataset_to_file(file_dir, file_name="dataset", dataset=None):
    # Open a file in binary mode
    file_name = str('/' + file_name)
    if dataset is not None:
        try:
            with open(file_dir + file_name + ".joblib", "wb") as file:
                # Dump the object to the file   
                    # pickle.dump(dataset, file)
                    joblib.dump(dataset, file)

            print(f"File {file_name} saved to directory: {str(file_dir+file_name+'.pkl')} ")

        except Exception as e:
            print(f"Error while saving {file_name}.pkl: {e}")
    else: 
        print("No dataset object has been passed.")

