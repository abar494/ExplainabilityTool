import pickle
import os
from PIL import Image
import numpy as np
# Convert gray images to RGB (increase number of channels)
from skimage.color import gray2rgb
import matplotlib.pyplot as plt  # Plotting Images


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
