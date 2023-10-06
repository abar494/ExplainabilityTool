import os
import shutil
import random

# Define the source directory where all animal images are located
source_dir = '/home/caleb/Desktop/p4p/ExplainabilityTool/Datasets/keras_datasets/animals'

# Define the destination directories for train, validation, and test sets
train_dir = '/home/caleb/Desktop/p4p/ExplainabilityTool/Datasets/keras_datasets/train'
validation_dir = '/home/caleb/Desktop/p4p/ExplainabilityTool/Datasets/keras_datasets/valid'
test_dir = '/home/caleb/Desktop/p4p/ExplainabilityTool/Datasets/keras_datasets/test'

# Create destination directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define class labels (subdirectories)
classes = ['cat', 'dog', 'panda']

# Set the split ratios
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.1

for class_label in classes:
    class_source_dir = os.path.join(source_dir, class_label)
    class_train_dir = os.path.join(train_dir, class_label)
    class_validation_dir = os.path.join(validation_dir, class_label)
    class_test_dir = os.path.join(test_dir, class_label)
    
    # Create class-specific destination directories
    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_validation_dir, exist_ok=True)
    os.makedirs(class_test_dir, exist_ok=True)
    
    # List all image files in the source directory
    image_files = os.listdir(class_source_dir)
    random.shuffle(image_files)  # Shuffle the list randomly
    
    # Split the images into train, validation, and test sets
    num_images = len(image_files)
    num_train = int(num_images * train_ratio)
    num_validation = int(num_images * validation_ratio)
    
    # Copy images to the appropriate directories based on the split ratios
    for i, image_file in enumerate(image_files):
        src_path = os.path.join(class_source_dir, image_file)
        if i < num_train:
            dst_path = os.path.join(class_train_dir, image_file)
        elif i < num_train + num_validation:
            dst_path = os.path.join(class_validation_dir, image_file)
        else:
            dst_path = os.path.join(class_test_dir, image_file)
        shutil.copy(src_path, dst_path)
