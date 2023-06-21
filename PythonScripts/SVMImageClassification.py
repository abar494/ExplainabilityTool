# Code in this file has been adapted from the offical LIME tutorial for image classifiers using Random Forests for Support Vector Machines (SVMs):
# https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20MNIST%20and%20RF.ipynb

# Tasks
# 0: Import the appropriate libraries
import os
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
# since the ImageExplanation(image, segments) wants 3D numpy arrays (colour images) for segments
from skimage.color import gray2rgb, rgb2gray, label2rgb

# 1: Load training data for image classification.
root_directory = 'ExplainabilityTool'
root_path = os.path.abspath(root_directory)
print(f"root_path {root_path}")
datasets_directory = 'Datasets'
# target_directory = os.path.join(root_path, datasets_directory)
target_file = 'mnist-original.mat'

# Walk through the directory tree rooted at the target directory
for dirpath, dirnames, filenames in os.walk(root_directory):
    if target_file in filenames:
        file_path = os.path.join(dirpath, target_file)
        print(f"File path: {file_path}")
        break


mnist = loadmat(file_path)
df = pd.DataFrame(mnist)
# print(mnist)

keys = mnist.keys()
values = mnist.values()

print(len(keys))
print(f"keys: {keys}")
print(len(values))
print(values)
# make each image color so lime_image works correctly (i.e. 3D numpy arrays)
# X_vec = np.stack([gray2rgb(iimg)
#                  for iimg in mnist.data.reshape((-1, 28, 28))], 0).astype(np.uint8)
# y_vec = mnist.target.astype(np.uint8)

# fig, ax1 = plt.subplots(1, 1)
# ax1.imshow(X_vec[0], interpolation='none')
# ax1.set_title('Digit: {}'.format(y_vec[0]))


# # 2: Apply SVM model using Scikit learn, choose hyperparameters
# # at your own discretion, or via hyperparameter tuning e.g use gridsearch.

# # 3: Apply Lime explainability object and try to plot the results
# # for one particular prediction.

# # 4: Investigate how you did this and whether you had to modify the
# # structure of the model code to do task 3.

# # 5: try multiple different predictions with lime and write what you find.

# # 6: Extra for experts: Try using a different model such as bayes or xgboost/random forest
# # and try see how your explainability evaluations change for the same image dataset and predictions.

# # 7: Document the following: What parts of your code had to change to achieve this?
# # Do you think this code can be easily swapped out for other models for example?
