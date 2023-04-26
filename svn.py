import numpy as np
import pandas as pd
import os
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
from skimage.color import rgb2gray

import PIL
import cv2
import pickle

sign_paths = glob('archive/images/*')

example_image = rgb2gray(np.asarray(PIL.Image.open(sign_paths[0]))[:,:,:3])
hog_features, visualized = hog(example_image,orientations=9,pixels_per_cell=(16,16),
                              cells_per_block=(2,2),
                              visualize=True
                             )
fig = plt.figure(figsize=(12,6))
fig.add_subplot(1,2,1)
plt.imshow(example_image)
plt.axis("off")
fig.add_subplot(1,2,2)
plt.imshow(visualized, cmap="gray")
plt.axis("off")
plt.show()

hog_features.shape

sign_images = []
sign_labels = np.ones(len(sign_paths))

start = time.time()

for sign_path in sign_paths:
    img = np.asarray(PIL.Image.open(sign_path))
    img = cv2.cvtColor(cv2.resize(img, (96, 64)), cv2.COLOR_RGB2GRAY)
    img = hog(img, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2))

    sign_images.append(img)

x = np.asarry(sign_images)
y = np.asarray(list(sign_labels))

processTime = round(time.time() - start, 2)
print("Reading images and extracting features has taken {} sedons".format(processTime))

print("Shape of image set", x.shape)
print("Shape of label set", y.shape)