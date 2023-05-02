import numpy as np
from sklearn.model_selection import train_test_split
import os

# Path: yolo\data\yolo_data_split.py

for dir in os.listdir('diy_model/data/train/'):
    images = os.listdir('diy_model/data/train/' + dir)

    # split the data into train and test sets
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=27)

    # move the test images and labels into the test folder
    for image in test_images:
        os.rename("diy_model/data/train/" + dir + '/' + image, "diy_model/data/test/" + dir + '/' + image)