import numpy as np
from sklearn.model_selection import train_test_split
import os

# Path: yolo\data\yolo_data_split.py

# get list of images
images = os.listdir("C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolo/data/images")
# get list of yolo labels
labels = os.listdir("C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolo/data/labels")

# split the data into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=27)

# move the train images and labels into the train folder
for image in train_images:
    os.rename("C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolo/data/images/" + image, "C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolov7/data/train/" + image)
for label in train_labels:
    os.rename("C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolo/data/labels/" + label, "C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolov7/data/train/" + label)

# move the test images and labels into the test folder
for image in test_images:
    os.rename("C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolo/data/images/" + image, "C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolov7/data/val/" + image)
for label in test_labels:
    os.rename("C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolo/data/labels/" + label, "C:/Users/DHarp/Documents/GitHub/traffic_sign_detection/yolov7/data/val/" + label)