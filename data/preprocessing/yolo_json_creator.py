import json
import os

# Path to the directory containing the YOLO annotation files
ann_dir = 'data/yolo_annotations'

# Path to the directory where you want to save the COCO-style JSON file
output_dir = 'yolo'

# Define the number of classes in your dataset
num_classes = 4  # Replace with the number of classes in your dataset

# Create a dictionary to store the dataset
dataset = {}

# Create a list to store the images
images = []

img_dir = 'data/images/'

# Loop over the annotation files
for ann_file in os.listdir(ann_dir):
    # Get the image ID from the file name
    file_name = os.path.splitext(ann_file)[0]
    img_id = int(file_name.split('.')[0][4:])

    # Get the image width and height
    img_file = img_dir + file_name + '.jpg'
    img_width, img_height = 640, 640 

    # Create a dictionary to store the image information
    image = {
        'id': img_id,
        'width': img_width,
        'height': img_height,
        'file_name': img_file
    }

    # Append the image dictionary to the list of images
    images.append(image)

# Add the images to the dataset dictionary
dataset['images'] = images

# Add the number of classes to the dataset dictionary
dataset['nc'] = num_classes

# Save the dataset as a JSON file
with open('yolo/road_sign_detection.json', 'w') as f:
    json.dump(dataset, f)