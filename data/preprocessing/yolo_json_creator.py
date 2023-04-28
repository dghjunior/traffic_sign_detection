import json
import os

# Path to the directory containing the YOLO annotation files
ann_dir = 'data/yolo_annotations'

# Path to the directory where you want to save the COCO-style JSON file
output_dir = 'yolo'

# Image size
img_width = 640
img_height = 640

# List of categories in your dataset
categories = [{'id': 0, 'name': 'trafficlight'}, {'id': 1, 'name': 'speedlimit'}, {'id': 2, 'name': 'crosswalk'}, {'id': 3, 'name': 'stop'}]

# Initialize the COCO-style dictionary
coco_dict = {
    "images": [],
    "annotations": [],
    "categories": categories
}

# Loop over each YOLO annotation file
for file_name in os.listdir(ann_dir):
    # Read the YOLO annotation file
    with open(os.path.join(ann_dir, file_name), 'r') as f:
        lines = f.readlines()
    
    # Get the image ID from the file name
    img_id = int(file_name.split('.')[0])
    
    # Add the image information to the dictionary
    image_info = {"id": img_id, "file_name": str(img_id) + ".jpg"}
    coco_dict["images"].append(image_info)
    
    # Loop over each bounding box in the YOLO annotation file
    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # Convert YOLO format to COCO format
        x_min = int((x - w/2) * img_width)
        y_min = int((y - h/2) * img_height)
        box_width = int(w * img_width)
        box_height = int(h * img_height)
        
        # Add the annotation information to the dictionary
        ann_info = {"id": len(coco_dict["annotations"]) + 1,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0}
        coco_dict["annotations"].append(ann_info)

# Save the COCO-style JSON file
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, 'instances_val2017.json'), 'w') as f:
    json.dump(coco_dict, f)
