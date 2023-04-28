import os
import json

# Specify the directories containing the annotations
train_dir = "yolov7/data/train/"
val_dir = "yolov7/data/val/"

# Define the categories in your dataset
# ["trafficlight", "speedlimit", "crosswalk", "stop"]
categories = [{"id": 0, "name": "trafficlight"}, {"id": 1, "name": "speedlimit"}, {"id": 2, "name": "crosswalk"}, {"id": 3, "name": "stop"}]

# Define the dictionary that will contain the annotations for both the training and validation sets
sets = {"train": [], "val": []}

img_width = 640
img_height = 640

# Loop over the training set annotations directory and read the txt files
for filename in os.listdir(train_dir):
    if filename.endswith(".txt"):
        # Create an entry for this image in the training set
        img_id = int(filename.split(".")[0][4:])
        img_info = {"id": img_id, "file_name": f"{img_id}.jpg"}
        with open(os.path.join(train_dir, filename), "r") as f:
            bbox = f.readline().split()
            ann_info = {"image_id": img_id, "category_id": bbox[0]}
            x, y, w, h = map(float, bbox[1:])
            w, h = round(w * img_width), round(h * img_height)
            x, y = round(x * img_width - w / 2), round(y * img_height - h / 2)
            ann_info["bbox"] = [x, y, w, h]
        sets["train"].append(ann_info)

# Loop over the validation set annotations directory and read the txt files
for filename in os.listdir(val_dir):
    if filename.endswith(".txt"):
        # Create an entry for this image in the validation set
        img_id = int(filename.split(".")[0][4:])
        img_info = {"id": img_id, "file_name": f"{img_id}.jpg"}
        with open(os.path.join(val_dir, filename), "r") as f:
            bbox = f.readline().split()
            ann_info = {"image_id": img_id, "category_id": bbox[0]}
            x, y, w, h = map(float, bbox[1:])
            w, h = round(w * img_width), round(h * img_height)
            x, y = round(x * img_width - w / 2), round(y * img_height - h / 2)
            ann_info["bbox"] = [x, y, w, h]
        sets["val"].append(ann_info)

# Create the json file with the annotations
json_data = {"nc": 4, "images": [], "annotations": [], "categories": categories}
for img_set, img_anns in sets.items():
    for img_ann in img_anns:
        json_data["annotations"].append(img_ann)
        json_data["images"].append({"id": img_ann["image_id"], "file_name": f"{img_ann['image_id']}.jpg"})
    json_data["sets"] = [{"images": [img_ann["image_id"] for img_ann in img_anns], "set_name": img_set}]
with open("yolo/annotations.json", "w") as f:
    json.dump(json_data, f)
