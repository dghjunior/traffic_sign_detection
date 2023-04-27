from numba import jit, cuda
import os

@jit(target_backend='cuda')
def yolo_train():
    os.system('python yolo/train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --hyp yolo/data/hyp.scratch.custom.yaml --name yolov7-custom --weights yolo/yolov7.pt')

def yolo_test():
    os.system('python yolo/detect.py --weights runs/train/yolov7-custom4/weights/best.pt --conf 0.5 --img-size 640')
    os.system('python detect.py --source archive/test/ --weights runs/train/yolo_road_det5/weights/best.pt --conf 0.25 --name yolov7-custom')

yolo_train()

# yolo_test()