from numba import jit, cuda
import os

@jit(target_backend='cuda')
def yolo_train():
    os.system('python yolo/train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --hyp yolo/data/hyp.scratch.custom.yaml --name yolov7-custom --weights yolo/yolov7.pt')

def yolo_test():
    os.system('python yolo/detect.py --weights rums/train/yolov7-custom4/weights/best.pt --conf 0.5 --img-size 640')

# yolo_train()

# yolo_test()