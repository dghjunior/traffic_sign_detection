from numba import jit, cuda
import os

@jit(target_backend='cuda')
def yolo():
    os.system('python yolo/train.py --workers 1 --device 0 --batch-size 16 --epochs 100 --img 640 640 --hyp yolo/data/hyp.scratch.custom.yaml --name yolov7-custom --weights yolo/yolov7.pt')

yolo()