from numba import jit, cuda
import os

@jit(target_backend='cuda')
def yolo_train():
    os.system('python yolov7/train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --hyp yolov7/data/hyp.scratch.custom.yaml --name yolov7-custom --weights yolov7/yolov7.pt')

def yolo_test():
    # os.system('python yolov7/detect.py --weights yolov7/best.pt --img-size 640 --conf 0.5 --source yolov7/data/val/ --view-img --no-trace')
    # os.system('python yolov7/test.py --weights yolov7/best.pt --task val')
    os.system('python yolov7/test.py --data yolo/road_sign_detection.json --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights yolov7/best.pt --name yolov7-custom')

# yolo_train()

yolo_test()