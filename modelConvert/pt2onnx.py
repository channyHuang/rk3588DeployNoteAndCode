# -*- coding: utf-8 -*-

# Using ultralytics to convert pt model 
from ultralytics import YOLO

def pt2onnx(path = '../data/yolov8n.pt'):
    model = YOLO(path)
    res = model.export(format="onnx", opset = 19, simplify = True)  # export the model to ONNX format

if __name__ == '__main__':
    pt2onnx()