import argparse

from ultralytics import YOLO

def pt2onnx(path = '../model/yolov8n.pt', sim = True):
    model = YOLO(path)
    res = model.export(format="onnx", simplify = sim, opset = 19) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplify", action='store_true', default=False, help="simplify")
    parser.add_argument("--opset", type=int, default=19, help="opset")
    parser.add_argument("--model", type=str, default="../model/yolov8n.pt", help="Input your PT model.")
    args = parser.parse_args()
    
    pt2onnx(args.model, args.simplify)
