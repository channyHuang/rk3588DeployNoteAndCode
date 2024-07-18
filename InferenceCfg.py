# -*- coding: utf-8 -*-

# ===========================相机配置==========================
cameraUrl = 'rtsp://admin:@192.168.0.2'

streamreadtype = 1
campix_w = 1920
campix_h = 1080
# =======================结果发送网络配置======================
sendUrl = 'http://192.168.0.3:8081/images/'
# 不发送数据只在本地显示结果，用于测试
showInLocal = True

# ===========================识别配置=========================

# 用于目标识别的模型，以Inference文件夹内部开始为相对路径，list中如有多个模型将串联检测，目前多模型只支持相同输入分辨率
model = ['/../model/yolov8n.rknn'] 
# 模型输入分辨率，建议和模型相一致；当设置成比模型输入分辨率大时容易出现检测不到目标的现象
IMG_SIZE = (640, 640)
# 是否分割图像进行检测再拼接
crop_image = False
crop_size = (640, 640)
# 置信度设置
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
# 目标类型
CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")
