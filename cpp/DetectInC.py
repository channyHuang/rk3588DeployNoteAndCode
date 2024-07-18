# -*- coding:utf-8 -*-

import threading
import ctypes
import cv2
import numpy as np
import time
import multiprocessing

# 目标检测返回数据结构
class stDetectResult(ctypes.Structure):
    _fields_ = [('pFrame', ctypes.c_void_p), 
                ('nDetectNum', ctypes.c_int), 
                ('nWidth', ctypes.c_int), 
                ('nHeight', ctypes.c_int), 
                ('pClasses', ctypes.POINTER(ctypes.c_int)), 
                ('pBoxes', ctypes.POINTER(ctypes.c_int)), 
                ('pProb', ctypes.POINTER(ctypes.c_float))]

TypeCCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(stDetectResult), ctypes.c_void_p)
send_queue = multiprocessing.Queue(maxsize=10)

# Python中获取检测结果的回调函数
def DetectCallback(detectResult, user):
    if not detectResult:
        # print('Nothing')
        return
    else:
        num = detectResult.contents.nDetectNum
        if num > 0:
            classesPtrs = ctypes.cast(detectResult.contents.pClasses, ctypes.POINTER(ctypes.c_int))
            classesPtr = ctypes.cast(classesPtrs, ctypes.POINTER(ctypes.c_int * num))
            classes = np.frombuffer(classesPtr.contents, dtype=np.int32, count = num)

            boxesPtrs = ctypes.cast(detectResult.contents.pBoxes, ctypes.POINTER(ctypes.c_int))
            boxesPtr = ctypes.cast(boxesPtrs, ctypes.POINTER(ctypes.c_int * num * 4))
            boxes = np.frombuffer(boxesPtr.contents, dtype=np.int32, count = num * 4)
            
            probPtrs = ctypes.cast(detectResult.contents.pProb, ctypes.POINTER(ctypes.c_float))
            probPtr = ctypes.cast(probPtrs, ctypes.POINTER(ctypes.c_float * num))
            prob = np.frombuffer(probPtr.contents, dtype = np.float32, count = num)

        nWidth = detectResult.contents.nWidth
        nHeight = detectResult.contents.nHeight
        byteCount = nWidth * nHeight * 3
        imagePtr = ctypes.cast(detectResult.contents.pFrame, ctypes.POINTER(ctypes.c_uint8 * byteCount))
        picture = np.frombuffer(imagePtr.contents, dtype=np.ubyte, count=byteCount)
        picture = np.reshape(picture, (nHeight, nWidth, 3))
        
        while (send_queue.full()):
            time.sleep(0.002)
        send_queue.put([picture, classes, boxes, prob])
        
# 全局回调函数，非全局有时效性，多次调用非全局回调时会崩溃，暂时只能全局
pCallbackFunc = TypeCCallback(DetectCallback)

# Python调用C库目标检测
class DetectInC():
    # 单例
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(DetectInC, "_instance"):
            with DetectInC._instance_lock:
                if not hasattr(DetectInC, "_instance"):
                    DetectInC._instance = object.__new__(cls)
        return DetectInC._instance
    
    # 检测库，PC上交叉编译后push到板上
    def __init__(self) -> None:
        self.libpath = './libRknnDetect.so'
        self.lib = None

    # 加载检测库，调用初始化接口加载rknn模型
    def init(self, model):
        if self.lib is None:
            self.lib = ctypes.cdll.LoadLibrary(self.libpath)
        if self.lib is None:
            return False
        # 初始化
        self.lib.init.argtype = (ctypes.POINTER(ctypes.c_char_p))
        self.lib.init.restype = ctypes.c_bool
        res = self.lib.init(model.encode())
        if res == False:
            print('init failed', res)
        # 设置检测接口参数类型
        self.lib.detect.argtype = (ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int)
        self.lib.detect.restype = ctypes.POINTER(stDetectResult)

        self.lib.deinit.restype = ctypes.c_bool
        return res            

    def deinit(self):
        if self.lib is None:
            return False
        self.lib.deinit()

    # 单帧检测，返回[结果图，目标类别class，框box，置信度prob]
    def detect(self, frame):
        if self.lib is None:
            return None, None, None, None
        
        dataPtr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_char_p))
        detectResult = self.lib.detect(dataPtr, frame.shape[1], frame.shape[0])
        if not detectResult or detectResult.contents is None:
            print('Nothing')
            return None, None, None, None
        else:
            # 解析检测结果
            num = detectResult.contents.nDetectNum
            classes = []
            boxes = []
            prob = []
            if num > 0:
                # 目标类型
                classesPtr = ctypes.cast(detectResult.contents.pClasses, ctypes.POINTER(ctypes.c_int * num))
                classes = np.frombuffer(classesPtr.contents, dtype=np.int32, count = num)
                # 目标框
                boxesPtr = ctypes.cast(detectResult.contents.pBoxes, ctypes.POINTER(ctypes.c_int * num * 4))
                boxes_raw = np.frombuffer(boxesPtr.contents, dtype=np.int32, count = num * 4)
                boxes = []
                box = []
                for i, b in enumerate(boxes_raw):
                    box.append(b)
                    if i % 4 == 3:
                        boxes.append(box)
                        box = []
                # 目标置信度
                probPtr = ctypes.cast(detectResult.contents.pProb, ctypes.POINTER(ctypes.c_float * num))
                prob = np.frombuffer(probPtr.contents, dtype = np.float32, count = num)
            # 带框图像
            nWidth = detectResult.contents.nWidth
            nHeight = detectResult.contents.nHeight
            byteCount = nWidth * nHeight * 3
            imagePtr = ctypes.cast(detectResult.contents.pFrame, ctypes.POINTER(ctypes.c_uint8 * byteCount))
            picture = np.frombuffer(imagePtr.contents, dtype=np.ubyte, count=byteCount)
            picture = np.reshape(picture, (nHeight, nWidth, 3))
            return picture, classes, boxes, prob

    def printProfile(self):
        if self.lib is None:
            return
        self.lib.printProfile()

    # 设置回调
    def setCallback(self, set = False):
        if self.lib is None:
            return None
        if set:        
            self.lib.setCallback(pCallbackFunc, None)
        else:
            self.lib.setCallback(None, None)

    # 多线程检测
    def detectAsync(self, frame):
        if self.lib is None:
            return None
        dataPtr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_char_p))
        self.lib.detectAsync(dataPtr, frame.shape[1], frame.shape[0])

detectInC = DetectInC()

if __name__ == '__main__':
    model_path = './model/yolov8n.rknn'
    uri = './model/1920_test.mp4'

    detectInC.init(model_path)

    cap = cv2.VideoCapture(uri)
    loopTime = time.time()
    framenum = 0
    spend_time = "30 frame average fps: "
    while cap.isOpened():
        ret, image = cap.read()
        if ret == False:
            break
        res_image, classes, boxes, probs = detectInC.detect(image)
        framenum += 1
        if res_image is not None:
            if framenum >= 30:
                spend_time = "30 frame average fps: {:.2f}".format(round(30 / (time.time() - loopTime), 2))
                loopTime = time.time()
                framenum = 0
            cv2.putText(res_image, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('res', res_image)
            cv2.waitKey(1)
    detectInC.printProfile()
    detectInC.deinit()
    cv2.destroyAllWindows()

