import threading
import os
from ctypes import *
import ctypes
import cv2
import numpy as np

class stDetectResult(Structure):
    _fields_ = [('pFrame', c_void_p), ('nDetectNum', c_int), ('classes', ctypes.POINTER(ctypes.c_int)), ('boxes', ctypes.POINTER(ctypes.c_int)), ('prob', ctypes.POINTER(ctypes.c_float))]

class CallCLibrary():
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(CallCLibrary, "_instance"):
            with CallCLibrary._instance_lock:
                if not hasattr(CallCLibrary, "_instance"):
                    CallCLibrary._instance = object.__new__(cls)
        return CallCLibrary._instance
    
    def __init__(self) -> None:
        self.libpath = './librknn_yolov8_demo.so'
        self.lib = None

    def init(self, model):
        if self.lib is None:
            self.lib = cdll.LoadLibrary(self.libpath)
        if self.lib is None:
            return False
        
        self.lib.Init.argtype = (POINTER(ctypes.c_char_p))
        self.lib.Init.restype = ctypes.c_bool

        res = self.lib.Init(model.encode())
        print('init ', res)
        return res            

    def detect(self, frame):
        if self.lib is None:
            return None
        
        byteCount = frame.shape[0] * frame.shape[1] * 3
        dataPtr = frame.ctypes.data_as(POINTER(ctypes.c_char_p))

        self.lib.Detect.argtype = (POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int)
        self.lib.Detect.restype = POINTER(stDetectResult)

        detectResult = self.lib.Detect(dataPtr, frame.shape[1], frame.shape[0])
        if not detectResult:
            print('Nothing')
        else:
            num = detectResult.contents.nDetectNum

            classesPtrs = cast(detectResult.contents.classes, ctypes.POINTER(ctypes.c_int))
            classesPtr = cast(classesPtrs, POINTER(c_int * num))
            classes = np.frombuffer(classesPtr.contents, dtype=np.int32, count = num)

            boxesPtrs = cast(detectResult.contents.boxes, ctypes.POINTER(ctypes.c_int))
            boxesPtr = cast(boxesPtrs, POINTER(c_int * num * 4))
            boxes = np.frombuffer(boxesPtr.contents, dtype=np.int32, count = num * 4)
            
            probPtrs = cast(detectResult.contents.prob, ctypes.POINTER(ctypes.c_float))
            probPtr = cast(probPtrs, POINTER(c_float * num))
            prob = np.frombuffer(probPtr.contents, dtype = np.float32, count = num)
            print('result ', num, prob, classes, boxes)

            imagePtr = cast(detectResult.contents.pFrame, POINTER(c_void_p))
            picturePtr = cast(imagePtr, POINTER(c_uint8 * byteCount))
            picture = np.frombuffer(picturePtr.contents, dtype=np.ubyte, count=byteCount)
            picture = np.reshape(picture, (frame.shape[1], frame.shape[0], frame.shape[2]))
            cv2.imshow('res', picture)
            cv2.waitKeyEx(1000)
        
callCLibrary = CallCLibrary()

if __name__ == '__main__':
    image = cv2.imread('./model/bus.jpg')
    callCLibrary.init('./model/yolov8n.rknn')
    callCLibrary.detect(image)
