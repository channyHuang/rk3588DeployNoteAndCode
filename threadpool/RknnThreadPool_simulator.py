
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import os

from rknn.api import RKNN

from LetterBox import letterBox

# thread pool
class RknnThreadPool():
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(RknnThreadPool, "_instance"):
            with RknnThreadPool._instance_lock:
                if not hasattr(RknnThreadPool, "_instance"):
                    RknnThreadPool._instance = object.__new__(cls)
        return RknnThreadPool._instance
    
    def __init__(self):
        self.num_model = 3
        self.queue = Queue()
        self.num = 0
        self.co_helper = letterBox #Simple_COCO_test_helper(enable_letter_box=True)
        self.rknnPool = None
        self.pool = None
        self.func = None
        self.alive = False

    def startThreads(self, model_path, num_model, func):
        if self.alive == True:
            return
        self.alive = True
        self.num_model = num_model
        self.rknnPool = self.initRKNNs(model_path, num_model)
        self.pool = ThreadPoolExecutor(max_workers = num_model)
        self.func = func

    def stopThreads(self):
        if self.alive == False:
            return
        self.alive = False
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            if isinstance(rknn_lite, list):
                for rknn in rknn_lite:
                    rknn.release()
            else:
                rknn_lite.release()
    
    def initRKNNs(self, model_path, num_model = 3):
        rknn_list = [self.initRKNN(model_path, i % 3) for i in range(num_model)]
        return rknn_list 

    def initRKNN(self, models = '', id = -1):
        if isinstance(models, list): 
            rknns = []
            for model_path in models:
                model_path = os.path.dirname(os.path.abspath(__file__)) + model_path
                model_name = model_path.rsplit('.', 1)[0] + '.onnx'
                rknn = RKNN()
                rknn.config(mean_values=[128, 128, 128], std_values=[128, 128, 128], 
                            quant_img_RGB2BGR = True, 
                            quantized_dtype = 'asymmetric_quantized-8', quantized_method = 'layer', quantized_algorithm = 'mmse', optimization_level = 3,
                            target_platform = 'rk3588',
                            model_pruning = True)
                ret = rknn.load_onnx(model_name)
                if ret != 0:
                    print('load rknn model failed:', model_path)
                    exit(ret)
                rknn.build(do_quantization=False)
                if id == 0:
                    ret = rknn.init_runtime()
                elif id == 1:
                    ret = rknn.init_runtime()
                elif id == 2:
                    ret = rknn.init_runtime()
                else:
                    ret = rknn.init_runtime()
                if ret != 0:
                    print('Init runtime failed')
                    exit(ret)
                rknns.append(rknn)
            return rknns
        else:
            rknn = RKNN()
            models = os.path.dirname(os.path.abspath(__file__)) + models
            model_name = models.rsplit('.', 1)[0] + '.onnx'
            rknn.config(mean_values=[128, 128, 128], std_values=[128, 128, 128], 
                        quant_img_RGB2BGR = True, 
                        quantized_dtype = 'asymmetric_quantized-8', quantized_method = 'layer', quantized_algorithm = 'mmse', optimization_level = 3,
                        target_platform = 'rk3588',
                        model_pruning = True)
            ret = rknn.load_onnx(model_name)
            if ret != 0:
                print('load rknn model failed:', model_path)
                exit(ret)
            rknn.build(do_quantization=False)
            if id == 0:
                ret = rknn.init_runtime()
            elif id == 1:
                ret = rknn.init_runtime()
            elif id == 2:
                ret = rknn.init_runtime()
            else:
                ret = rknn.init_runtime()
            if ret != 0:
                print('Init runtime failed')
                exit(ret)
            return rknn
    
    def put(self, frame, id = -1, full_frame = None):
        if id < 0 or id >= self.num_model:
            index = self.num % self.num_model
        else:
            index = id
        self.queue.put(self.pool.submit(self.func, self.rknnPool[index], frame, letterBox, id))
        self.num += 1
        if self.num > 99999:
            self.num = 1

    def put_struct(self, stSyncData):
        if stSyncData.id < 0 or stSyncData.id >= self.num_model:
            stSyncData.id = self.num % self.num_model
        self.queue.put(self.pool.submit(self.func, self.rknnPool[stSyncData.id], stSyncData))
        self.num += 1
        if self.num > 99999:
            self.num = 1
    
    def get(self):
        if self.queue.empty():
            return None, False
        temp = []
        temp.append(self.queue.get())
        for data in as_completed(temp):
            return data.result(), True
    
rknnThreadPool = RknnThreadPool()

if __name__ == '__main__':
    from Postprocess import postprocess
    import cv2
    numThread = 3
    rknnThreadPool.startThreads('/../data/yolov8n.onnx', numThread, postprocess.detect_frame)
    cap = cv2.VideoCapture('../data/video.avi')
    print('open video ', cap.isOpened())
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            rknnThreadPool.put(frame)

            if count < numThread:
                count += 1
                continue
            data, suc = rknnThreadPool.get()
            if suc == False:
                continue
            cv2.imshow('result', data[0])
            cv2.waitKey(1)
    rknnThreadPool.stopThreads()
    
