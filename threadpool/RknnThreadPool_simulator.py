
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
        self.num_model = 1
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
                # model_path = os.path.dirname(os.path.abspath(__file__)) + model_path
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
            # models = os.path.dirname(os.path.abspath(__file__)) + models
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

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def transforLabelStr(labels):
    labels = labels.split(' ')
    labels[0] = int(labels[0])
    labels[1] = float(labels[1])
    labels[2] = float(labels[2])
    labels[3] = float(labels[3])
    labels[4] = float(labels[4])
    # labels[5] = float(labels[5])
    return labels

def detectImageFolder(model_name = 'yolov8n.onnx', 
                      image_folder = './dataset/images/'):
    from Postprocess import postprocess
    import cv2
    numThread = 1
    rknnThreadPool.startThreads(model_name, numThread, postprocess.detect_frame)
    file_list = sorted(os.listdir(image_folder))

    total = 0
    correct = 0
    wrong = 0
    miss = 0

    for path in file_list:
        if img_check(path):
            frame = cv2.imread(image_folder + '/' + path)

            label_name = './dataset/labels/' + path.rsplit('.', 1)[0] + '.txt'
            if not os.path.exists(label_name):
                continue

            infos = []
            with open(label_name, "r") as f:
                ground_trues = f.readlines()
                for ground_true in ground_trues:
                    trues = transforLabelStr(ground_true)
                    infos.append(trues)
                f.close()

            rknnThreadPool.put(frame)
            data, suc = rknnThreadPool.get()
            if suc == False:
                continue

            for info in infos:
                box = info[1:]
                box[0] *= 1920
                box[1] *= 1080
                box[2] *= 1920
                box[3] *= 1080
                box[2] += box[0]
                box[3] += box[1]

                postprocess.draw(data[0], [box], [1.0], [info[0]], color = (0, 255, 0))

            cv2.imshow('result', data[0])
            cv2.waitKey(1)

            total += len(infos)
            detectCorrect = 0
            if data[1] is not None:
                classes = data[1]
                boxes = data[2]
                probs = data[3]
                detectNum = len(boxes)

                for (cls, box, prob) in zip(classes, boxes, probs):
                    wrongFlag = True
                    centerx = (box[0] + box[2]) / 2.0
                    centery = (box[1] + box[3]) / 2.0
                    for info in infos:
                        if info[0] != cls:
                            continue
                        ctnx = info[1] * 1920
                        ctny = info[2] * 1080
                        if (centerx - ctnx) < 15 and (centery - ctny) < 15:
                            correct += 1
                            detectCorrect += 1
                            wrongFlag = False
                            break
                    if wrongFlag == True:
                        wrong += 1
                miss += (len(infos) - detectCorrect)
                correct += detectCorrect
    print('detect result: ', total, correct, miss, wrong)
    rknnThreadPool.stopThreads()

def detectVideo(model_name = 'yolov8n.onnx', 
                video_name = '1920_test.mp4'):
    from Postprocess import postprocess
    import cv2
    numThread = 1
    rknnThreadPool.startThreads(model_name, numThread, postprocess.detect_frame)
    cap = cv2.VideoCapture(video_name)
    print('open video ', cap.isOpened())
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            rknnThreadPool.put(frame)

            if count < numThread - 1:
                count += 1
                continue
            data, suc = rknnThreadPool.get()
            if suc == False:
                continue
            cv2.imshow('result', data[0])
            cv2.waitKey(1)
    rknnThreadPool.stopThreads()

if __name__ == '__main__':
    detectImageFolder()
    
