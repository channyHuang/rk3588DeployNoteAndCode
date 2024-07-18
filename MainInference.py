import cv2
import copy
import math
import multiprocessing
from queue import Queue
import threading
import time
import logging

import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from InferenceStream import inferenceStream
from RknnThreadPool_simulator import rknnThreadPool
from Postprocess import postprocess
from InferenceSendMessage import inferenceSendMessage
import InferenceCfg

class InferenceCrop:
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(InferenceCrop, "_instance"):
            with InferenceCrop._instance_lock:
                if not hasattr(InferenceCrop, "_instance"):
                    InferenceCrop._instance = object.__new__(cls)
        return InferenceCrop._instance
    
    def __init__(self):
        self.log=logging.getLogger("MainInference")
        self.model = InferenceCfg.model
        self.url = InferenceCfg.cameraUrl
        self.sendUrl = InferenceCfg.sendUrl
        self.crop_image = InferenceCfg.crop_image
        self.crop_size = InferenceCfg.crop_size
        self.send_queue = multiprocessing.Queue(maxsize = 2)
        self.thread_inference = None
        self.rknn_num_threads = 3
        self.alive = multiprocessing.Manager().Value('b', False)

    def startProcess(self):
        if self.alive.value == True:
            return
        self.alive.value = True

        # inference thread pool
        postprocess.setlog(self.log)
        rknnThreadPool.startThreads(self.model, self.rknn_num_threads, postprocess.detect_frame_new)
        inferenceStream.startProcess(self.log)
        inferenceSendMessage.startProcess(self.sendUrl, self.send_queue, self.log)
        
        if self.crop_image == True:
            self.thread_inference = threading.Thread(target=self.inference_crop, args=())
        else:
            self.thread_inference = threading.Thread(target=self.inference, args=())
        self.thread_inference.setDaemon(True)
        self.thread_inference.start()
       
    def stopProcess(self):
        if self.alive.value == False:
            return
        self.alive.value = False
        self.thread_inference.join()
        inferenceSendMessage.stopProcess()
        inferenceStream.stopProcess()
        rknnThreadPool.stopThreads()
        time.sleep(2)
    
    def getData(self):
        return inferenceSendMessage.getData()
    
    # crop_size = [height, width]
    @staticmethod
    def cal_offset(shape = [1080, 1920], crop_size = [1080, 1920]):
        height_num = math.ceil(shape[0] / crop_size[0])
        width_num = math.ceil(shape[1] / crop_size[1])
        height_pad, width_pad = 0, 0
        if height_num > 1:
            height_pad = (height_num * crop_size[0] - shape[0]) // (height_num - 1)
        if width_num > 1:
            width_pad = (width_num * crop_size[1] - shape[1]) // (width_num - 1)
        height_offset = 0
        width_offset = 0
        offset = []
        edge_flag = [False, False]
        while height_offset < shape[0]:
            edge_flag[1] = False
            if height_offset + crop_size[0] >= shape[0]:
                height_offset = shape[0] - crop_size[0]
                edge_flag[0] = True
            while width_offset < shape[1]:
                if width_offset + crop_size[1] > shape[1]:
                    width_offset = shape[1] - crop_size[1]
                    edge_flag[1] = True
                offset.append([height_offset, width_offset])
                if edge_flag[1]:
                    break
                width_offset += crop_size[1] - width_pad
            if edge_flag[0]:
                break
            height_offset += crop_size[0] - height_pad
            width_offset = 0
        return offset

    def inference(self):
        spend_time = "30 frame average fps:"
        frames, loopTime = 0, time.time()
        count = 0
        
        while self.alive.value:
            data = inferenceStream.getData()
            if data is None:
                time.sleep(0.002)
                continue

            origin_img = data[0]
            
            rknnThreadPool.put(origin_img, -1, origin_img)
            
            if count < self.rknn_num_threads:
                count += 1
                continue
            
            data_tuple, flag_pool_get = rknnThreadPool.get()
            if flag_pool_get == False:
                time.sleep(0.002)
                print('flag_pool_get failed')
                continue
            boxes, classes, scores, id, final_img = data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3], data_tuple[4]
            if boxes is not None:
                postprocess.draw(final_img, boxes, scores, classes)
            frames += 1
            if frames  >= 30:
                spend_time = "30 frame average fps: {:.2f}".format(round(30 / (time.time() - loopTime), 2))
                loopTime = time.time()
                frames = 0
            cv2.putText(final_img, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if self.send_queue.full():
                self.send_queue.get()
            self.send_queue.put([classes, final_img])
            
    def inference_crop(self):
        spend_time = "30 frame average fps:"
        frames, loopTime = 0, time.time()
        offset = None
        count = 0
        while self.alive.value:
            data = inferenceStream.getData()
            if data is None:
                time.sleep(0.002)
                continue
            
            origin_img = data[0]
            if offset is None:
                shape = origin_img.shape
                offset = self.cal_offset(shape, [self.crop_size[0], self.crop_size[1]])
                block = len(offset)
                offset_end = [[offset[i][0] + self.crop_size[0], offset[i][1] + self.crop_size[1]] for i in range(block)]
                print('clip info:', shape, block, offset)
            for i in range(block):
                img = origin_img[offset[i][0]:offset_end[i][0], offset[i][1]:offset_end[i][1]]
                source_img = copy.deepcopy(img)
                rknnThreadPool.put(source_img, i, origin_img)
            count += block
            if count <= 3:
                continue

            for i in range(block):
                data_tuple, flag_pool_get = rknnThreadPool.get()
                if flag_pool_get == False:
                    time.sleep(0.002)
                    i -= 1
                    continue
                boxes, classes, scores, id, final_img = data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3], data_tuple[4]
                if boxes is not None:
                    newboxes = [[offset[id][1], offset[id][0], offset[id][1], offset[id][0]] + box for box in boxes]
                    postprocess.draw(final_img, newboxes, scores, classes)
            frames += 1
            if frames  >= 30:
                spend_time = "30 frame average fps: {:.2f}".format(round(30 / (time.time() - loopTime), 2))
                loopTime = time.time()
                frames = 0
            cv2.putText(final_img, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if self.send_queue.full():
                self.send_queue.get()
            self.send_queue.put(final_img)

inferenceCrop = InferenceCrop()

if __name__ == '__main__':
    for i in range(10):
        inferenceCrop.startProcess()
        for i in range(320):
            try:
                data = inferenceCrop.getData()
                if data is not None:
                    cv2.imshow('res', data[1])
                    cv2.waitKey(1)
                time.sleep(0.1)
            except Exception as e:
                time.sleep(2)
                print('Error ', e)
        cv2.destroyAllWindows()
        inferenceCrop.stopProcess()