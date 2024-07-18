# -*- coding: utf-8 -*-

import cv2
import multiprocessing
import threading
import time
from queue import Queue

import InferenceCfg

# 读取视频流数据，返回[frame, time_stamp, cameraUrl]
class InferenceStream():
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(InferenceStream, "_instance"):
            with InferenceStream._instance_lock:
                if not hasattr(InferenceStream, "_instance"):
                    InferenceStream._instance = object.__new__(cls)
        return InferenceStream._instance
    
    def __init__(self):
        self.shape = (InferenceCfg.campix_w, InferenceCfg.campix_h)
        self.url = InferenceCfg.cameraUrl
        self.streamreadtype = InferenceCfg.streamreadtype
        self.stream_queue = Queue(maxsize = 2)
        self.put_stream_queue = Queue(maxsize = 2)
        self.stream_list = [None for i in range(5)]
        self.index = 0
        self.showInLocal = InferenceCfg.showInLocal
        self.log = None
        self.alive = multiprocessing.Manager().Value('b', False)

    def startProcess(self, log = None):
        if self.alive.value == True:
            return

        self.log = log
        thread_stream = threading.Thread(target=self.readStreamPython, args=())
        thread_stream.setDaemon(True)
        thread_stream.start()

        self.alive.value = True

    def stopProcess(self):
        if self.alive.value == False:
            return
        self.alive.value = False
        time.sleep(3)

    def getData(self):
        if self.stream_queue.empty():
            return None
        data = self.stream_queue.get()
        return data
    
    def putData(self, frame):
        if self.put_stream_queue.full():
            self.put_stream_queue.get()
        self.put_stream_queue.put(frame)

    def readStreamPython(self):
        for i in range(5):
            cap = cv2.VideoCapture(self.url)
            if self.log is not None:
                self.log.info('video fps (cv2.CAP_PROP_FPS): %s', cap.get(cv2.CAP_PROP_FPS))
                self.log.info('start Inference video reader: open %s', 'True' if cap.isOpened() else 'False')
            needresize = None
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if needresize is None:
                        needresize = (True if frame.shape[0] != self.shape[0] or frame.shape[1] != self.shape[1] else False)
                    if needresize == True:
                        frame = cv2.resize(frame, self.shape)
                    cur_time = time.time()
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    send_data = [frame, cur_time, self.url]
                    if self.stream_queue.full():
                        self.stream_queue.get()
                    self.stream_queue.put(send_data)
                else:
                    time.sleep(0.002)
                if not self.alive.value:
                    break
            cap.release()     
            if self.log is not None:
                self.log.info('stop Inference video reader')   

inferenceStream = InferenceStream()

if __name__ == '__main__':
    import queue
    inferenceStream.startProcess()
    spend_time = ''
    frames, loopTime, initTime = 0, time.time(), time.time()
    try:
        while (True):
            data = inferenceStream.getData()
            if data is None:
                time.sleep(0.02)
                continue
            frames += 1
            frame = cv2.cvtColor(data[0], cv2.COLOR_BGR2RGB)
            
            if frames >= 30:
                print("30帧平均帧率:\t", round(30 / (time.time() - loopTime), 2), "帧")
                spend_time = "30 frame average fps: {:.2f}".format(round(30 / (time.time() - loopTime), 2))
                loopTime = time.time()
                frames = 0
            cv2.putText(frame, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # print('origin image shape', frame.shape)
            cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
