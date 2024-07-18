import json
import multiprocessing
import requests
import threading
import time

from ImageProcess import imageProcess
import InferenceCfg

class InferenceSendMessage():
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(InferenceSendMessage, "_instance"):
            with InferenceSendMessage._instance_lock:
                if not hasattr(InferenceSendMessage, "_instance"):
                    InferenceSendMessage._instance = object.__new__(cls)
        return InferenceSendMessage._instance
    
    def __init__(self):
        self.log = None
        self.showInLocal = InferenceCfg.showInLocal
        self.url = None
        self.data_queue = multiprocessing.Queue(maxsize=5)
        self.alive = multiprocessing.Manager().Value('b', False)

    def startProcess(self, uri = 'http://127.0.0.1:8080/folder/fun', send_queue = multiprocessing.Queue(maxsize = 5), log = None):
        if self.alive.value == True:
            return
        self.url = uri
        self.log = log
        self.alive.value = True
        
        process_send = multiprocessing.Process(target = self.sendProcess, args = (send_queue, ))
        processes = [process_send]
        [setattr(process, "daemon", True) for process in processes]
        [process.start() for process in processes]

    def stopProcess(self):
        if self.alive.value == False:
            return
        self.alive.value = False

    def getData(self):
        if self.data_queue.empty():
            return None
        return self.data_queue.get()

    def sendProcess(self, send_queue):
        while self.alive.value:
            if send_queue.empty():
                time.sleep(0.002)
                continue
            send_data = send_queue.get()
            classes, dst_image = send_data[0], send_data[1]
            if self.showInLocal:
                if self.data_queue.full():
                    self.data_queue.get()
                self.data_queue.put([classes, dst_image])
            else:
                self.sendMessage(classes, dst_image)

    def sendMessage(self, classes, dst_image):
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        imgdata = imageProcess.image2bytes(dst_image)
        infodata = {'timestamp': localtime}
        senddata1 = json.dumps(infodata).encode()
        len_senddata1 = len(senddata1)
        head = int(len_senddata1).to_bytes(length=2, byteorder='little', signed=False)
        content = head + senddata1 + imgdata

        res = requests.post(url = self.url, data=content)

inferenceSendMessage = InferenceSendMessage()
