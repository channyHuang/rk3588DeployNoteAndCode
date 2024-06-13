import asyncio
import cv2
import argparse
import time

from rknnlite.api import RKNNLite

from Postprocess import postprocess
from LetterBox import letterBox
from InferenceCfg import *

def initRKNN(self, model = '', id = -1):
    rknn = RKNNLite()
    ret = rknn.load_rknn(model)
    if ret != 0:
        print('load rknn model failed:', model)
        exit(ret)
    if id == 0:
        ret = rknn.init_runtime(core_mask = RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn.init_runtime(core_mask = RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn.init_runtime(core_mask = RKNNLite.NPU_CORE_2)
    else:
        ret = rknn.init_runtime(core_mask = RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print('Init runtime failed')
        exit(ret)
    return rknn

def initRKNNs(model_path, num_model = 3):
        rknn_list = [initRKNN(model_path, i % 3) for i in range(num_model)]
        return rknn_list 
    
def init(args):
    co_helper = letterBox
    rknns = initRKNNs(args.model, 3)
    return rknns, co_helper

async def image_inference(rknn, co_helper, origin_img):
    start_time = time.time()
    origin_img = cv2.resize(origin_img, IMG_SIZE) 
    img, _ = postprocess.detect_frame(rknn, origin_img, co_helper)
    final_img = cv2.resize(img, (750, 750))
    spend_time = "per frame time : {:.2f} s".format(time.time() - start_time)
    cv2.putText(final_img, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('res', final_img)
    cv2.waitKey(1)

async def inference_frame(origin_img, id, rknns, co_helper):
    await image_inference(rknns[id], co_helper, origin_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='--model model_path --uri video_path')
    parser.add_argument('--model', type = str, default = '../model/yolov8n.rknn', help = '.rknn')
    parser.add_argument('--uri', type=str, default='../model/video.avi', help='uri')
    args = parser.parse_args()

    rknns, co_helpter = init(args)
    loop = asyncio.get_event_loop()
    cap = cv2.VideoCapture(args.uri)
    frames = 0
    while cap.isOpened():
        ret, org_img = cap.read()
        if not ret:
            break
        frames += 1
        tasks = [asyncio.ensure_future(inference_frame(org_img, frames % 3, rknns, co_helpter))]
        loop.run_until_complete(asyncio.wait(tasks))
        if frames > 99999:
            frames = 0
