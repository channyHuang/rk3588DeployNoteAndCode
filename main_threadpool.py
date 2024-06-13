import argparse
import copy
import cv2
import multiprocessing
import os
import time

from InferenceCfg import *
from RknnThreadPool import rknnThreadPool
from Postprocess import postprocess
from LetterBox import letterBox
from InferenceSendMessage import inferenceSendMessage

def inference_image(args, rknn, co_helper = letterBox, saved = False):
    start_time = time.time()
    origin_img = cv2.imread(args.img)
    origin_img = cv2.resize(origin_img, IMG_SIZE) 
    img, _ =  postprocess.detect_frame(rknn, origin_img, co_helper)
    spend_time = "{:.2f}  s".format((time.time() - start_time))
    cv2.putText(img, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    final_img = cv2.resize(img, (640, 640))
    cv2.imshow('res', final_img)
    if saved:
        cv2.imwrite('test_res.jpg', final_img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

def inference_image_dir(args):
    num_model = 3
    rknnThreadPool.startThreads(args.model, num_model, postprocess.detect_frame)

    img_list = os.listdir(args.folder)
    frame_num = 0
    loopTime = time.time()
    spend_time = 'frame average time: '
    for img_path in img_list:
        origin_img = cv2.imread(args.folder + '/' + img_path)
        rknnThreadPool.put(origin_img)
        frame_num += 1
        if frame_num <= 3:
            continue
        data_tuple, flag = rknnThreadPool.get()
        final_img, classes = data_tuple[0], data_tuple[1]

        if frame_num >= 33:
            spend_time = "{} frame average time: {:.2f} s".format(frame_num, (time.time() - loopTime) / frame_num)
            loopTime = time.time()
            frame_num = 3
        cv2.putText(final_img, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('res', final_img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

def inference_video(args):
    num_model = 3
    rknnThreadPool.startThreads(args.model, num_model, postprocess.detect_frame_multi)
    cap = cv2.VideoCapture(args.uri)
    print('video fps (cv2.CAP_PROP_FPS): ', cap.get(cv2.CAP_PROP_FPS))
    spend_time = "30 frame average fps: "

    send_queue = multiprocessing.Queue(maxsize = 5)
    inferenceSendMessage.startProcess(args.send_uri, send_queue)

    if (cap.isOpened()):
        for i in range(num_model):
            ret, org_img = cap.read()
            if not ret:
                exit(-1)
            origin_img = copy.deepcopy(org_img)
            rknnThreadPool.put(origin_img)

    frames, loopTime = 0, time.time()
    while (cap.isOpened()):
        ret, org_img = cap.read()
        if not ret:
            break
        origin_img = copy.deepcopy(org_img)
        rknnThreadPool.put(origin_img)
        data_tuple, flag = rknnThreadPool.get()
        final_img, classes = data_tuple[0], data_tuple[1]
        frames += 1
        if frames >= 30:
            spend_time = "30 frame average fps: {:.2f}".format(round(30 / (time.time() - loopTime), 2))
            loopTime = time.time()
            frames = 0
        
        cv2.putText(final_img, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if not args.send:
            cv2.imshow('res', final_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if send_queue.full():
                send_queue.get()
            send_queue.put([classes, final_img])
    cap.release()
    cv2.destroyAllWindows()
    rknnThreadPool.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='--model model_path --img image_path')
    parser.add_argument('--model', type = str, default = '../model/yolov8n.rknn', help = '.rknn')
    parser.add_argument('--input_type', type = int, default=0, help = 'type[0,1,2]: video or image dir or image')
    parser.add_argument('--uri', type=str, default='../model/video.avi', help='uri')
    parser.add_argument('--send_uri', type=str, default= 'http://127.0.0.1:8080/folder/fun')
    parser.add_argument('--folder', type=str, default='../model/images/')
    parser.add_argument('--img', type=str, default='../model/test.jpg')
    parser.add_argument('--send', action='store_true', default=False)
    args = parser.parse_args()

    if args.input_type == 0:
        inference_video(args)
    elif args.input_type == 1:
        inference_image_dir(args)
    else:
        inference_image(args)
