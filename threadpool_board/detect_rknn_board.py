
from rknnlite.api import RKNNLite
import os
import cv2
import argparse
import numpy as np
import time

from LetterBox import letterBox
from Postprocess import postprocess
from RknnThreadPool import rknnThreadPool

CLASSES = ["1rail obstacles", "2steel rail", "3tank mines",
           "4wire netting", "5embankment", "6triangle heap",
           "7block wall", "8block trench", "9buoyant mine",
           "10hedgehogs", "11", "12", "13"]

class StSyncData:
    def __init__(self, detect_frame, detections, img_name):
        self.id = -1
        self.img_name = img_name
        self.detect_frame = detect_frame
        self.detections = detections
        self.boxes = None
        self.classes = None
        self.scores = None

def loadModel(model_name):
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_name)
    rknn.init_runtime(core_mask = RKNNLite.NPU_CORE_0_1_2)
    return rknn

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def readLabels(args, img_name):
    folder = args.label_folder

    name = img_name.rsplit('.', 1)[0]
    path = folder + name + '.txt'
    if not os.path.exists(path):
        return None, None
    f = open(folder + name + '.txt', 'r')
    lines = f.readlines()
    # centerx, centery, width, height
    detections = []
    # top, left, bottom, right
    pdetections = [] 
    for line in lines:
        det = list(line.strip('\n').split(' '))
        detn = []
        for i,d in enumerate(det):
            if i == 0:
                detn.append(int(d))
            elif i == 1 or i == 3:
                detn.append(int(float(d) * 1920))
            else:
                detn.append(int(float(d) * 1080))
        detections.append(detn)

        half = [detn[3] >> 1, detn[4] >> 1]
        top, left = detn[1] - half[0], detn[2] - half[1]
        right, bottom = detn[1] + half[0], detn[2] + half[1]
        pdetections.append([detn[0], top, left, right, bottom])
    f.close()

    return detections, pdetections

prob = 0
# boxes: left, top, right, bottom
def compareDetections(detections, boxes, classes, scores):
    global prob
    wrong, miss, correct = 0, 0, 0
    if boxes is None:
        miss = len(detections)
        return wrong, miss, correct
    if len(detections) == 0:
        wrong = len(boxes)
        return wrong, miss, correct
    numDetect = len(detections)
    detect_flag = np.array([0 for _ in range(numDetect)])
    wrong_flag = np.array([1 for _ in range(len(boxes))])
    for i, box in enumerate(boxes):
        found = False
        for j, det in enumerate(detections):
            if detect_flag[j] == 1 or det[0] != classes[i]:
                continue
            if np.linalg.norm(det[1:] - box) < 10:
                correct += 1
                detect_flag[j] = 1
                wrong_flag[i] = 0
                found = True
                break
        if not found:
            wrong += 1
    miss = numDetect - sum(detect_flag)
    for i, box in enumerate(boxes):
        if wrong_flag[i] == 1:
            if prob < scores[i]:
                prob = scores[i]
    return wrong, miss, correct

def inference_image_dir(args):
    global prob
    rknnThreadPool.startThreads(args.model_path, 3, postprocess.detect_frame_new)

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    numImages = len(img_list)

    total_target = 0
    detect_target = 0
    miss_target = 0
    wrong_target = 0
    looptime = time.time()
    frames = 0
    # run test
    for i in range(numImages):
        print('infer {}/{} - {} -> {}'.format(i+1, len(img_list), img_list[i], prob), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        detections, pdetections = readLabels(args, img_name)
        if detections is None:
            continue
        stSyncData = StSyncData(img_src, pdetections, img_name)
        rknnThreadPool.put(stSyncData)
        
        stSyncData_get, flag_pool_get = rknnThreadPool.get()
        if flag_pool_get == False:
            continue
        # compare
        img_src = stSyncData_get.detect_frame
        boxes = stSyncData_get.boxes
        classes = stSyncData_get.classes
        scores = stSyncData_get.scores
        pdetections = stSyncData_get.detections
        
        total_target += len(pdetections)
        wrong, miss, correct = compareDetections(pdetections, boxes, classes, scores)
        detect_target += correct
        miss_target += miss
        wrong_target += wrong
        # if wrong > 0 or miss > 0 or correct != len(detections):
        #     print('infer {}/{} wrong {} miss {} correct {} in {}'.format(i + 1, len(img_list), wrong, miss, correct, len(detections)))
        
        # draw
        for det in pdetections:
            top, left, right, bottom = det[1], det[2], det[3], det[4]
            cv2.rectangle(img_src, (top, left), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img_src, '{0}'.format(det[0] + 1), (right, bottom + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # if boxes is not None:
        #     img_p = img_src.copy()
        #     cv2.imshow("full post process result", img_p)
        #     cv2.waitKeyEx(1)
        frames += 1
        # if wrong == 0 and miss == 0:
        #     cv2.imwrite('./result/' + stSyncData_get.img_name, img_src)
        # else:
        #     cv2.imwrite('./result_check/' + stSyncData_get.img_name, img_src)
    print('\n')
    print('average fps: ', frames / (time.time() - looptime))
    print('xxxxxxxxxxxxxxxxxxxx wrong prob', prob)
    print('wrong/miss/correct {}({})/{}({})/{}({}) in {}'.format(
        wrong_target, wrong_target / total_target,
        miss_target, miss_target / total_target,
        detect_target, detect_target / total_target,
        total_target))
    # rknn.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, default='/../model/yolov8n.rknn', help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_folder', type=str, default='../model/rgb_dataset/images/', help='img folder path')
    parser.add_argument('--label_folder', type=str, default='../model/rgb_dataset/labels/', help='img folder path')
    parser.add_argument('--uri', type=str, default='rtsp://admin:admin@127.0.0.1', help='camera url or image path')
    args = parser.parse_args()

    inference_image_dir(args)
    # inference_video(args)
    # inference_image(args)
