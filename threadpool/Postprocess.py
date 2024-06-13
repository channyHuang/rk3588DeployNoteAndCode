import cv2
import numpy as np
import scipy
import threading
import time

IMG_SIZE = [640, 640]
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")
from LetterBox import letterBox

class Postprocess():
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(Postprocess, "_instance"):
            with Postprocess._instance_lock:
                if not hasattr(Postprocess, "_instance"):
                    Postprocess._instance = object.__new__(cls)
        return Postprocess._instance
    
    def __init__(self):
        self.log = None

    def setlog(self, log):
        self.log = log

    @staticmethod
    def filter_boxes(boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    @staticmethod
    def nms_boxes(boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size <= 1:
                break

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1
            diff = (areas[i] + areas[order[1:]] - inter)
            ovr = np.divide(inter , diff, where=diff != 0)
            if len(np.where(ovr <= NMS_THRESH)) <= 0:
                break
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    @staticmethod
    def dfl(position):
        # Distribution Focal Loss (DFL)
        x = position
        n, c, h, w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n,p_num,mc,h,w)
        y = scipy.special.softmax(y, 2)
        acc_metrix = np.arange(mc).reshape(1,1,mc,1,1)
        y = (y*acc_metrix).sum(2)
        return y
        '''
        import torch
        x = torch.tensor(position)
        n,c,h,w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n,p_num,mc,h,w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
        y = (y*acc_metrix).sum(2)
        return y.numpy()
        '''

    @staticmethod
    def box_process(position):
        inf_index = np.isinf(position)
        if True in inf_index:
            return None
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

        position = Postprocess.dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
        return xyxy

    @staticmethod
    def post_process(input_data):
        boxes, scores, classes_conf = [], [], []
        classes = []
        defualt_branch=3
        pair_per_branch = len(input_data)//defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            xyxy = Postprocess.box_process(input_data[pair_per_branch*i])
            if xyxy is None:
                return None, None, None#boxes, classes, scores
            boxes.append(xyxy)
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = Postprocess.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = Postprocess.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    @staticmethod
    def draw(image, boxes, scores, classes, color = (255, 0, 0)):
        for box, score, cl in zip(boxes, scores, classes):
            non_index = np.isnan(box)
            if True in non_index:
                continue
            top, left, right, bottom = [int(_b) for _b in box]
            # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), color, 2)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    @staticmethod
    def detect_frame(rknn, img_src, co_helper = letterBox, id = -1, drawed = True):
        img = letterBox.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        
        image_data = np.array(img) #/ 255.0
        image_data = np.transpose(image_data, (0, 1, 2))
        image_data = np.expand_dims(image_data, axis = 0).astype(np.float16)

        outputs = rknn.inference(inputs = [image_data])#[img])#, data_format = 'nchw')  
        boxes, classes, scores = Postprocess.post_process(outputs)
        if drawed:
            if boxes is not None:
                Postprocess.draw(img_src, letterBox.get_real_box(boxes), scores, classes)
            return img_src, classes, boxes, scores
        return img_src, classes, boxes, scores

    # @staticmethod
    # stSyncData: detect_frame, id, origin_img, ...
    def detect_frame_new(self, rknn_list, stSyncData):
        img_src = stSyncData.detect_frame
        img_full = stSyncData.origin_img

        img, info = letterBox.letter_box_new(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        
        image_data = np.array(img) #/ 255.0
        image_data = np.transpose(image_data, (0, 1, 2))
        image_data = np.expand_dims(image_data, axis = 0).astype(np.float16)

        boxes_all = None
        classes_all = None
        scores_all = None
        if isinstance(rknn_list, list):
            for rknn in rknn_list:
                # starttime = time.time()
                outputs = rknn.inference(inputs = [image_data]) #[img])#, data_format = 'nchw')  
                # if self.log is not None:
                #     self.log.info('sdk api rknn.inference spend (s): %s', time.time() - starttime)
                # posttime = time.time()
                boxes, classes, scores = Postprocess.post_process(outputs)
                # if self.log is not None:
                #     self.log.info('post_process spend (s): %s', time.time() - posttime)
                
                if (boxes is None) or (len(boxes) <= 0):
                    continue
                boxes_fix = letterBox.get_real_box_new(boxes, info)
                if boxes_all is None:
                    boxes_all = boxes_fix
                    classes_all = classes
                    scores_all = scores
                else:
                    np.append(classes_all, classes)
                    np.append(boxes_all, boxes_fix)
                    np.append(scores_all, scores)
        else:
            outputs = rknn_list.inference(inputs = [image_data])#[img])#, data_format = 'nchw')
            boxes, classes, scores = Postprocess.post_process(outputs)
            if (boxes is not None) and (len(boxes) > 0):
                boxes = letterBox.get_real_box_new(boxes, info)
                boxes_all = boxes
                classes_all = classes
                scores_all = scores
        if boxes_all is not None:
            Postprocess.draw(img_full, boxes_all, scores_all, classes_all)
        stSyncData.boxes = boxes_all
        stSyncData.classes = classes_all
        stSyncData.scores = scores_all
        stSyncData.origin_img = img_full
        return stSyncData

postprocess = Postprocess()
