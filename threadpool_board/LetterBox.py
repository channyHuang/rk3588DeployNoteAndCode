import cv2
import numpy as np
import threading

# copy from coco_utils.py in rknn_model_zoo
class Simple_Letter_Box_Info():
    def __init__(self, shape, new_shape, w_ratio, h_ratio, dw, dh, pad_color) -> None:
        self.origin_shape = shape
        self.new_shape = new_shape
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.dw = dw 
        self.dh = dh
        self.pad_color = pad_color

class Simple_COCO_test_helper():
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(Simple_COCO_test_helper, "_instance"):
            with Simple_COCO_test_helper._instance_lock:
                if not hasattr(Simple_COCO_test_helper, "_instance"):
                    Simple_COCO_test_helper._instance = object.__new__(cls)
        return Simple_COCO_test_helper._instance
    
    def __init__(self, enable_letter_box = False) -> None:
        self.record_list = []
        self.enable_ltter_box = enable_letter_box
        if self.enable_ltter_box is True:
            self.letter_box_info_list = []
        else:
            self.letter_box_info_list = None

    def letter_box(self, im, new_shape, pad_color=(0,0,0), info_need=False):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border
        
        if self.enable_ltter_box is True:
            self.letter_box_info_list.append(Simple_Letter_Box_Info(shape, new_shape, ratio, ratio, dw, dh, pad_color))
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im
    
    def letter_box_new(self, im, new_shape, pad_color=(0,0,0), info_need=False):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border
        
        if self.enable_ltter_box is True:
            info = Simple_Letter_Box_Info(shape, new_shape, ratio, ratio, dw, dh, pad_color)
            return im, info
        else:
            return im
        
    def get_real_box(self, box, in_format='xyxy'):
        bbox = box
        if self.enable_ltter_box == True:
        # unletter_box result
            if in_format=='xyxy':
                bbox[:,0] -= self.letter_box_info_list[-1].dw
                bbox[:,0] /= self.letter_box_info_list[-1].w_ratio
                bbox[:,0] = np.clip(bbox[:,0], 0, self.letter_box_info_list[-1].origin_shape[1])

                bbox[:,1] -= self.letter_box_info_list[-1].dh
                bbox[:,1] /= self.letter_box_info_list[-1].h_ratio
                bbox[:,1] = np.clip(bbox[:,1], 0, self.letter_box_info_list[-1].origin_shape[0])

                bbox[:,2] -= self.letter_box_info_list[-1].dw
                bbox[:,2] /= self.letter_box_info_list[-1].w_ratio
                bbox[:,2] = np.clip(bbox[:,2], 0, self.letter_box_info_list[-1].origin_shape[1])

                bbox[:,3] -= self.letter_box_info_list[-1].dh
                bbox[:,3] /= self.letter_box_info_list[-1].h_ratio
                bbox[:,3] = np.clip(bbox[:,3], 0, self.letter_box_info_list[-1].origin_shape[0])
        return bbox

    def get_real_box_new(self, box, info, in_format='xyxy'):
        bbox = box
        if self.enable_ltter_box == True:
        # unletter_box result
            if in_format=='xyxy':
                bbox[:,0] -= info.dw
                bbox[:,0] /= info.w_ratio
                bbox[:,0] = np.clip(bbox[:,0], 0, info.origin_shape[1])

                bbox[:,1] -= info.dh
                bbox[:,1] /= info.h_ratio
                bbox[:,1] = np.clip(bbox[:,1], 0, info.origin_shape[0])

                bbox[:,2] -= info.dw
                bbox[:,2] /= info.w_ratio
                bbox[:,2] = np.clip(bbox[:,2], 0, info.origin_shape[1])

                bbox[:,3] -= info.dh
                bbox[:,3] /= info.h_ratio
                bbox[:,3] = np.clip(bbox[:,3], 0, info.origin_shape[0])
        return bbox

letterBox = Simple_COCO_test_helper(True)
