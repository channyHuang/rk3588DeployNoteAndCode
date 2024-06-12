import os
import cv2

from rknn.api import RKNN

import numpy as np

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")



def xywh2xyxy(*box):
    ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, box[0] + box[2] // 2, box[1] + box[3] // 2]
    return ret

def get_inter(box1, box2):
    x1, y1, x2, y2 = xywh2xyxy(*box1)
    x3, y3, x4, y4 = xywh2xyxy(*box2)
    if x1 >= x4 or x2 <= x3:
        return 0
    if y1 >= y4 or y2 <= y3:
        return 0
    x_list = sorted([x1, x2, x3, x4])
    x_inter = x_list[2] - x_list[1]
    y_list = sorted([y1, y2, y3, y4])
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter

def get_iou(box1, box2):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    inter_area = get_inter(box1, box2)
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou

def nms(pred, conf_thres, iou_thres):
    box = pred[pred[..., 4] > conf_thres]
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))
    output_box = []
    output_class = []
    output_conf = []
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        temp = box[:, :6]
        for j in range(len(cls)):
            if cls[j] == clss:
                temp[j][5] = clss
                cls_box.append(temp[j][:6])
        cls_box = np.array(cls_box)
        sort_cls_box = sorted(cls_box, key = lambda x : -x[4])

        max_conf_box = sort_cls_box[0]
        output_box.append(max_conf_box)
        output_class.append(int(max_conf_box[4]))
        output_conf.append(max_conf_box[5])
        sort_cls_box = np.delete(sort_cls_box, 0, 0)

        while len(sort_cls_box) > 0:
            max_conf_box = output_box[-1]
            del_index = []
            for j in range(len(sort_cls_box)):
                current_box = sort_cls_box[j]
                iou = get_iou(max_conf_box, current_box)
                if iou > iou_thres:
                    del_index.append(j)
            sort_cls_box = np.delete(sort_cls_box, del_index, 0)
            if len(sort_cls_box) > 0:
                output_box.append(sort_cls_box[0])
                output_class.append(int(sort_cls_box[0][4]))
                output_conf.append(sort_cls_box[0][5])
                sort_cls_box = np.delete(sort_cls_box, 0, 0)
    return output_box, output_class, output_conf

def cod_trf(result, pre, after):
    res = np.array(result)
    x, y, w, h, conf, cls = res.transpose((1, 0))
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    scale = max(w_pre / w_after, h_pre / h_after)
    h_pre, w_pre = h_pre / scale, w_pre / scale
    x_move, y_move = abs(w_pre - w_after) // 2, abs(h_pre - h_after) // 2
    ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
    ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
    ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    return ret

def post_process(output):
    # (1, 13, 75600) -> box4[xywh], conf1, cls1, xxx
    pred = np.squeeze(output)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis = -1)
    pred = np.insert(pred, 4, pred_conf, axis = -1)

    box, classes, confs = nms(pred, 0.5, 0.5)
    return box, classes, confs

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def convert(model_path = 'yolov8n.onnx'):
    rknn = RKNN(verbose = False, verbose_file = 'build.log')
    rknn.config(mean_values=[128, 128, 128], std_values=[128, 128, 128], quant_img_RGB2BGR = True, quantized_method = 'channel', target_platform = 'rk3588')
    # export_outputs = ['/model.22/Mul_2_output_0', '/model.22/Split_output_1']
    ret = rknn.load_onnx(model = model_path) #, outputs = export_outputs)
    #ret = rknn.load_pytorch(model = '../model/v8n.pt', input_size_list = [[1, 3, 1920, 1920]])
    ret = rknn.build(do_quantization = False)
    ret = rknn.export_rknn(export_path = 'yolov8n.rknn', simplify = False)
    return rknn

if __name__ == '__main__':
    rknn = convert()
    
    imglist = ['bus.jpg']

    origin_img = cv2.imread('bus.jpg')
    origin_img_rz = cv2.resize(origin_img, IMG_SIZE)
    img_height, img_width = origin_img.shape[:2]
    img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (0, 1, 2))
    image_data = np.expand_dims(image_data, axis = 0).astype(np.float16)
    
    ret = rknn.load_rknn(path = 'yolov8n.rknn')
    ret = rknn.init_runtime(target = None, eval_mem = False, perf_debug = False)
    print(rknn.get_sdk_version())
    outputs = rknn.inference(inputs = [img])#, data_format = 'nchw')  
    output = np.concatenate((outputs[0], outputs[1]), axis = 1)
    boxes, classes, scores = post_process(output)
    xyxyboxes = []
    for b in boxes:
        xyxyboxes.append(xywh2xyxy(b[0], b[1], b[2], b[3]))
    draw(origin_img_rz, xyxyboxes, scores, classes)
    cv2.imshow('res', cv2.resize(origin_img_rz, (750, 750)))
    cv2.imwrite('test_res.jpg', cv2.resize(origin_img_rz, (750, 750)))
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    # only support in device, not in simulator
    #ret = rknn.eval_perf()
    #ret = rknn.eval_memory()
    rknn.release()