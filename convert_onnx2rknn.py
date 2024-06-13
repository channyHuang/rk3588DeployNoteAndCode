import argparse
import sys
from rknn.api import RKNN

DATASET_PATH = '../model/dataset.txt'

def onnx2rknn(args):
    model_name = args.model_path.rsplit('.', 1)[0]

    rknn = RKNN(verbose = False, verbose_file = 'build.log')
    print('--> Config model')
    rknn.config(mean_values=[128, 128, 128], std_values=[128, 128, 128], 
                quant_img_RGB2BGR = False, 
                quantized_dtype = 'asymmetric_quantized-8', 
                quantized_method = 'channel', 
                quantized_algorithm = 'normal', 
                optimization_level = 3,
                target_platform = args.target,
                model_pruning = True)
    
    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model = args.model_path)
    #ret = rknn.load_pytorch(model = '../model/v8n.pt', input_size_list = [[1, 3, 1920, 1920]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization = args.quant, dataset=DATASET_PATH, rknn_batch_size = None)
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(export_path = model_name + '.rknn', 
                           simplify = True, 
                           opset_version = 19)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    #ret = rknn.export_encrypted_rknn_model(input_model = 'yolov8n.rknn', crypt_level = 3)

    rknn.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, default='../model/yolov8n.onnx', help='model path')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--quant', type=bool, default=True, help='do_quant')
    args = parser.parse_args()

    rknn = onnx2rknn(args)
