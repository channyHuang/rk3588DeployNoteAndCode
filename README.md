# rk3588DeployNoteAndCode
Deploy in rk3588 with yolo model notes and codes

记录把Yolov8n预训练的模型push到rk3588的板子上进行目标检测的整个过程，笔记和代码。

[rk3588笔记](https://channyhuang.github.io/linux/2024/04/08/Yolov8n_Opt_In_rk3588_(1))
[所有笔记](https://channyhuang.github.io)

[代码](https://github.com/channyHuang/rk3588DeployNoteAndCode)

rk3588部署笔记

# 环境篇
## 官网
1. [YOLOv8](https://github.com/ultralytics/ultralytics) 
参考，main.py可用于验证onnx的检测结果
2. [ultralytics_yolov8](https://github.com/airockchip/ultralytics_yolov8.git) 
rknn的pt到onnx的转换代码
3. [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) 部署到rk3588的样例

## pt转换成rknn
1. 使用`ultralytics_yolov8`把pt转换成onnx，其中`cfg/default.yaml`中修改模型路径、输入尺寸等参数
```sh
export PYTHONPATH=./
python ./ultralytics/engine/exporter.py
```

在`cfg/default.yaml`中修改参数，主要修改pt模型路径`model`和尺寸`imgsz`
```yaml
......
model: /home/channy/Desktop/yolov8n.pt
......
imgsz: 1920 
......
```

2. 使用`rknn_model_zoo`把onnx转换成rknn
```sh
./rknn_model_zoo/examples/yolov8/python
```
执行该路径下的`convert.py`进行转换，其中参数为:onnx模型、`rk3588`、int化（选`i8`或`fp`）、输出路径，具体样例为
```sh
python3 convert.py ~/Desktop/yolov8n.onnx rk3588 i8 ~/Desktop/yolov8n.rknn
```

最终会在输出目录下生成`rknn`文件

## C++检测库
### 基本库文件
| 文件名称 | 功能 |
|:---:|:---:|
| libHGRknnDetect.so | 目标检测主库 |
| librga.so | 依赖库，RGA图像加速引擎库 |
| librknnrt.so | 依赖库，板端runtime库 | 
| labels_list.txt | 目标检测对应的label，初始化时读取该文件 |

### 数据结构设计
| stDetectResult | 检测结果数据结构 | |
|:---:|:---:|:---|
| 属性 | 类型 | 说明 | 
| pFrame | unsigned char* | 带框的检测结果图像 |
| nDetectNum | int | 检测到的目标总数量 |
| nWidth | int | 检测结果图像的宽度，和输入图像宽度一致 |
| nHeight | int | 检测结果图像的高度，和输入图像高度一致 |
| pClasses | int* | 检测到的目标类型数组[cls1, cls2, ...] |
| pBoxes | int* | 检测到的目标框数组[left1, top1, right1, bottom1, left2, top2, ...] |
| pProb | int* | 检测到的目标置信度数组[p1, p2, ...] |

### 接口设计
| 序号 | 接口名称 | 输入参数 | 输出参数 | 功能说明 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | Init | (const char* pModelString) | bool | 初始化，输入权重模型路径名称，加载模型，返回bool类型记录是否加载成功 |
| 2 | Deinit | - | bool | 反初始化，释放模型，返回bool类型记录是否释放成功 |
| 3 | Detect | (char* pChar, int nWidth, int nHeight) | stDetectResult | 检测，输入图像数据，图像宽高，输出检测结果 |
| 4 | setThreshold | (float fThreshold) | - | 设置阈值，即时生效 |
| 5 | setClassNum | (int nClassNum) | - | 设置模型能检测的目标总数 |
| 6 | printProfile | - | - | 打印性能统计信息到控制台 | 
