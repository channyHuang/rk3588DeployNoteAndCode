import base64
import cv2
import numpy as np
import time
import threading

class ImageProcess():
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(ImageProcess, "_instance"):
            with ImageProcess._instance_lock:
                if not hasattr(ImageProcess, "_instance"):
                    ImageProcess._instance = object.__new__(cls)
        return ImageProcess._instance
    
    @staticmethod
    def image2bytes(image):
        '''
        数组转二进制
        image : numpy矩阵/cv格式图片
        byte_data：二进制数据
        '''
        #对数组的图片格式进行编码
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            print('image2bytes failed')
        #将数组转为bytes
        byte_data = encoded_image.tobytes()
        return byte_data

    @staticmethod
    def encode_image(image):
        try:
            result, imgencode = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # 建立矩阵
            data = np.array(imgencode)
            stringData = data.tostring()
            stringData = base64.b64encode(stringData)
            # 将numpy矩阵转换成字符形式，以便在网络中传输
            stringData = stringData.decode()
            return True, stringData
        except Exception as e:
            #log.critical(e)
            return False, None

    @staticmethod
    def decode_image(image_data):
        recv_data = image_data.encode()
        data = base64.b64decode(recv_data)
        img_data = np.asarray(bytearray(data),dtype='uint8')
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def save_images(record_dir, camera_id, image):
        time_str = time.strftime('20%y%m%d_%H%M%S', time.localtime(time.time()))
        m_time_str = time_str + '_' + str(time.time()).split('.')[-1][:3]
        image_path = record_dir + '/' + str(camera_id) + '_' + m_time_str + '.jpg'
        cv2.imwrite(image_path, image)
    
imageProcess = ImageProcess()
