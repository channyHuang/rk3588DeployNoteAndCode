import cv2
import time

from MainInference import inferenceCrop 

if __name__ == '__main__':
    inferenceCrop.startProcess()
    starttime = time.time()
    try:
        while time.time() - starttime < 300:
            data = inferenceCrop.getData()
            if data is not None:
                cv2.imshow('res', data[1])
                cv2.waitKey(1)
            time.sleep(0.1)

    except Exception as e:
        time.sleep(2)
        print('Error ', e)
    cv2.destroyAllWindows()
    inferenceCrop.stopProcess()
