import numpy as np
import cv2
from hough_function import *

def back(path):
    list = []
    x = np.array(list)
    y = np.array(list)
    p_frame = np.array(list)

    cap = cv2.VideoCapture(path)
    t_frames = cap.get(7)

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    i = 0
    while(1):
        ret, frame = cap.read()
        if ret == True:
            i = i +1
            fgmask = fgbg.apply(frame)
            fgmask = cv2.bitwise_not(fgmask)

            fgmask = cv2.GaussianBlur(fgmask, (15, 15), cv2.BORDER_DEFAULT)

            # Definir los parametros para el detector de blobs
            params = cv2.SimpleBlobDetector_Params()

            # Filtro por area
            params.filterByArea = True
            params.minArea = 5000
            params.maxArea = 100000

            #Filtro por circunferencia
            params.filterByCircularity = True
            params.minCircularity = 0.7

            # Set up the detector with default parameters.
            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs.
            keypoints = detector.detect(fgmask)
            if len(keypoints) == 1:
                x = np.append(x, keypoints[0].pt[0])
                y = np.append(y, keypoints[0].pt[1])
                p_frame = np.append(p_frame, i)

        else:
            break

    cap.release()

    output = [t_frames, x, y, p_frame]
    return output