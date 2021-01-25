import cv2
import numpy as np
from hough_function import *

def hsv(path, l_hsv, h_hsv):
    list = []
    x = np.array(list)
    y = np.array(list)
    p_frame = np.array(list)

    cap = cv2.VideoCapture(path)
    t_frames = cap.get(7)

    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            i = i + 1
            f_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(f_hsv, l_hsv, h_hsv)
            mask = cv2.bitwise_not(mask)
            mask = cv2.GaussianBlur(mask, (15, 15), cv2.BORDER_DEFAULT)

            #Definir los parametros para el detector de blobs
            params = cv2.SimpleBlobDetector_Params()

            #Filtro por area
            params.filterByArea = True
            params.minArea = 5000
            params.maxArea = 100000

            #Filtro por circunferencia
            params.filterByCircularity = True
            params.minCircularity = 0.7

            # Set up the detector with default parameters.
            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs.
            keypoints = detector.detect(mask)

            if len(keypoints) == 1:
                x = np.append(x, keypoints[0].pt[0])
                y = np.append(y, keypoints[0].pt[1])
                p_frame = np.append(p_frame, i)

        else:
            break

    cap.release()

    output = [t_frames, x, y, p_frame]
    return output