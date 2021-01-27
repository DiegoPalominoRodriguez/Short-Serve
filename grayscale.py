import cv2
import numpy as np
from hough_function import *

def gray(path, l_limit, h_limit):
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
            f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #ret, th1 = cv2.threshold(f_gray, 125, 175, cv2.THRESH_BINARY)
            #ret, th1 = cv2.threshold(f_gray, 127, 255, cv2.THRESH_BINARY_INV)
            #ret, th1 = cv2.threshold(f_gray, 100, 175, cv2.THRESH_TRUNC)
            ret, th1 = cv2.threshold(f_gray, l_limit, h_limit, cv2.THRESH_TOZERO)
            #ret, th1 = cv2.threshold(f_gray, 100, 175, cv2.THRESH_TOZERO_INV)

            th1 = cv2.GaussianBlur(th1, (15, 15), cv2.BORDER_DEFAULT)

            # Definir los parametros para el detector de blobs
            params = cv2.SimpleBlobDetector_Params()

            # Filtro por area
            params.filterByArea = True
            params.minArea = 5000
            params.maxArea = 100000

            # Filtro por circunferencia
            params.filterByCircularity = True
            params.minCircularity = 0.7

            # Set up the detector with default parameters.
            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs.
            keypoints = detector.detect(th1)
            if len(keypoints) == 1:
                x = np.append(x, keypoints[0].pt[0])
                y = np.append(y, keypoints[0].pt[1])
                p_frame = np.append(p_frame, i)

        else:
            break

    cap.release()

    output = [t_frames, x, y, p_frame]
    return output