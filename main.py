import cv2
import numpy as np
import os
from background import *
from grayscale import *
from hsv import *
import tkinter as tk
from tkinter import filedialog
from messages import *

def file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def main(path, l_limit, h_limit, mode):

    if os.path.exists('./images/img_show.png'):
        os.remove('./images/img_show.png')

    if os.path.exists('./images/img_seg.png'):
        os.remove('./images/img_seg.png')

    if mode == 0:
        output = back(path)
    elif mode == 1:
        output = gray(path, l_limit, h_limit)
    elif mode == 2:
        output = hsv(path, l_limit, h_limit)

    t_frames = output[0]
    x = output[1]
    y = output[2]
    p_frame = output[3]

    if len(x) == 0:
        wanr4()
        img = './images/error.png'
        output = np.array([img, img, 0, 0, 0, 0, 0, 0], dtype=object)
        return output
    else:
        #Se calcula el porcentaje de detección
        perc = int((len(x)/t_frames)*100)

        #Se determina la posición del punto más bajo detectado en el array de numpy y
        ymin_pos = np.where(y == np.amax(y))
        xmin = int(x[ymin_pos])
        i_frame = int(p_frame[ymin_pos])

        #Se extrae el frame del punto de impacto
        cap = cv2.VideoCapture(path)
        cap.set(1, i_frame)
        ret, show_frame = cap.read()
        ret, img_show = cap.read()
        j = 0
        while j < len(x):
            xp = int(x[j])
            yp = int(y[j])
            cv2.circle(img_show, (xp, yp), radius=50, color=(0, 0, 255), thickness=2)
            j = j + 1

        lines = line(show_frame, img_show, 180, 20, 50, 100)
        lines = line(show_frame, img_show, 180, 15, 240, 255)

        cv2.imwrite('./images/img_show.png', img_show)
        img_show = './images/img_show.png'

        if mode == 1:
            f_gray = cv2.cvtColor(show_frame, cv2.COLOR_BGR2GRAY)
            ret, th1 = cv2.threshold(f_gray, l_limit, h_limit, cv2.THRESH_TOZERO)
            cv2.imwrite('./images/img_seg.png', th1)
            img_seg = './images/img_seg.png'

        if mode == 2:
            f_hsv = cv2.cvtColor(show_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(f_hsv, l_limit, h_limit)
            mask = cv2.bitwise_not(mask)
            cv2.imwrite('./images/img_seg.png', mask)
            img_seg = './images/img_seg.png'

        cap.release()

        x1 = lines[1]
        y1 = lines[2]
        x2 = lines[3]
        y2 = lines[4]

        m = (y2-y1)/(x2-x1)
        xline = int((y[ymin_pos]-y1)/m + x1)

        if xline > xmin:
            des = str('NO FAUL')
        elif xline < xmin:
            des = str('FAUL')

        if mode == 0:
            cap = cv2.VideoCapture(path)
            fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            k = 0
            while (1):
                ret, frame = cap.read()
                if ret == True:
                    k = k + 1
                    fgmask = fgbg.apply(frame)
                    fgmask = cv2.bitwise_not(fgmask)
                    if k == i_frame:
                        cv2.imwrite('./images/img_seg.png', fgmask)
                        img_seg = './images/img_seg.png'
                        break

                else:
                    break
            cap.release()
        output = np.array([img_seg, img_show, t_frames, perc, i_frame, des, xline, xmin], dtype = object)
        return output

