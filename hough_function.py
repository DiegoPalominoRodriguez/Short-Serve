import cv2
import numpy as np

def line(img, img_out, angle, long, l_gray, h_gray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, l_gray, h_gray, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / angle, long)
    #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for r, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

    cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    output = np.array([img_out, x1, y1, x2, y2], dtype=object)

    return output