from PyQt5.QtWidgets import QMessageBox
from gui import *

def verify(l_limit,h_limit, mode):
    if mode == 0:
        return 0
    elif mode == 1:
        if l_limit == '' or h_limit == '':
            warn2()
            return 1
        try:
            number = float(l_limit)
            if float(l_limit) < 0 or float(l_limit) > 255 or float(h_limit) < 0 or float(l_limit) > 255:
                warn3()
                return 1
            else:
                return 0
        except ValueError:
            warn3()
            return 1
    elif mode == 2:
        k = 0
        for i in l_limit:
            if i == '':
                warn2()
                return 1
            else:
                try:
                    number = float(i)
                    if float(i) < 0 or float(i) > 255:
                        warn3()
                        return 1
                    else:
                        k = k +1
                except ValueError:
                    warn3()
                    return 1
    if k == 6:
        return 0

def warn1():
    msg = QMessageBox()
    msg.setWindowTitle("Warning")
    msg.setText("The video file must be loaded before pressing start.")
    msg.setIcon(QMessageBox.Warning)
    msg.exec_()

def warn2():
    msg = QMessageBox()
    msg.setWindowTitle("Warning")
    msg.setText("All the required inputs need to be typed.")
    msg.setIcon(QMessageBox.Warning)
    msg.exec_()

def warn3():
    msg = QMessageBox()
    msg.setWindowTitle("Warning")
    msg.setText("The typed values must be between 0-255.")
    msg.setIcon(QMessageBox.Warning)
    msg.exec_()

def wanr4():
    msg = QMessageBox()
    msg.setWindowTitle("Warning")
    msg.setText("Detection process failed. Video file or parameters must be modified.")
    msg.setIcon(QMessageBox.Warning)
    msg.exec_()
