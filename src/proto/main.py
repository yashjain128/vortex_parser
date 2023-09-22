'''
This script is for running 

Written by Yash Jain
'''
import sys, platform
import ctypes
from gui import Window
from PyQt5.QtWidgets import QApplication

if platform.system()=="Windows":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u'sailparser')
else:
    print("Not tested for Mac/Linux")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
