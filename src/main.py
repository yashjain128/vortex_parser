'''
Created on 12.06.2018


Written by Yash Jain
'''
import sys, platform
import ctypes
from gui import Window
from PyQt5.QtWidgets import QApplication

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u'sailparser')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
