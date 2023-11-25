'''
Creates the application and runs it.

Written for the Space and Atmospheric Instrumentation Laboratory
by Yash Jain
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
