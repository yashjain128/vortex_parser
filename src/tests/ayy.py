import os, sys
from os.path import dirname, abspath
from PyQt5.QtWidgets import QComboBox, QApplication, QWidget, QGridLayout, QLineEdit
from PyQt5.QtGui import QIcon

app = QApplication(sys.argv)
w = QWidget()

search_dir = dirname(dirname(dirname(abspath(__file__))))+"\lib"
combobox = QComboBox()
for file in os.listdir(search_dir):
    if file.endswith(".xlsx"):
        combobox.addItem(file)

edit = QLineEdit()
edit.setReadOnly(True)
combobox.setLineEdit(edit)
combobox.setCurrentText("pick a file")
layout = QGridLayout()
layout.addWidget(combobox, 0, 0)

w.setLayout(layout)

w.show()
sys.exit(app.exec_())