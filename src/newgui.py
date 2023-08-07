import sys, os
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
                             QMenu, QPushButton, QRadioButton, QWidget, QLabel, QLineEdit, QFileDialog)
import test

class win(QWidget):
    def getFile(self, title, fdir, ftype):
        self.statusLabel.setText("Pick a file")
        fname, ftypeused = QFileDialog.getOpenFileName(self, title, fdir, ftype)
        if fname:
            self.statusLabel.setText("Ready")
            return fname
        self.statusLabel.setText("Ready")
        return None

   
    def chooseRecording(self):
        file_path = self.getFile("Pick a udp recording", "", "UDP Files (*.udp; *.bin);;All files (*)")
        self.pickReadFileNameLabel.setText(os.path.basename(file_path))
        if file_path is not None:
#            test.readfile = open(file_path, "rb")
            pass

    def chooseMap(self):
        file_path = self.getFile("Pick a map file", "", "Mat Map Files (*.mat);;All files (*)") 
        self.pickMapNameLabel.setText(os.path.basename(file_path))
        if file_path is not None:
            #test.plotting.change_map(file_path)
            pass
        
    def chooseFormatFile(self):
        file_path = self.getFile("Pick an instrument file", "", "Excel Files (*.xlsx);;All files (*)")
        if file_path is not None:
            #test.formatfunc(file_path)
            pass


    def __init__(self, parent=None):
        super(win, self).__init__(parent)

        # Top ------------------------------
        self.topLayout = QGridLayout()

        self.pickInstrLabel = QLabel("Instrument Format (.xlsx)")
        self.pickInstrButton = QPushButton("...")
        self.pickInstrButton.setFixedWidth(24)
        self.pickInstrNameLabel = QLabel("Pick a file")
        self.pickInstrNameLabel.setStyleSheet("background-color: white")

        self.topLayout.addWidget(self.pickInstrLabel, 0, 0)
        self.topLayout.addWidget(self.pickInstrNameLabel, 0, 1)
        self.topLayout.addWidget(self.pickInstrButton, 0, 2)

        self.pickMapLabel = QLabel("Map File (.mat)")
        self.pickMapButton = QPushButton("...")
        self.pickMapButton.setFixedWidth(24)
        self.pickMapNameLabel = QLabel("Pick a file")
        self.pickMapNameLabel.setStyleSheet("background-color: white")

        self.topLayout.addWidget(self.pickMapLabel, 1, 0)
        self.topLayout.addWidget(self.pickMapNameLabel, 1, 1)
        self.topLayout.addWidget(self.pickMapButton, 1, 2)

        # Read File ------------------------
        self.readFileBox = QGroupBox("Read File")
        self.readFileBoxLayout = QGridLayout()

        self.pickReadFileButton = QPushButton("...")
        self.pickReadFileButton.setFixedWidth(24)
        self.pickReadFileNameLabel = QLabel("Pick a file")
        self.pickReadFileNameLabel.setStyleSheet("background-color: white")

        self.readFileBoxLayout.addWidget(self.pickReadFileNameLabel, 0, 0)
        self.readFileBoxLayout.addWidget(self.pickReadFileButton, 0, 1)

        self.readFileBox.setLayout(self.readFileBoxLayout)

        # Connection Box -------------------
        self.liveUDPBox = QGroupBox("UDP")
        self.liveUDPBoxLayout = QGridLayout()

        self.hostLabel = QLabel("Local Host")
        self.hostInputLine = QLineEdit()
        self.hostInputLine.setFixedWidth(75)

        self.portLabel = QLabel("Local Port")
        self.portInputLine = QLineEdit()
        self.portInputLine.setFixedWidth(75)

        self.liveUDPBoxLayout.addWidget(self.hostLabel, 0, 0)
        self.liveUDPBoxLayout.addWidget(self.hostInputLine, 0, 1)
        self.liveUDPBoxLayout.addWidget(self.portLabel, 1, 0)
        self.liveUDPBoxLayout.addWidget(self.portInputLine, 1, 1)
        self.liveUDPBox.setLayout(self.liveUDPBoxLayout)

        self.setupBox = QGridLayout()
        self.setupBox.setColumnStretch(0, 1)
        self.setupBox.setColumnStretch(1, 1)
        self.setupBox.addLayout(self.topLayout, 0, 0, 1, 2)
        self.setupBox.addWidget(self.readFileBox, 1, 0, 2, 1)
        self.setupBox.addWidget(self.liveUDPBox, 1, 1, 2, 1)





        self.mainGrid = QGridLayout()        
        self.mainGrid.addLayout(self.setupBox, 0, 0)
        self.setLayout(self.mainGrid)
 
 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = win()
    clock.show()
    sys.exit(app.exec_())
