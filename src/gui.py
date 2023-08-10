import sys, os

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
                             QMenu, QPushButton, QRadioButton, QWidget, QLabel, QLineEdit, QFileDialog)



class QSelectedGroupBox(QGroupBox):                  
    clicked = QtCore.pyqtSignal(str, object)     

    def __init__(self, name, title, *args):     
        super(QSelectedGroupBox, self).__init__(title, *args)
        self.name = name
        self.title = title

    def mousePressEvent(self, event):
        child = self.childAt(event.pos())
        if not child:
            win.toggle_read_udp(self.name)
        
class Window(QWidget):
    def getFile(self, title, fdir, ftype):
        self.statusLabel.setText("Pick a file")
        fname, ftypeused = QFileDialog.getOpenFileName(self, title, fdir, ftype)
        if fname:
            self.statusLabel.setText("Ready")
            return fname
        self.statusLabel.setText("Ready")
        return None

   
    def choose_recording(self):
        self.read_file = self.getFile("Pick a udp recording", "", "UDP Files (*.udp; *.bin);;All files (*)")
        if self.read_file is not None:
            self.pickReadFileNameLabel.setText(os.path.basename(self.read_file))

    def choose_map(self):
        file_path = self.getFile("Pick a map file", "", "Mat Map Files (*.mat);;All files (*)") 
        if file_path is not None:
            self.pickMapNameLabel.setText(os.path.basename(file_path))
        
    def choose_instr(self):
        file_path = self.getFile("Pick an instrument file", "", "Excel Files (*.xlsx);;All files (*)")
        if file_path is not None:
            self.pickInstrNameLabel.setText(os.path.basename(file_path))

    def toggle_read_udp(self, name):
        if name=='udp':
            self.liveUDPBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #000000; font-weight: bold;}") 
            self.readFileBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #aaaaaa;}")  
            self.mode = 1
        if name=='read':
            self.liveUDPBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #aaaaaa;}")
            self.readFileBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #000000; font-weight: bold;}")
            self.mode = 0
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.mode = 0
        self.read_file = None

        # Top ------------------------------
        self.setupGroupBox = QGroupBox("Setup")
        self.topLayout = QGridLayout()

        self.pickInstrLabel = QLabel("Instrument Format (.xlsx)")
        self.pickInstrButton = QPushButton("...")
        self.pickInstrButton.setFixedWidth(24)
        self.pickInstrButton.clicked.connect(self.choose_instr)
        self.pickInstrNameLabel = QLabel("Pick a file")
        self.pickInstrNameLabel.setStyleSheet("background-color: white")

        self.topLayout.addWidget(self.pickInstrLabel, 0, 0)
        self.topLayout.addWidget(self.pickInstrNameLabel, 0, 1)
        self.topLayout.addWidget(self.pickInstrButton, 0, 2)

        self.pickMapLabel = QLabel("Map File (.mat)")
        self.pickMapButton = QPushButton("...")
        self.pickMapButton.clicked.connect(self.choose_map)
        self.pickMapButton.setFixedWidth(24)
        self.pickMapNameLabel = QLabel("Pick a file")
        self.pickMapNameLabel.setStyleSheet("background-color: white")



        self.topLayout.addWidget(self.pickMapLabel, 1, 0)
        self.topLayout.addWidget(self.pickMapNameLabel, 1, 1)
        self.topLayout.addWidget(self.pickMapButton, 1, 2)

        self.toggleReadUDPLabel = QLabel("toggle Read File / UDP")


        # Read File ------------------------
        self.readFileBox = QSelectedGroupBox("read", "Read File")
        self.readFileBoxLayout = QGridLayout()
        self.readFileBox.setObjectName("ColoredGroupBox")
        self.readFileBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #000000; font-weight: bold;}")
        
        self.pickReadFileButton = QPushButton("...")
        self.pickReadFileButton.setFixedWidth(24)
        self.pickReadFileButton.clicked.connect(self.choose_recording)
        self.pickReadFileNameLabel = QLabel("Pick a file")
        self.pickReadFileNameLabel.setStyleSheet("background-color: white")

        self.readFileBoxLayout.addWidget(self.pickReadFileNameLabel, 0, 0)
        self.readFileBoxLayout.addWidget(self.pickReadFileButton, 0, 1)

        self.readFileBox.setLayout(self.readFileBoxLayout)

        # Connection Box -------------------
        self.liveUDPBox = QSelectedGroupBox("udp", "UDP")
        self.liveUDPBoxLayout = QGridLayout()
        self.liveUDPBox.setObjectName("ColoredGroupBox") 
        self.liveUDPBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #aaaaaa;}")

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

        
        
        self.plotHertzLabel = QLabel("Plot Update Rate (Hz)")
        self.plotHertzInput = QLineEdit(text="5")
        self.plotWidthLabel = QLabel("Plot Width (Seconds)")
        self.plotWidthInput= QLineEdit(text="10")
        self.plotSettingsBox = QGridLayout()
        self.plotSettingsBox.addWidget(self.plotHertzLabel, 0, 0)
        self.plotSettingsBox.addWidget(self.plotHertzInput, 0, 1)
        self.plotSettingsBox.addWidget(self.plotWidthLabel, 0, 2)
        self.plotSettingsBox.addWidget(self.plotWidthInput, 0, 3)

        self.setupBox = QGridLayout()
        self.setupBox.setColumnStretch(0, 1)
        self.setupBox.setColumnStretch(1, 1)
        self.setupBox.addLayout(self.topLayout, 0, 0, 1, 2)
        self.setupBox.addWidget(self.readFileBox, 1, 0, 2, 1)
        self.setupBox.addWidget(self.liveUDPBox, 1, 1, 2, 1)
        self.setupBox.addLayout(self.plotSettingsBox, 3, 0, 1, 2)
        self.setupGroupBox.setLayout(self.setupBox)

        # Left Box
        self.liveControlGroupBox = QGroupBox("Live Control")
        self.readStartLabel = QLabel("Collect Data")
        self.readStart = QPushButton("Start")
        self.readStart.setFixedWidth(40)
        self.readStart.setStyleSheet("background-color: #e34040")
        self.readStart.setCheckable(True)
        # 29d97e

        self.writeStartLabel = QLabel("Write to file ")
        self.writeStart = QPushButton("Start")
        self.writeStart.setFixedWidth(40)
        self.writeStart.setStyleSheet("background-color: #e34040")
        self.writeStart.setCheckable(True)

        self.hklabel = QLabel("Housekeeping counts/units ")
        self.hkCountUnit = QPushButton("Counts")
        self.hkCountUnit.setFixedWidth(40)
        self.hkCountUnit.setStyleSheet("background-color: #9e9e9e")

        self.leftBox = QGridLayout()
        self.leftBox.setRowStretch(0, 1)
        self.leftBox.addWidget(self.readStartLabel, 0, 0)
        self.leftBox.addWidget(self.readStart, 0, 1)
        self.leftBox.addWidget(self.writeStartLabel, 1, 0)
        self.leftBox.addWidget(self.writeStart, 1, 1)
        self.leftBox.addWidget(self.hklabel, 2, 0)
        self.leftBox.addWidget(self.hkCountUnit, 2, 1)

        # Right Box
        self.readTimeLabel = QLabel("Read Session Time")
        self.readTimeOutput = QLineEdit()
        self.readTimeOutput.setFixedWidth(100)
        self.writeTimeLabel = QLabel("Write Session Time")
        self.writeTimeOutput = QLineEdit() 
        self.writeTimeOutput.setFixedWidth(100)
        self.writeFileNameLabel = QLabel("Write File Name")
        self.writeFileNameEdit = QLineEdit()

        self.rightBox = QGridLayout()
        self.rightBox.setRowStretch(0, 1)
        self.rightBox.addWidget(self.readTimeLabel, 0, 0)
        self.rightBox.addWidget(self.readTimeOutput, 0, 1)
        self.rightBox.addWidget(self.writeTimeLabel, 1, 0)
        self.rightBox.addWidget(self.writeTimeOutput, 1, 1)
        self.rightBox.addWidget(self.writeFileNameLabel, 2, 0)
        self.rightBox.addWidget(self.writeFileNameEdit, 2, 1)

        self.liveControlBox = QGridLayout()
        self.liveControlBox.setColumnStretch(0, 1)
        self.liveControlBox.setColumnStretch(1, 1)
        self.liveControlBox.addLayout(self.leftBox, 0, 0)
        self.liveControlBox.addLayout(self.rightBox, 0, 1)

        self.liveControlGroupBox.setLayout(self.liveControlBox)
        self.statusLabel = QLabel("Ready")
         
        ### Add all 
        self.mainGrid = QGridLayout()        
        self.mainGrid.addWidget(self.setupGroupBox, 0, 0)
        self.mainGrid.addWidget(self.liveControlGroupBox, 1, 0)
        self.mainGrid.addWidget(self.statusLabel, 2, 0)

        self.setLayout(self.mainGrid)


app = QApplication(sys.argv)
win = Window()
win.show()

