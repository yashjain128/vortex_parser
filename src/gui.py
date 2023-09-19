"""
Module to handle GUI
Written by Yash Jain
"""
import sys, os
from os.path import dirname, abspath, basename
from datetime import datetime, timedelta

from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QGridLayout, QGroupBox, QComboBox, QHBoxLayout, QFrame, QMainWindow,
                             QPushButton, QWidget, QLabel, QLineEdit, QFileDialog, QSpinBox)

from plotting import Plotting, plt

class QSelectedGroupBox(QGroupBox): 
    clicked = QtCore.pyqtSignal(str, object)     

    def __init__(self, title, func): 
        super(QSelectedGroupBox, self).__init__(title) 
        self.func = func
    def mousePressEvent(self, event):
        child = self.childAt(event.pos())
        if not child:
            self.func()
            
        
class Window(QMainWindow):
    def getFile(self, title, fdir, ftype):
        #self.statusLabel.setText("Pick a file")
        fname, ftypeused = QFileDialog.getOpenFileName(self, title, fdir, ftype)
        if fname:
            #self.statusLabel.setText("Ready")
            return fname
        #self.statusLabel.setText("Ready")
        return None
   
    def choose_recording(self):
        self.read_file = self.getFile("Pick a udp recording", "", "UDP Files (*.udp; *.bin);;All files (*)")
        if self.read_file is not None:
            self.pickReadFileNameLabel.setText(basename(self.read_file))

    def choose_map(self):
        file_path = self.getFile("Pick a map file", "", "Mat Map Files (*.mat);;All files (*)") 
        if file_path is not None:   
            self.pickMapNameLabel.setText(basename(file_path))
            self.plotting.add_map(file_path)
    
    def pick_instr(self, n):
        print(n)
        if n==0:
            self.instr_file = None
            return
        file_path = self.found_instr_files[n-1]
        self.change_instr(file_path)

    def find_instr(self):
        file_path = self.getFile("Pick an instrument file", "", "Excel Files (*.xlsx);;All files (*)")
        if file_path is not None:
            self.pickInstrNameEdit.setText(file_path)
            self.change_instr(file_path)
        
    def change_instr(self, file_path):
        if self.instr_file != file_path:
            self.instr_file = file_path
            self.pickInstrCombo.setEnabled(False)
            self.pickInstrButton.setEnabled(False)
            self.plotWidthSpin.setDisabled(True)
            self.plotWidthLabel.setDisabled(True)
            self.plotting.start_excel(self.instr_file)

    def toggle_to_udp(self):
        self.liveUDPBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #000000; font-weight: bold;}") 
        self.readFileBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #aaaaaa;}")  
        self.read_mode = 1
            
    def toggle_to_read(self):
        self.liveUDPBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #aaaaaa;}")
        self.readFileBox.setStyleSheet("QGroupBox#ColoredGroupBox { border: 1px solid #000000; font-weight: bold;}")
        self.read_mode = 0

    def toggle_write(self):
        if self.do_write:
            self.writeStart.setStyleSheet("background-color: #e34040")
            self.do_write=False
            self.writeFileNameEdit.setEnabled(True)
            self.write_file.close()
        else:
            self.writeStart.setStyleSheet("background-color: #29d97e")
            self.do_write=True
            self.writeFileNameEdit.setEnabled(False) 
        self.write_file = open(self.dir+"/recordings/"+self.writeFileNameEdit.text()+".udp", "ab")
    
    
    def time_run(self):
        self.read_time+=1
        self.readTimeOutput.setText(str(timedelta(seconds=self.read_time)))
        if self.do_write:
            self.write_time+=1
            self.writeTimeOutput.setText(str(timedelta(seconds=self.write_time)))
    def reset_read_time(self):
        self.read_time = 0
        self.readTimeOutput.setText(str(timedelta(0)))

    def reset_write_time(self):
        self.write_time = 0
        self.writeTimeOutput.setText(str(timedelta(0)))
    
    def toggle_hk(self):
        self.do_hkunits = not self.do_hkunits
        if self.do_hkunits:
            self.hkCountUnit.setText("Units")
        else:
            self.hkCountUnit.setText("Counts")


    def toggle_parse(self):
        if self.readStart.isChecked():
            if self.read_mode==0 and self.read_file==None:
                print("Please select a read file")
                self.readStart.setChecked(True)
                return
            elif self.read_mode==1 and self.portInputLine.text()=="":
                print("No ip adress given")
                self.readStart.setChecked(True)
                return
            
            self.readStart.setText("Stop")
            self.setupGroupBox.setDisabled(True)
            self.plotHertzSpin.setDisabled(True)
            self.plotHertzLabel.setDisabled(True)
            self.readStart.setStyleSheet("background-color: #29d97e")
            
            self.plotThread = QtCore.QThread(parent=self)
            self.plotting.moveToThread(self.plotThread)

            self.plotThread.started.connect(self.plotting.parse)
            self.plotting.finished.connect(self.plotThread.quit)
            self.plotting.finished.connect(self.timer.stop)
            self.plotThread.finished.connect(self.plotThread.deleteLater)
            self.plotThread.start()

            self.timer.start(1000)

        else:
            print("stopped")
            self.plotting.run = False

    def closeEvent(self, close_msg):
        plt.close()

    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle("SAIL parser")

        self.dir = dirname(dirname(abspath(__file__)))

        self.plotThread = None
        self.mainThread = QtCore.QCoreApplication.instance().thread()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.time_run)
        self.read_mode = 0
        self.read_file = None
        self.read_time = 0

        self.do_write = False
        self.write_file = None
        self.write_time = 0

        self.instr_file = None
        self.search_dir = self.dir + "\\lib\\"
        self.found_instr_files = []

        self.do_hkunits = True
        for file in os.listdir(self.search_dir):
             if file.endswith(".xlsx"):
                self.found_instr_files.append(self.search_dir + file)
        self.instr_file = None

        # Top ------------------------------
        self.setupGroupBox = QGroupBox("Setup")

        self.pickInstrLayout = QGridLayout()
        self.pickInstrLabel = QLabel("Instrument Format (.xlsx)")
        self.pickInstrButton = QPushButton("...")
        self.pickInstrButton.setFixedWidth(24)
        self.pickInstrButton.clicked.connect(self.find_instr)
        self.pickInstrNameEdit = QLineEdit("Pick a file")
        self.pickInstrNameEdit.setReadOnly(True)
        self.pickInstrCombo = QComboBox()
        self.pickInstrCombo.addItem("-- select file --")
        self.pickInstrCombo.addItems(map(basename, self.found_instr_files))
        self.pickInstrCombo.setCurrentIndex(0)
        self.pickInstrCombo.currentIndexChanged.connect(self.pick_instr)
        self.pickInstrCombo.setLineEdit(self.pickInstrNameEdit)
        self.pickInstrNameEdit.setStyleSheet("background-color: white")


        self.pickInstrLayout.addWidget(self.pickInstrLabel, 0, 0)
        self.pickInstrLayout.addWidget(self.pickInstrCombo, 0, 1)
        self.pickInstrLayout.addWidget(self.pickInstrButton, 0, 2)

        self.pickMapLayout = QGridLayout()
        self.pickMapLabel = QLabel("Map File (.mat)")
        self.pickMapButton = QPushButton("...")
        self.pickMapButton.clicked.connect(self.choose_map)
        self.pickMapButton.setFixedWidth(24)
        self.pickMapNameLabel = QLabel("Pick a file")
        self.pickMapNameLabel.setStyleSheet("background-color: white")

        self.pickMapLayout.addWidget(self.pickMapLabel, 1, 0)
        self.pickMapLayout.addWidget(self.pickMapNameLabel, 1, 1)
        self.pickMapLayout.addWidget(self.pickMapButton, 1, 2)

        self.toggleReadUDPLabel = QLabel("toggle Read File / UDP")


        # Read File ------------------------
        self.readFileBox = QSelectedGroupBox("Read File", self.toggle_to_read)
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
        self.liveUDPBox = QSelectedGroupBox("UDP", self.toggle_to_udp)
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
        self.plotHertzSpin = QSpinBox()
        self.plotHertzSpin.setMinimum(1)
        self.plotHertzSpin.setMaximum(24)
        self.plotHertzSpin.setValue(5)

        self.plotWidthLabel = QLabel("Plot Width (Seconds)")
        self.plotWidthSpin = QSpinBox()
        self.plotWidthSpin.setMinimum(1)
        self.plotWidthSpin.setMaximum(20)
        self.plotWidthSpin.setValue(5)

        self.plotSettingsBox = QGridLayout()
        self.plotSettingsBox.addWidget(self.plotHertzLabel, 0, 0)
        self.plotSettingsBox.addWidget(self.plotHertzSpin, 0, 1)
        self.plotSettingsBox.addWidget(self.plotWidthLabel, 0, 2)
        self.plotSettingsBox.addWidget(self.plotWidthSpin, 0, 3)

        self.setupBox = QGridLayout()
        self.setupBox.setColumnStretch(0, 1)
        self.setupBox.setColumnStretch(1, 1)
        self.setupBox.addLayout(self.pickInstrLayout, 0, 0, 1, 2)
        self.setupBox.addLayout(self.pickMapLayout, 1, 0, 1, 2)
        self.setupBox.addWidget(self.readFileBox, 2, 0, 2, 1)
        self.setupBox.addWidget(self.liveUDPBox, 2, 1, 2, 1)
        self.setupBox.addLayout(self.plotSettingsBox, 4, 0, 1, 2)
        self.setupGroupBox.setLayout(self.setupBox)

        # Left Box
        self.liveControlGroupBox = QGroupBox("Live Control")

        self.readStartLabel = QLabel("Collect Data")
        self.readStart = QPushButton(u"Start")
        self.readStart.setFixedWidth(40)
        self.readStart.setStyleSheet("background-color: #e34040")
        self.readStart.setCheckable(True)

        self.readStart.clicked.connect(self.toggle_parse)
        # 29d97e

        self.writeStartLabel = QLabel("Write to file ")
        self.writeStart = QPushButton("Start")
        self.writeStart.setFixedWidth(40)
        self.writeStart.setStyleSheet("background-color: #e34040")
        self.writeStart.setCheckable(True)
        self.writeStart.clicked.connect(self.toggle_write)

        self.hklabel = QLabel("Housekeeping counts/units ")
        self.hkCountUnit = QPushButton("Counts")
        self.hkCountUnit.setFixedWidth(40)
        self.hkCountUnit.setStyleSheet("background-color: #9e9e9e")
        self.hkCountUnit.released.connect(self.toggle_hk)

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
        self.readTimeReset = QPushButton(u"\u27F3")
        self.readTimeReset.setFlat(True)
        self.readTimeReset.setFixedWidth(20)
        self.readTimeReset.clicked.connect(self.reset_read_time)
        self.readTimeOutput = QLineEdit(text="0:00:00", alignment=QtCore.Qt.AlignRight)
        self.readTimeOutput.setReadOnly(True)
        self.readTimeOutput.setFixedWidth(100)
        self.writeTimeLabel = QLabel("Write Session Time")
        self.writeTimeReset= QPushButton(u"\u27F3")
        self.writeTimeReset.setFlat(True)
        self.writeTimeReset.setFixedWidth(20)
        self.writeTimeReset.clicked.connect(self.reset_write_time)
        self.writeTimeOutput = QLineEdit(text="0:00:00", alignment=QtCore.Qt.AlignRight) 
        self.writeTimeOutput.setReadOnly(True)
        self.writeTimeOutput.setFixedWidth(100)
        self.writeFileNameLabel = QLabel("Write File Name")
        self.writeFileNameEdit = QLineEdit("Recording"+datetime.today().strftime('%Y-%m-%d'))
        self.writeFileNameEdit.setFixedWidth(122)

        self.rightBox = QGridLayout()
        self.rightBox.setHorizontalSpacing(1)
        self.rightBox.setRowStretch(0, 1)
        self.rightBox.addWidget(self.readTimeLabel, 0, 0, 1, 1)
        self.rightBox.addWidget(self.readTimeReset, 0, 1, 1, 1)
        self.rightBox.addWidget(self.readTimeOutput, 0, 2, 1, 2)
        self.rightBox.addWidget(self.writeTimeLabel, 1, 0, 1, 1)
        self.rightBox.addWidget(self.writeTimeReset, 1, 1, 1, 1)
        self.rightBox.addWidget(self.writeTimeOutput, 1, 2, 1, 2)
        self.rightBox.addWidget(self.writeFileNameLabel, 2, 0)
        self.rightBox.addWidget(self.writeFileNameEdit, 2, 1, 1, 3)

        self.liveControlBox = QGridLayout()
        self.liveControlBox.setColumnStretch(0, 1)
        self.liveControlBox.setColumnStretch(1, 1)
        self.liveControlBox.addLayout(self.leftBox, 0, 0)
        self.liveControlBox.addLayout(self.rightBox, 0, 1)

        self.liveControlGroupBox.setLayout(self.liveControlBox)
        self.statusLabel = QLabel("Select Mission")
         
 
        self.hkLayout = QHBoxLayout()
        self.hkWidget = QFrame()
        self.hkWidget.setLayout(self.hkLayout)
        self.hkWidget.hide() 
        
        self.gpsLayout = QHBoxLayout()
        self.gpsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.gpsWidget = QFrame()
        self.gpsWidget.setLayout(self.gpsLayout)
        self.gpsWidget.hide()

        self.central_widget = QWidget()               # define central widget
        self.setCentralWidget(self.central_widget)    # set QMainWindow.centralWidget
        ### Add all 
        self.mainGrid = QGridLayout()        
        self.mainGrid.addWidget(self.setupGroupBox, 0, 0)
        self.mainGrid.addWidget(self.liveControlGroupBox, 1, 0)
        #self.mainGrid.addWidget(self.statusLabel, 2, 0)
        self.mainGrid.addWidget(self.hkWidget, 0, 1, 2, 1)
        self.mainGrid.addWidget(self.gpsWidget, 0, 2, 2, 1)
        self.central_widget.setLayout(self.mainGrid)

        self.plotting = Plotting(self)