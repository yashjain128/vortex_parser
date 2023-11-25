"""
Module to handle GUI
Written by Yash Jain
"""
import time
import os
from os.path import dirname, abspath, basename
from datetime import datetime, timedelta

from openpyxl import load_workbook

from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QGridLayout, QGroupBox, QComboBox, QHBoxLayout, QFrame, QMainWindow,
                             QPushButton, QWidget, QLabel, QLineEdit, QFileDialog, QSpinBox, QDialog)

import plotting

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
   
    def findRecording(self):
        self.read_file = self.getFile("Pick a udp recording", "", "UDP Files (*.udp; *.bin);;All files (*)")
        if self.read_file is not None:
            self.pickReadFileNameLabel.setText(basename(self.read_file))

    def findMap(self):
        self.map_file = self.getFile("Pick a map file", "", "Mat Map Files (*.mat);;All files (*)") 
        if self.map_file is not None:   
            self.pickMapNameLabel.setText(basename(self.map_file))
            plotting.set_map(self.map_file)
    
    def pickInstr(self, n):
        if n==0:
            self.instr_file = None
            return
        file_path = self.found_instr_files[n-1]
        self.changeInstr(file_path)

    def findInstr(self):
        file_path = self.getFile("Pick an instrument file", "", "Excel Files (*.xlsx);;All files (*)")
        if file_path is not None:
            self.pickInstrNameEdit.setText(file_path)
            self.changeInstr(file_path)
        
    def changeInstr(self, file_path):
        if self.instr_file == file_path:
            return

        self.instr_file = file_path

        # The format and plot width cannot be changed after the plots are made
        self.pickInstrCombo.setEnabled(False)
        self.pickInstrButton.setEnabled(False)
        self.plotWidthSpin.setDisabled(True)
        self.plotWidthLabel.setDisabled(True) 

        plotting.plot_width = self.plotWidthSpin.value()
        
        xl_sheet = load_workbook(file_path, data_only=True).active
        getval = lambda c: str(xl_sheet[c].value)

        # Plots
        graph_row_start, graph_row_end = getval("C3"), getval("D3")
        for row_num in range(int(graph_row_start), int(graph_row_end)+1):
            row = [getval("B"+str(row_num))]
            i = 1
            while row[-1] != "None":
                row.append( getval(chr(ord("B")+i) + str(row_num)) )
                i += 1
            # Remove last cell "None"
            row = row[:-1]

            if len(row) == 0:
                continue
            elif row[0][0] == '#':
                plotting.add_channel(*row)
            else:
                plotting.add_graph(*row)

        # Map
        map_row_start, map_row_end = getval("C4"), getval("D4")
        for row_num in range(int(map_row_start), int(map_row_end)+1):
            row = [getval("B"+str(row_num))]
            i = 1
            while row[-1] != "None":
                row.append( getval(chr(ord("B")+i) + str(row_num)) )
                i += 1
            # Remove last cell "None"
            row = row[:-1]

            if len(row) == 0:
                continue
            else:
                plotting.add_map(*row)
                
        # Housekeeping
        hk_row_start, hk_row_end = getval("C5"), getval("D5")
        for row_num in range(int(hk_row_start), int(hk_row_end)+1):
            row = [getval("B"+str(row_num))]
            i = 1
            while row[-1] != "None":
                row.append( getval(chr(ord("B")+i) + str(row_num)) )
                i += 1

            # Remove last cell "None"
            row = row[:-1]

            if len(row) == 0:
                continue
            else:
                hkValues = self.addHousekeeping(row[0], row[7:])
                plotting.add_housekeeping(*row[1:7], hkValues)
        
        self.valuesWidget.show()
        plotting.finish_creating()

        for fig in plotting.figures.values():
            fig.native.setWindowIcon(QIcon('icon.png'))


    def addHousekeeping(self, title, ttable): 
        hkGroupBox = QGroupBox(title)
        hkLayout = QGridLayout()
        hkValues = [] # list of all edits
        
        # Add a label and edit for each gps value
        for ind, name in enumerate(plotting.HK_NAMES):
            hkLabel = QLabel(name)
            hkValue = QLineEdit()

            hkValue.setFixedWidth(65)
            hkValue.setReadOnly(True)

            if ttable[ind] == "False":
                hkLabel.setEnabled(False)
                hkValue.setEnabled(False)
    
            
            hkLayout.addWidget(hkLabel, ind, 0)
            hkLayout.addWidget(hkValue, ind, 1)
            
            hkValues.append(hkValue)
        
        hkGroupBox.setLayout(hkLayout)

        self.valuesLayout.addWidget(hkGroupBox)
        self.hkBoxes.append(hkGroupBox)

        return hkValues

                

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
        plotting.do_write = self.do_write 
        plotting.write_file = self.write_file
    
    def time_run(self):
        self.read_time+=1
        self.readTimeOutput.setText(str(timedelta(seconds=self.read_time)))
        if self.do_write:
            self.write_time+=1
            self.writeTimeOutput.setText(str(timedelta(seconds=self.write_time)))
            
    def time_read_reset(self):
        self.read_time = 0
        self.readTimeOutput.setText(str(timedelta(0)))

    def time_write_reset(self):
        self.write_time = 0
        self.writeTimeOutput.setText(str(timedelta(0)))
    
    def toggle_hk(self):
        self.do_hkunits = not self.do_hkunits
        print(f"set>>{self.do_hkunits}")
        plotting.set_hkunits(self.do_hkunits)

        if self.do_hkunits:
            self.hkCountUnit.setText("Units")
        else:
            self.hkCountUnit.setText("Counts")

    def toggle_parse(self):
        if self.readStart.isChecked():
            if self.read_mode==0 and self.read_file==None:
                print("Please select a read file")
                self.readStart.setChecked(False)
                return

            elif self.read_mode==1 and self.portInputLine.text()=="":
                print("No ip adress given")
                self.readStart.setChecked(False)
                return
            
            self.readStart.setText("Stop")
            self.readStart.setStyleSheet("background-color: #29d97e")
            self.setupGroupBox.setDisabled(True)
            self.time_read_reset()
            self.time_write_reset()
            self.timer.start(1000)
            
            plotting.parse(self.read_mode, self.plotHertzSpin.value(), self.read_file, self.hostInputLine.text(), self.portInputLine.text())
            #self.read_mode, self.read_file, self.hostInputLine.text(), self.portInputLine.text())
            
            self.timer.stop()
            self.readStart.setText("Start")
            self.readStart.setStyleSheet("background-color: #e34040")
            self.setupGroupBox.setEnabled(True)

            self.readStart.setChecked(False)
        else:
            plotting.running = False
    def close_plots(self):
        # Reset GUI after closing plotting window
        self.pickInstrCombo.setEnabled(True)
        self.pickInstrButton.setEnabled(True)
        self.plotWidthSpin.setEnabled(True)
        self.plotWidthLabel.setEnabled(True)
        self.pickInstrCombo.setCurrentIndex(0)
        self.instr_file = None

        for hkBox in self.hkBoxes:
            layout = hkBox.layout()
            while layout.count():
                layout.takeAt(0).widget().deleteLater()
            hkBox.deleteLater()

        self.hkBoxes.clear()
        #self.clear_layout(self.win.hkLayout)
        #self.clear_layout(self.widget_layout)
        self.valuesWidget.hide()

    # QMainWindow.closeEvent
    def closeEvent(self, close_msg):
        plotting.on_close(None)
    
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle("SAIL parser")

        self.dir = dirname(dirname(abspath(__file__)))

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.time_run)

        self.read_mode = 0
        self.read_file = None
        self.read_time = 0

        self.do_write = False
        self.write_file = None
        self.write_time = 0

        self.map_file = None

        self.instr_file = None
        self.search_dir = self.dir + "\\lib\\"
        self.found_instr_files = []

        self.do_hkunits = True

        self.plot_windows = {}
        self.hkBoxes = []

        plotting.close_signal = self.close_plots

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
        self.pickInstrButton.clicked.connect(self.findInstr)
        self.pickInstrNameEdit = QLineEdit("Pick a file")
        self.pickInstrNameEdit.setReadOnly(True)
        self.pickInstrCombo = QComboBox()
        self.pickInstrCombo.addItem("-- select file --")
        self.pickInstrCombo.addItems(map(basename, self.found_instr_files))
        self.pickInstrCombo.setCurrentIndex(0)
        self.pickInstrCombo.currentIndexChanged.connect(self.pickInstr)
        self.pickInstrCombo.setLineEdit(self.pickInstrNameEdit)
        self.pickInstrNameEdit.setStyleSheet("background-color: white")


        self.pickInstrLayout.addWidget(self.pickInstrLabel, 0, 0)
        self.pickInstrLayout.addWidget(self.pickInstrCombo, 0, 1)
        self.pickInstrLayout.addWidget(self.pickInstrButton, 0, 2)

        self.pickMapLayout = QGridLayout()
        self.pickMapLabel = QLabel("Map File (.mat)")
        self.pickMapButton = QPushButton("...")
        self.pickMapButton.clicked.connect(self.findMap)
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
        self.pickReadFileButton.clicked.connect(self.findRecording)
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
        self.hkCountUnit = QPushButton("Units")
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
        self.readTimeReset.clicked.connect(self.time_read_reset)

        self.readTimeOutput = QLineEdit(text="0:00:00", alignment=QtCore.Qt.AlignRight)
        self.readTimeOutput.setReadOnly(True)
        self.readTimeOutput.setFixedWidth(100)

        self.writeTimeLabel = QLabel("Write Session Time")
        self.writeTimeReset = QPushButton(u"\u27F3")
        self.writeTimeReset.setFlat(True)
        self.writeTimeReset.setFixedWidth(20)
        self.writeTimeReset.clicked.connect(self.time_write_reset)
        
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

        # live control box
        self.liveControlBox = QGridLayout()
        self.liveControlBox.setColumnStretch(0, 1)
        self.liveControlBox.setColumnStretch(1, 1)
        self.liveControlBox.addLayout(self.leftBox, 0, 0)
        self.liveControlBox.addLayout(self.rightBox, 0, 1)
        self.liveControlGroupBox.setLayout(self.liveControlBox)

        # Status Label [not working]
        self.statusLabel = QLabel("Select Mission")
         
        # Widget to hide/show gps and housekeeping
        self.valuesWidget = QFrame()
        self.valuesLayout = QHBoxLayout()

        # Gps values
        self.gpsGroupBox = QGroupBox("GPS")
        self.gpsLayout = QGridLayout()
        plotting.gpsValues = [] # list of all edits
        
        # Add a label and edit for each gps value
        for ind, (name, name_short) in enumerate(zip(plotting.GPS_NAMES, plotting.GPS_NAMES_ID)):
            gpsLabel = QLabel(name)
            gpsValue = QLineEdit()

            gpsValue.setFixedWidth(65)
            gpsValue.setReadOnly(True)
            
            self.gpsLayout.addWidget(gpsLabel, ind, 0)
            self.gpsLayout.addWidget(gpsValue, ind, 1)
            
            plotting.gps_values[name_short] = gpsValue
        
        self.gpsGroupBox.setLayout(self.gpsLayout)

        self.valuesLayout.addWidget(self.gpsGroupBox);
        self.valuesWidget.setLayout(self.valuesLayout)



        self.central_widget = QWidget()               # define central widget
        self.setCentralWidget(self.central_widget)    # set QMainWindow.centralWidget

        ### Add all of the groupboxes
        self.mainGrid = QGridLayout()        
        self.mainGrid.addWidget(self.setupGroupBox, 0, 0)
        self.mainGrid.addWidget(self.liveControlGroupBox, 1, 0)
        #self.mainGrid.addWidget(self.statusLabel, 2, 0)
        self.valuesWidget.hide()
        self.mainGrid.addWidget(self.valuesWidget, 0, 1, 2, 1)
        self.central_widget.setLayout(self.mainGrid)