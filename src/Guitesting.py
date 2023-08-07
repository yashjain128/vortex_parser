import sys, os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
                             QMenu, QPushButton, QRadioButton, QWidget, QLabel, QLineEdit, QFileDialog)
import test

class win(QWidget):
    def __init__(self, parent=None):
        super(win, self).__init__(parent)

        grid = QGridLayout()
        grid.addWidget(self.createSetupBox(), 0, 0)
        grid.addWidget(self.createLiveControlBox(), 1, 0)
        grid.addWidget(self.createGPSBox(), 0, 1, 2, 1)
        grid.addWidget(self.createHKbox("EFP", do_temp2=False, do_temp3=False, do_inttemp=False), 0, 2, 2, 1)
        grid.addWidget(self.createHKbox("mNLP", do_temp3=False, do_inttemp=False), 0, 3, 2, 1)
        grid.addWidget(self.createHKbox("ACC", do_temp2=False, do_temp3=False, do_inttemp=False, do_n12V=False, do_p12V=False, do_digtemp=True), 0, 4, 2, 1)


        self.statusLabel = QLabel("Ready")
        grid.addWidget(self.statusLabel, 2, 0)
            
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        self.setLayout(grid)

        self.setWindowTitle("Parser Title")
        self.resize(400, 300)

    def getFile(self, title, fdir, ftype, label=None):
        self.statusLabel.setText("Pick a file")
        fname, ftypeused = QFileDialog.getOpenFileName(self, title, fdir, ftype)
        if fname:
            if label is not None:
                label.setText(os.path.basename(fname))
            self.statusLabel.setText("Ready")
            return fname
        self.statusLabel.setText("Ready")
        return None

    def toggleParsing(self):
        if test.run:
            test.run = False
        else:
            if test.readfile is not None:
                test.run = True
                test.parse()
            else:
                self.statusLabel.setStyleSheet("color: red")
                self.statusLabel.setText("No file selected")

    def chooseRecording(self, label):
        file_path = self.getFile("Pick a udp recording", "", "UDP Files (*.udp; *.bin);;All files (*)", label)
        if file_path is not None:
            test.readfile = open(file_path, "rb")

    def chooseMap(self, label):
        file_path = self.getFile("Pick a map file", "", "Mat Map Files (*.mat);;All files (*)", label)
        if file_path is not None:
            test.plotting.change_map(file_path)

    def chooseFormatFile(self, label):
        file_path = self.getFile("Pick an instrument file", "", "Excel Files (*.xlsx);;All files (*)", label)
        if file_path is not None:
            test.formatfunc(file_path)

    def createSetupBox(self):
        groupBox = QGroupBox("Setup")
        # Top ------------------------------
        topLayout = QGridLayout()

        pickInstrLabel = QLabel("Instrument Format (.xlsx)")
        pickInstrButton = QPushButton("...")
        pickInstrButton.setFixedWidth(24)
        pickInstrNameLabel = QLabel("Pick a file")
        pickInstrNameLabel.setStyleSheet("background-color: white")
        pickInstrButton.clicked.connect(
            lambda: self.chooseFormatFile(pickInstrNameLabel))

        topLayout.addWidget(pickInstrLabel, 0, 0)

        topLayout.addWidget(pickInstrNameLabel, 0, 1)
        topLayout.addWidget(pickInstrButton, 0, 2)

        pickMapLabel = QLabel("Map File (.mat)")
        pickMapButton = QPushButton("...")
        pickMapButton.setFixedWidth(24)
        pickMapNameLabel = QLabel("Pick a file")
        pickMapNameLabel.setStyleSheet("background-color: white")
        pickMapButton.clicked.connect(
            lambda: self.chooseMap(pickMapNameLabel))

        topLayout.addWidget(pickMapLabel, 1, 0)
        topLayout.addWidget(pickMapNameLabel, 1, 1)
        topLayout.addWidget(pickMapButton, 1, 2)

        # Read File ------------------------
        readFileBox = QGroupBox("Read File")
        readFileBoxLayout = QGridLayout()

        pickReadFileButton = QPushButton("...")
        pickReadFileButton.setFixedWidth(24)
        pickReadFileNameLabel = QLabel("Pick a file")
        pickReadFileNameLabel.setStyleSheet("background-color: white")
        pickReadFileButton.clicked.connect(
            lambda: self.chooseRecording(pickReadFileNameLabel))

        readFileBoxLayout.addWidget(pickReadFileNameLabel, 0, 0)
        readFileBoxLayout.addWidget(pickReadFileButton, 0, 1)

        readFileBox.setLayout(readFileBoxLayout)

        # Connection Box -------------------
        liveUDPBox = QGroupBox("UDP")
        liveUDPBoxLayout = QGridLayout()

        hostLabel = QLabel("Local Host")
        hostInputLine = QLineEdit()
        hostInputLine.setFixedWidth(75)

        portLabel = QLabel("Local Port")
        portInputLine = QLineEdit()
        portInputLine.setFixedWidth(75)

        liveUDPBoxLayout.addWidget(hostLabel, 0, 0)
        liveUDPBoxLayout.addWidget(hostInputLine, 0, 1)
        liveUDPBoxLayout.addWidget(portLabel, 1, 0)
        liveUDPBoxLayout.addWidget(portInputLine, 1, 1)

        liveUDPBox.setLayout(liveUDPBoxLayout)

        vbox = QGridLayout()
        vbox.setColumnStretch(0, 1)
        vbox.setColumnStretch(1, 1)
        vbox.addLayout(topLayout, 0, 0, 1, 2)
        vbox.addWidget(readFileBox, 1, 0, 2, 1)
        vbox.addWidget(liveUDPBox, 1, 1, 2, 1)
        groupBox.setLayout(vbox)

        return groupBox

    def createLiveControlBox(self):
        groupBox = QGroupBox("Live Control")

        # Left Box
        dataLabel = QLabel("Collect Data")
        dataOnOff = QPushButton("Start")
        dataOnOff.setFixedWidth(40)
        dataOnOff.setStyleSheet("background-color: #e34040")
        dataOnOff.setCheckable(True)
        dataOnOff.clicked.connect(self.toggleParsing)
        # 29d97e

        writelabel = QLabel("Write to file ")
        writeOnOff = QPushButton("Start")
        writeOnOff.setFixedWidth(40)
        writeOnOff.setStyleSheet("background-color: #e34040")
        writeOnOff.setCheckable(True)

        hklabel = QLabel("Housekeeping counts/units ")
        hkCountUnit = QPushButton("Counts")
        hkCountUnit.setFixedWidth(40)
        hkCountUnit.setStyleSheet("background-color: #9e9e9e")

        leftBox = QGridLayout()

        leftBox.setRowStretch(0, 1)
        leftBox.addWidget(dataLabel, 0, 0)
        leftBox.addWidget(dataOnOff, 0, 1)
        leftBox.addWidget(writelabel, 1, 0)
        leftBox.addWidget(writeOnOff, 1, 1)
        leftBox.addWidget(hklabel, 2, 0)
        leftBox.addWidget(hkCountUnit, 2, 1)

        # Right Box
        readTimeLabel = QLabel("Read Session Time")
        readTimeOutput = QLineEdit()
        writeTimeLabel = QLabel("Write Session Time")
        writeTimeOutput = QLineEdit()
        writeFileNameLabel = QLabel("Write File Name")
        writeFileNameEdit = QLineEdit()

        rightBox = QGridLayout()
        rightBox.setRowStretch(0, 1)
        rightBox.addWidget(readTimeLabel, 0, 0)
        rightBox.addWidget(readTimeOutput, 0, 1)
        rightBox.addWidget(writeTimeLabel, 1, 0)
        rightBox.addWidget(writeTimeOutput, 1, 1)
        rightBox.addWidget(writeFileNameLabel, 2, 0)
        rightBox.addWidget(writeFileNameEdit, 2, 1)

        #
        vbox = QGridLayout()
        vbox.setColumnStretch(0, 1)
        vbox.setColumnStretch(1, 1)
        vbox.addLayout(leftBox, 0, 0)
        vbox.addLayout(rightBox, 0, 1)
        groupBox.setLayout(vbox)
        return groupBox

    def createGPSBox(self):
        groupBox = QGroupBox("GPS")

        latLabel = QLabel("Latitude (deg)")
        latField = QLineEdit()
        latField.setFixedWidth(75)

        lonLabel = QLabel("Longitude (deg)")
        lonField = QLineEdit()
        lonField.setFixedWidth(75)

        altLabel = QLabel("Altitude (deg)")
        altField = QLineEdit()
        altField.setFixedWidth(75)

        vEastLabel = QLabel("vEast (m/s)")
        vEastField = QLineEdit()
        vEastField.setFixedWidth(75)

        vWestLabel = QLabel("vWest (m/s)")
        vWestField = QLineEdit()
        vWestField.setFixedWidth(75)

        vUpLabel = QLabel("vUp (m/s)")
        vUpField = QLineEdit()
        vUpField.setFixedWidth(75)

        HorzLabel = QLabel("Horz. Speed (m/s)")
        HorzField = QLineEdit()
        HorzField.setFixedWidth(75)

        numSatsLabel = QLabel("Num Sats")
        numSatsField = QLineEdit()
        numSatsField.setFixedWidth(75)


        vbox = QGridLayout()
        vbox.addWidget(latLabel, 0, 0)
        vbox.addWidget(latField, 0, 1)
        vbox.addWidget(lonLabel, 1, 0)
        vbox.addWidget(lonField, 1, 1)
        vbox.addWidget(altLabel, 2, 0)
        vbox.addWidget(altField, 2, 1)
        vbox.addWidget(vEastLabel, 3, 0)
        vbox.addWidget(vEastField, 3, 1)
        vbox.addWidget(vWestLabel, 4, 0)
        vbox.addWidget(vWestField, 4, 1)
        vbox.addWidget(vUpLabel, 5, 0)
        vbox.addWidget(vUpField, 5, 1)
        vbox.addWidget(HorzLabel, 6, 0)
        vbox.addWidget(HorzField, 6, 1)
        vbox.addWidget(numSatsLabel, 7, 0)
        vbox.addWidget(numSatsField, 7, 1)

        groupBox.setLayout(vbox)

        return groupBox

    def createHKbox(self, name, do_temp1=True, do_temp2=True, do_temp3=True, do_inttemp=True, do_Vbat=True, do_n12V=True, do_p12V=True, do_p5V=True, do_33V=True, do_batmon=True, do_digtemp=False):
        groupBox = QGroupBox(name)

        temp1Label = QLabel("Temp 1")
        temp1Field = QLineEdit()
        if not do_temp1:
            temp1Label.setEnabled(False)
            temp1Field.setEnabled(False)
        temp2Label = QLabel("Temp 2")
        temp2Field = QLineEdit()
        if not do_temp2:
            temp2Label.setEnabled(False)
            temp2Field.setEnabled(False)
        temp3Label = QLabel("Temp 3")
        temp3Field = QLineEdit()
        if not do_temp3:
            temp3Label.setEnabled(False)
            temp3Field.setEnabled(False)
        inttempLabel = QLabel("Int. Temp")
        inttempField = QLineEdit()
        if not do_inttemp:
            inttempLabel.setEnabled(False)
            inttempField.setEnabled(False)
        vbatLabel = QLabel("V Bat")
        vbatField = QLineEdit()
        if not do_Vbat:
            vbatLabel.setEnabled(False)
            vbatField.setEnabled(False)
        n12VLabel = QLabel("-12 V")
        n12VField = QLineEdit()
        if not do_n12V:
            n12VLabel.setEnabled(False)
            n12VField.setEnabled(False)
        p12VLabel = QLabel("+12 V")
        p12VField = QLineEdit()
        if not do_p12V:
            p12VLabel.setEnabled(False)
            p12VField.setEnabled(False)
        p5VLabel = QLabel("+5 V")
        p5VField = QLineEdit()
        if not do_p5V:
            p5VLabel.setEnabled(False)
            p5VField.setEnabled(False)
        p33VLabel = QLabel("+3.3 V")
        p33VField = QLineEdit()
        if not do_33V:
            p33VLabel.setEnabled(False)
            p33VField.setEnabled(False)
        batmonLabel = QLabel("+3.3 V")
        batmonField = QLineEdit()
        if not do_batmon:
            batmonLabel.setEnabled(False)
            batmonField.setEnabled(False)
        vbox = QGridLayout()

        vbox.addWidget(temp1Label, 0, 0)
        vbox.addWidget(temp1Field, 0, 1)
        vbox.addWidget(temp2Label, 1, 0)
        vbox.addWidget(temp2Field, 1, 1)
        vbox.addWidget(temp3Label, 2, 0)
        vbox.addWidget(temp3Field, 2, 1)
        vbox.addWidget(inttempLabel, 3, 0)
        vbox.addWidget(inttempField, 3, 1)
        vbox.addWidget(vbatLabel, 4, 0)
        vbox.addWidget(vbatField, 4, 1)
        vbox.addWidget(n12VLabel, 5, 0)
        vbox.addWidget(n12VField, 5, 1)
        vbox.addWidget(p12VLabel, 6, 0)
        vbox.addWidget(p12VField, 6, 1)
        vbox.addWidget(p5VLabel, 7, 0)
        vbox.addWidget(p5VField, 7, 1)
        vbox.addWidget(p33VLabel, 8, 0)
        vbox.addWidget(p33VField, 8, 1)
        vbox.addWidget(batmonLabel, 9, 0)
        vbox.addWidget(batmonField, 9, 1)

        if do_digtemp:
            digtempLabel = QLabel("Dig. Temp")
            digtempField = QLineEdit()
            vbox.addWidget(digtempLabel, 10, 0)
            vbox.addWidget(digtempField, 10, 1)
        groupBox.setLayout(vbox)
        return groupBox


if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = win()
    clock.show()
    sys.exit(app.exec_())
