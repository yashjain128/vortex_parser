# Matplotlib and Numpy for plotting
import matplotlib.pyplot as plt
import matplotlib
from math import sqrt

import numpy as np

fig, axes = None, []
ncols, nrows = 0, 0


def init(n):
    global fig, axes, nrows, ncols
    lengths = []
    for i in range(1, int(sqrt(n)) + 1):
        if n / i == n // i:
            lengths.append(i)
            lengths.append(n // i)
    lengths.sort()
    mid = len(lengths) // 2
    ncols = lengths[mid]
    nrows = lengths[~mid]

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(9, 6))
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.4)

    #fig.patch.set_facecolor('#181818')
    plt.show(block=False)


class DataGraph:
    class DataLine:
        def __init__(self, line, protocol, signed, xlim, ylim, bytedata):
            self.line = line
            self.protocol = ['all', 'odd frame', 'even frame', 'odd sfid', 'even sfid'].index(protocol)

            self.signed = signed
            self.ylim = ylim

            self.b = []
            for i in range(0, len(bytedata), 3):
                if bytedata[i] != -1:
                    self.b.append(bytedata[i:i + 3])
            self.datay = np.zeros(xlim)
            self.line.set_data(np.arange(xlim), self.datay)

        def getdata(self, minframes):
            l = len(minframes[self.protocol])
            self.datay[:l] = np.zeros(l)
            #print(l)

            # bsm stands for byte shift mask
            for bsm in self.b:
                if bsm[2] < 0:
                    self.datay[:l] += (minframes[self.protocol][:, bsm[0]] & bsm[1]) >> abs(bsm[2])
                else:
                    self.datay[:l] += (minframes[self.protocol][:, bsm[0]] & bsm[1]) << bsm[2]

            if self.signed:
                self.datay[:l] = self.datay[:l]+(self.datay[:l] >= self.ylim[1])*(2*self.ylim[0])
            self.datay = np.roll(self.datay, -l)
            self.line.set_ydata(self.datay)

    def __init__(self, nth, title, xlabel, ylabel, xlim, ylim1, ylim2):
        self.ax = axes[nth % nrows][nth // nrows]
        self.xlim = xlim
        self.ylim = [ylim1, ylim2]

        self.ax.set_title(title, fontsize=10, fontweight='bold')
        self.ax.set_xlabel(xlabel=xlabel, fontsize=8)
        self.ax.set_ylabel(ylabel=ylabel, fontsize=8)
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(ylim1 * 1.1, ylim2 * 1.1)
        self.ax.tick_params(axis='both', which='major', labelsize=6)
        self.ax.ticklabel_format(axis='both', scilimits=(0, 0))
        self.ax.yaxis.get_offset_text().set_fontsize(6)
        self.ax.xaxis.get_offset_text().set_fontsize(6)
        self.lines = []

    def addline(self, color, sfid, signed, bytedata):
        line, = self.ax.plot([], [], color=color, lw=1, linestyle='None', marker='.', markersize=0.1, animated=True)
        self.lines.append(DataGraph.DataLine(line, sfid, signed, self.xlim, self.ylim, bytedata))


def run():
    plt.show()
