import numpy as np
import time
import sys

from pandas import read_excel
import src.plotting as plotting

start_time = time.time()

readfilepath = "C:/Users/ayash/PycharmProjects/SAILproj/VortEx_test02.udp"
readfile = open(readfilepath, "rb")

formatfilepath = "C:/Users/ayash/PycharmProjects/SAILproj/SAILFormat.xlsx"
formatloc = read_excel(formatfilepath, 'Format', skiprows=0, nrows=2, usecols="C:D", names=[0, 1])
graphformat = read_excel(formatfilepath, 'Format', skiprows=formatloc[0][0] - 1, nrows=formatloc[1][0], usecols="C:H",
                         names=range(6))
plotting.init(formatloc[1][0])

graphs = []
for index, row in graphformat.iterrows():
    graphs.append(
        plotting.DataGraph(index, *row))

instrumentformat = read_excel(formatfilepath, 'Format', skiprows=formatloc[0][1] - 1, nrows=formatloc[1][1],
                              usecols="C:O", names=range(0, 13))
for index, row in instrumentformat.iterrows():
    g, color, protocol, signed, *bytedata = row.tolist()
    graphs[g].addline(color, protocol, signed, bytedata)

minframelength = 2 * 40
packetlength = minframelength + 44  # 128
maxreadlength = packetlength * 5000  # 640000
sync = np.array([64, 40, 107, 254], dtype=np.uint8)
targetsync = np.dot(sync, sync)
endianness = np.array([3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16,
                       23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28, 35, 34, 33, 32, 39, 38, 37, 36,
                       43, 42, 41, 40, 47, 46, 45, 44, 51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56,
                       63, 62, 61, 60, 67, 66, 65, 64, 71, 70, 69, 68, 75, 74, 73, 72, 79, 78, 77, 76])


def findSYNC(seq):
    candidates = np.where(np.correlate(seq, sync, mode='valid') == targetsync)[0]
    check = candidates[:, np.newaxis] + np.arange(4)
    mask = np.all((np.take(seq, check) == sync), axis=-1)
    return candidates[mask]


rawData = np.zeros(0).astype('uint8')
prev_ind = 0
plotting.fig.canvas.draw()
background = plotting.fig.canvas.copy_from_bbox(plotting.fig.bbox)

def main():
    while True:
        rawData = np.fromfile(readfile, dtype=np.uint8, count=maxreadlength)
        if rawData.size== 0:
            break
        rawData = np.append(rawData[prev_ind:], rawData)
        inds = findSYNC(rawData)

        prev_ind = inds[-1]
        inds = inds[:-1][(np.diff(inds) == packetlength)]
        inds[:-1] = inds[:-1][(np.diff(rawData[inds + 6]) != 0)]

        minframes = rawData[inds[:, None] + endianness].astype(int)

        oddframe = minframes[np.where(minframes[:, 57] & 3 == 1)]
        evenframe = minframes[np.where(minframes[:, 57] & 3 == 2)]
        oddsfid = minframes[np.where(minframes[:, 5] % 2 == 1)]
        evensfid = minframes[np.where(minframes[:, 5] % 2 == 0)]

        #Reset Canvas
        plotting.fig.canvas.restore_region(background)
        for g in graphs:
            for line in g.lines:
                line.getdata([minframes, oddframe, evenframe, oddsfid, evensfid])
                g.ax.draw_artist(line.line)
            plotting.fig.canvas.blit(g.ax.bbox)
        plotting.fig.canvas.flush_events()

if __name__=="__main__":
    print(f"{time.time() - start_time} seconds")
    main()
    print(f"Finished in  {time.time() - start_time} seconds")
    plotting.run()