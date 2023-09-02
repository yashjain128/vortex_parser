import sys
import numpy as np
from vispy import app, scene, visuals
from vispy.util.filter import gaussian_filter
from vispy.visuals.transforms import STTransform

from scipy.io import loadmat

canvas = scene.SceneCanvas(keys='interactive', bgcolor='w')
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up='z', fov=0, center = (0.5, 0.5))

gpsmap = loadmat("lib/Daytona_Map.mat")
latlim = gpsmap['latlim'][0]
lonlim = gpsmap['lonlim'][0]
mapdata = gpsmap['ZA']

img = scene.visuals.Image(mapdata)
img.transform = STTransform(scale=1)
view.add(img)

xax = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_color='k', tick_color='k', text_color='k', font_size=12, axis_width=1,  parent=view.scene)
yax = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_color='k', tick_color='k', text_color='k', font_size=12, axis_width=1, parent=view.scene)
zax = scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), axis_color='k', tick_color='k', text_color='k', font_size=12, axis_width=1, parent=view.scene)
zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()