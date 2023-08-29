import sys
import numpy as np
from vispy import app, scene
from vispy.util.filter import gaussian_filter
from vispy.io import read_png, load_data_file

from scipy.io import loadmat

canvas = scene.SceneCanvas(keys='interactive', bgcolor='w')
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up='z', fov=60)

gpsmap = loadmat("lib/Daytona_Map.mat")
latlim = gpsmap['latlim'][0]
lonlim = gpsmap['lonlim'][0]
mapdata = gpsmap['ZA']


view.camera.set_range(margin=0)
ss = scene.widgets.ViewBox(parent=view.scene, name='vb1', margin=2, border_color='red')
ss.camera = 'panzoom'
ss.camera.rect = (0, 0, 1, 1)
#image = scene.visuals.Image(mapdata, parent=ss, clim='auto')

xax = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_color='#000000', tick_color='#000000', text_color='#000000', font_size=16, parent=ss.scene,axis_width=1)
yax = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_color='#000000', tick_color='#000000', text_color='#000000', font_size=16, parent=ss.scene, axis_width=1)
zax = scene.Axis(pos=[[0, 0], [-150, 0]], tick_direction=(0, -1), axis_color='#000000', tick_color='#000000', text_color='#000000', font_size=16, parent=ss.scene, axis_width=1)
zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()