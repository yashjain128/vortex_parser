import sys
import numpy as np

from vispy import app, scene
from vispy.util.filter import gaussian_filter
from vispy.io import read_png, imread
from scipy.io import loadmat

canvas = scene.SceneCanvas(keys='interactive', bgcolor='w')
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up='z', fov=60)

# Simple surface plot example
# x, y values are not specified, so assumed to be 0:50

img = imread("icon.png")
print(img)
#img = np.swapaxes(img, 0, 1)
p1 = scene.visuals.Image(data=img )
view.add(p1)

# p1._update_data()  # cheating.
# cf = scene.filters.ZColormapFilter('fire', zrange=(z.max(), z.min()))
# p1.attach(cf)


xax = scene.Axis(pos=[[-0.5, -0.5], [0.5, -0.5]], tick_direction=(0, -1),
                 font_size=16, axis_color='k', tick_color='k', text_color='k',
                 parent=view.scene)
xax.transform = scene.STTransform(translate=(0, 0, -0.2))

yax = scene.Axis(pos=[[-0.5, -0.5], [-0.5, 0.5]], tick_direction=(-1, 0),
                 font_size=16, axis_color='k', tick_color='k', text_color='k',
                 parent=view.scene)
yax.transform = scene.STTransform(translate=(0, 0, -0.2))

# Add a 3D axis to keep us oriented
axis = scene.visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()