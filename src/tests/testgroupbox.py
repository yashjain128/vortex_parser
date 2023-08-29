import numpy as np

from vispy import app
from vispy import scene


canvas = scene.SceneCanvas(size=(800, 600), show=True, keys='interactive', bgcolor='#fff')

view3dax = scene.widgets.ViewBox(parent=canvas.scene, name='vb1',
                            margin=0, border_color='red')
grid = view3dax.add_grid(margin=0)
view3dax.pos = 0, 0
view3dax.size = 400, 300
view3dax.camera = 'turntable'
view3dax.camera.rect = (0, 0, 1, 1)

subview3dax = scene.widgets.ViewBox(parent=view3dax.scene, border_width=0.1,
                             margin=0, border_color='green')
subview3dax.pos = 0, 0
subview3dax.size = 1, 1
subview3dax.camera = 'panzoom'
subview3dax.camera.interactive = False
subview3dax.camera.rect = (0, 0, 10, 10)


xax =  scene.AxisWidget(orientation='bottom', axis_label='X Axis', axis_font_size=12, axis_label_margin=50, tick_label_margin=5, font_size = 25, axis_color = '#000', tick_color = '#000', text_color = '#000')
view3dax.add(xax)
xax.link_view(subview3dax)

right_padding = view3dax.add_widget(row=1, col=2, row_span=1)
right_padding.width_max = 50

#xax = scene.widgets.axis.AxisWidget(tick_direction=(0, -1), axis_color='#000000', tick_color='#000000', text_color='#000000', font_size=16, axis_width=1)


app.run()