import numpy as np
import sys
from vispy import app, visuals, scene

class Canvas(scene.SceneCanvas):
    """ A simple test canvas for testing the EditLineVisual """

    def __init__(self):
        scene.SceneCanvas.__init__(self, keys='interactive',
                                   size=(800, 800), show=True)

        # # Create some initial points
        self.unfreeze()
        # Add a ViewBox to let the user zoom/rotate
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 30
        self.show()
        self.selected_point = None
        scene.visuals.GridLines(parent=self.view.scene)
        self.freeze()

    def on_mouse_press(self, event):
        print(event.pos) # How to convert this pos to canvas position??


Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
canvas = Canvas()

p1 = Scatter3D(parent=canvas.view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)

# fake data
x = np.random.rand(100) * 10
y = np.random.rand(100) * 10
z = np.random.rand(100) * 10

# Draw it
point_list = [x, y, z]
point = np.array(point_list).transpose()
p1.set_data(point, symbol='o', size=6, edge_width=0.5, edge_color='blue')

if __name__ == "__main__":

    if sys.flags.interactive != 1:
        app.run()