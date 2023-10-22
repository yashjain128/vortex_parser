import numpy as np

from vispy import app, scene
from vispy.color import Color


class EditVisual(scene.visuals.Compound):
    def __init__(self, on_select_callback=None, *args, **kwargs):
        scene.visuals.Compound.__init__(self, [], *args, **kwargs)
        self.unfreeze()
        self._on_select_callback = on_select_callback
        self.freeze()

    def add_subvisual(self, visual):
        scene.visuals.Compound.add_subvisual(self, visual)
        visual.interactive = True

    def select(self):
        self._on_select_callback()
    @property
    def selectable(self):
        return self._selectable

    @selectable.setter
    def selectable(self, val):
        self._selectable = val



class EditEllipseVisual(EditVisual):
    def __init__(self, center=[0, 0], radius=[2, 2], *args, **kwargs):
        EditVisual.__init__(self, *args, **kwargs)
        self.unfreeze()
        self.ellipse = scene.visuals.Ellipse(center=center, radius=radius,
                                             color=Color("#e88834"),
                                             border_color="white",
                                             parent=self)
        self.ellipse.interactive = True

        self.freeze()
        self.add_subvisual(self.ellipse)


class Canvas(scene.SceneCanvas):
    """ A simple test canvas for drawing demo """

    def __init__(self):
        scene.SceneCanvas.__init__(self, keys='interactive',
                                   size=(800, 800))

        self.unfreeze()

        self.view = self.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera()

        self.ellipse_button = \
            EditEllipseVisual(parent=self.view,
                              on_select_callback=lambda:print("Hi"),
                              center=[50, 170],
                              radius=[15, 10])

        self.objects = []
        self.show()
        self.selected_object = None
        scene.visuals.GridLines(parent=self.view.scene)
        self.freeze()

    def on_key_press(self, event):
        if event.text == '\x12':
            print("hi")

    def on_mouse_press(self, event):

        tr = self.scene.node_transform(self.view.scene)
        pos = tr.map(event.pos)
        self.view.interactive = False
        selected = self.visual_at(event.pos)
        self.view.interactive = True
        if self.selected_object is not None:
            self.selected_object.select()
            self.selected_object = None

        if event.button == 1:
            if selected is not None:
                self.selected_object = selected.parent
                # update transform to selected object
                tr = self.scene.node_transform(self.selected_object)
                pos = tr.map(event.pos)

                self.selected_object.select()

if __name__ == '__main__':
    canvas = Canvas()
    app.run()