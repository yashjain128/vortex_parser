import numpy as np

from vispy import plot as plot, app
from vispy.io import load_data_file

from vispy import scene
from vispy.io import read_mesh
from vispy.geometry import MeshData
from vispy.plot.plotwidget import PlotWidget

class ScrollingPlotWidget(scene.Widget):
    """

    """

    def __init__(self, *args, **kwargs):
        self.grid = None
        self.camera = None
        self.title = None
        self.title_widget = None
        self.yaxis, self.xaxis, self.zaxis = None, None, None
        self.ylabel, self.xlabel, self.zlabel = None, None, None
        self.ylims, self.xlim, self.zlims = None, None, None
        self.data = None
        self.view_grid = None
        self._configured = False
        self.visuals = []
        self.section_y_x = None

        super(ScrollingPlotWidget, self).__init__(*args, **kwargs)
        self.grid = self.add_grid(spacing=0, margin=10)

    def configure2d(self, title, xlabel, ylabel, xlims=(0, 1), ylims=(0, 1)):

        fg = "#000000"
        self.ylims = ylims
        self.xlims = xlims

        #     c0        c1      c2      c3      c4               
        #  r0 +---------+-------+-------+-------+---------+
        #     |         |               | title |         |
        #  r1 |         +-------+-------+-------+         |
        #     |         | ylabel| yaxis |  view | padding |
        #  r2 | padding +-------+-------+-------+         |
        #     |         |               | xaxis |         |
        #  r3 |         +-------+-------+-------+         |
        #     |         |               | xlabel|         |
        #  r4 |---------+-------+-------+-------+---------|
        #     |                   padding                 |
        #     +---------+-------+-------+-------+---------+

        # padding left
        padding_left = self.grid.add_widget(None, row=0, row_span=4, col=0)
        padding_left.width_min = 20
        padding_left.width_max = 30

        # padding right
        padding_right = self.grid.add_widget(None, row=0, row_span=4, col=4)
        padding_right.width_min = 20
        padding_right.width_max = 30

        # padding down
        padding_bottom = self.grid.add_widget(None, row=4, col=0, col_span=5)
        padding_bottom.height_min = 20
        padding_bottom.height_max = 20

        # row 0
        # title - column 4 to 5
        self.title = scene.Label(title, font_size=12, color="#000000")
        self.title_widget = self.grid.add_widget(self.title, row=0, col=3)
        self.title_widget.height_min = self.title_widget.height_max = 40    

        # row 1
        # ylabel - column 1
        # yaxis - column 2
        # view - column 3
        self.ylabel = scene.Label(ylabel, font_size=8, rotation=-90)
        ylabel_widget = self.grid.add_widget(self.ylabel, row=1, col=1)
        ylabel_widget.width_max = 1

        self.yaxis = scene.AxisWidget(orientation='left',
                                      text_color=fg,
                                      axis_color=fg,
                                      tick_color=fg)
        yaxis_widget = self.grid.add_widget(self.yaxis, row=1, col=2)
        yaxis_widget.width_max = 20

        # row 2
        # xaxis - column 3
        self.xaxis = scene.AxisWidget(orientation='bottom', 
                                      text_color=fg,
                                      axis_color=fg,
                                      tick_color=fg)
        xaxis_widget = self.grid.add_widget(self.xaxis, row=2, col=3)
        xaxis_widget.height_max = 20


        # row 3
        # xlabel - column 3
        self.xlabel = scene.Label(xlabel, font_size=8)
        xlabel_widget = self.grid.add_widget(self.xlabel, row=3, col=3)
        xlabel_widget.height_max = 40

        # This needs to be added to the grid last (to fix #1742)
        self.view = self.grid.add_view(row=1, col=3, border_color='grey', bgcolor="#efefef") 
        self.view.camera = 'panzoom'
        self.camera = self.view.camera
        self.camera.set_range(x=self.xlims, y=self.ylims)

        self._configured = True
        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)

        return self
    def configure3d(self, title, xlabel, ylabel, zlabel, xlims=(0, 1), ylims=(0, 1), zlims=(0, 1) ):
        self.xax = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_color='r', tick_color='r', text_color='r', font_size=16)

        yax = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_color='g', tick_color='g', text_color='g', font_size=16, parent=view.scene)

        zax = scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), axis_color='b', tick_color='b', text_color='b', font_size=16, parent=view.scene)
        zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
        zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
        zax.transform.rotate(-45, (0, 0, 1))
        return self
    def add_line(self, line):
        self.view.add(line) 
        
        return line
    def add_gridlines(self):
        self.view_grid = scene.visuals.GridLines(color=(0, 0, 0, 0.5))
        self.view_grid.set_gl_state('translucent')
        self.view.add(self.view_grid)

if __name__ == "__main__":
    fig = plot.Fig(size=(800, 600))
    fig._grid._default_class = ScrollingPlotWidget
    ax = fig[0, 0].configure3d("title", "x", "y", "z" )
    fig.show()
    app.run()