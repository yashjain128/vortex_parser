import numpy as np

from vispy import plot as plot, app
from vispy import scene

class ScrollingPlotWidget(scene.Widget):
    """
    Widget for 2d and 3d plots.
    This widget is a container for a PlotWidget and a set of axes.
    """
    def __init__(self, *args, **kwargs):
        self.grid = None
        self.plot_grid = None
        self.plot_view = None
        self.camera = None
        self.title = None
        self.title_widget = None
        self.yaxis, self.xaxis, self.zaxis = None, None, None
        self.ylabel, self.xlabel, self.zlabel = None, None, None
        self.ylims, self.xlims, self.zlims = None, None, None
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
        self.plot_view = self.grid.add_view(row=1, col=3, border_color='grey', bgcolor="#efefef") 
        self.plot_view.camera = 'panzoom'
        self.camera = self.plot_view.camera
        self.camera.set_range(x=self.xlims, y=self.ylims)

        self._configured = True
        self.xaxis.link_view(self.plot_view)
        self.yaxis.link_view(self.plot_view)

        return self

    def configure3d(self, title, xlabel, ylabel, zlabel, xlims=(0, 1), ylims=(0, 1), zlims=(0, 1) ):

        fg = "#000000"
        self.plot_view = self.grid.add_view(row=1, col=1, border_color='grey', bgcolor="#efefef")
        self.plot_view.camera = 'turntable'
        self.plot_view.camera.center = (0.5, 0.5, 0.5)
        #.center((0.5, 0.5, 0.5))
        self.camera = self.plot_view.camera
        
        #return
        self.ylims = ylims
        self.xlims = xlims

        # padding left
        padding_left = self.grid.add_widget(None, row=1, row_span=2, col=0)
        padding_left.width_min = 20
        padding_left.width_max = 30

        # padding right
        padding_right = self.grid.add_widget(None, row=1, row_span=2, col=2)
        padding_right.width_min = 20
        padding_right.width_max = 30

        # padding down
        padding_bottom = self.grid.add_widget(None, row=2, col=0, col_span=3)
        padding_bottom.height_min = 20
        padding_bottom.height_max = 20

        # row 0 
        # title - column 4 to 5
        self.title = scene.Label(title, font_size=12, color=fg)
        self.title_widget = self.grid.add_widget(self.title, row=0, col=1)
        self.title_widget.height_min = self.title_widget.height_max = 40    

        self.xaxis = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_width=0.1, tick_width=1, domain=xlims,
                                axis_color=fg, tick_color=fg, text_color=fg, font_size=20, parent=self.plot_view.scene)
        self.yaxis = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_width=1, tick_width=1, domain=ylims,
                                axis_color=fg, tick_color=fg, text_color=fg, font_size=20, parent=self.plot_view.scene)
        self.zaxis = scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), axis_width=1, tick_width=1, domain=zlims,
                                axis_color=fg, tick_color=fg, text_color=fg, font_size=20, parent=self.plot_view.scene)
        self.zaxis.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
        self.zaxis.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
        self.zaxis.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)
        self._configured = True

        return self
    def add_line(self, line):
        self.plot_view.add(line) 
        
        return line
    def add_gridlines(self):
        self.view_grid = scene.visuals.GridLines(color=(0, 0, 0, 0.5))
        self.view_grid.set_gl_state('translucent')
        self.plot_view.add(self.view_grid)
    
    def reset_bounds(self):
        self.camera.set_range(x=self.xlims, y=self.ylims)