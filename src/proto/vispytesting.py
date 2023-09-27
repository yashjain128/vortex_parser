import numpy as np

from vispy import plot as plot, app
from vispy.io import load_data_file

from vispy import scene
from vispy.io import read_mesh
from vispy.geometry import MeshData
from vispy.plot.plotwidget import PlotWidget

class PlotWidget2(scene.Widget):
    """Widget to facilitate plotting

    Parameters
    ----------
    *args : arguments
        Arguments passed to the `ViewBox` super class.
    **kwargs : keywoard arguments
        Keyword arguments passed to the `ViewBox` super class.

    Notes
    -----
    This class is typically instantiated implicitly by a `Figure`
    instance, e.g., by doing ``fig[0, 0]``.
    """

    def __init__(self, *args, **kwargs):
        self._fg = kwargs.pop('fg_color', 'k')
        self.grid = None
        self.camera = None
        self.title = None
        self.title_widget = None
        self.yaxis, self.xaxis = None, None
        self.ylabel, self.xlabel = None, None
        self.ylims, self.xlims = None, None
        self.data = None
        self.view_grid = None
        self._configured = False
        self.visuals = []
        self.section_y_x = None

        super(PlotWidget2, self).__init__(*args, **kwargs)
        self.grid = self.add_grid(spacing=0, margin=10)

    def configure(self, title, xlabel, ylabel, xlims, ylims):

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

    def plot(self, color):
        """Plot a series of data using lines and markers

        Parameters
        ----------
        data : array | two arrays
            Arguments can be passed as ``(Y,)``, ``(X, Y)`` or
            ``np.array((X, Y))``.
        color : instance of Color
            Color of the line.
        symbol : str
            Marker symbol to use.
        line_kind : str
            Kind of line to draw. For now, only solid lines (``'-'``)
            are supported.
        width : float
            Line width.
        marker_size : float
            Marker size. If `size == 0` markers will not be shown.
        edge_color : instance of Color
            Color of the marker edge.
        face_color : instance of Color
            Color of the marker face.
        edge_width : float
            Edge width of the marker.
        title : str | None
            The title string to be displayed above the plot
        xlabel : str | None
            The label to display along the bottom axis
        ylabel : str | None
            The label to display along the left axis.
        connect : str | array
            Determines which vertices are connected by lines.

        Returns
        -------
        line : instance of LinePlot
            The line plot.

        See also
        --------
        LinePlot
        """
        datax = np.arange(self.xlims[1])
        datay = np.zeros(self.xlims[1])
        self.data = np.transpose(np.array([datax, datay]))
        line = scene.Markers(pos=self.data, edge_width=0, size=2, face_color=color, antialias=False)
        self.view.add(line)
        
        
        return line
    def add_gridlines(self):
        self.view_grid = vp.visuals.GridLines(color=(0, 0, 0, 0.3))
        self.view_grid.set_gl_state('translucent')
        self.view.add(self.view_grid)

fig = vp.Fig(size=(1200, 800), show=False)
fig._grid._default_class = PlotWidget2

ax = fig[0, 0].configure(
    title="My title", 
    xlabel="My xlabel", 
    ylabel="My ylabel", 
    xlims=(0, 2000), 
    ylims=(-2000, 2000))
l = ax.plot("#ff00ff")
ax.add_gridlines() 

fig[0, 1].configure(
    title="My title", 
    xlabel="My xlabel", 
    ylabel="My ylabel", 
    xlims=(0, 2000),
    ylims=(-1000, 1000))
fig[0, 2].configure(
    title="My title", 
    xlabel="My xlabel", 
    ylabel="My ylabel", 
    xlims=(0, 2000),
    ylims=(-1000, 1000))
fig[1, 0].configure(
    title="My title", 
    xlabel="My xlabel", 
    ylabel="My ylabel", 
    xlims=(0, 2000),
    ylims=(-1000, 1000))
fig[1, 1].configure(
    title="My title", 
    xlabel="My xlabel", 
    ylabel="My ylabel", 
    xlims=(0, 2000),
    ylims=(-1000, 1000))
fig[1, 2].configure(
    title="My title", 
    xlabel="My xlabel", 
    ylabel="My ylabel", 
    xlims=(0, 2000),
    ylims=(-1000, 1000))


if __name__ == '__main__':
    fig.show(run=True)
