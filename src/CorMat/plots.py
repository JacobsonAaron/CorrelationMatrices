from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.patheffects as pe


class plots():
    """A collection of helpful plots, built on matplotlib. These functions do not output plots; 
    instead, they accept axes as arguments and populate or modify those axes."""
    
    titleFontsize = 18
    xAxisFontsize = 15
    yAxisFontsize = 15
    tickLabelFontsize = 12
    
    def timeseries(ax, TS, title: str = None):
        """A standard timeseries plot."""
        ax.imshow(TS, cmap="inferno")
        ax.set_xlabel("Time", fontsize=plots.xAxisFontsize)
        ax.set_ylabel("Region Index", fontsize=plots.yAxisFontsize)
        ax.tick_params(axis='both', which='both', labelsize=plots.tickLabelFontsize)
        if title is not None:
            ax.set_title(title, fontsize=plots.titleFontsize)
        
    def correlationMat(ax, cmat, title: str = None):
        """A standard correlation matrix plot."""
        ax.imshow(cmat, cmap="inferno")
        ax.set_xlabel("Region Index", fontsize=plots.xAxisFontsize)
        ax.set_ylabel("Region Index", fontsize=plots.yAxisFontsize)
        ax.tick_params(axis='both', which='both', labelsize=plots.tickLabelFontsize)
        if title is not None:
            ax.set_title(title, fontsize=plots.titleFontsize)
    
    def make_legend(axs, labels, colormap, pos="best", fontsize=12):
        """Adds a legend given a list of (name, count) pairs.
        Count is the number of items corresponding to each label."""
        handles = []
        for i, (name, count) in enumerate(labels):
            hand = Line2D([0], [0], labels_txt=name.capitalize()+f" (x{count})", marker='o', markersize=6, linewidth=1, markerfacecolor=colormap.colors[i], markeredgecolor=colormap.colors[i], linestyle='')
            handles.append(hand)
        leg = axs.legend(handles=handles, fontsize=fontsize, loc=pos)
        return leg
        
    def imgAnnotate(ax, img, xy, zoom, cmap=None):
        """Place an image (img) as a picture-in-picture annotation at position xy."""
        im = offsetbox.OffsetImage(img, zoom=zoom, cmap=cmap)
        im.image.axes = ax
        ab = offsetbox.AnnotationBbox(im, xy, #xybox=(-50., 50.),
                            #xycoords='data', 
                            boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)

    def outlinedAnnotate(ax, text, location, inColor="black", outColor="white", linewidth=1, size=10):
        """Adds a text annotation to axes with outlined text. 
        Helpful for annotation legibility in certain cases."""
        # Note: [0,0] is lower left, [1,1] is upper right.
        ax.text(location[0]+1, location[1]+1, text,
                size=size,
                color=inColor,
                path_effects=[pe.withStroke(linewidth=linewidth, foreground=outColor)])
    
    def visualizeAlignment1D(ax, TS1, TS2, gap=5, P=None, title=None):
        """Visualizes the alignment of a 1D time series under a DTW path warping.
        <P> is the warping path to use, and <gap> is the vertical space to place between the two series."""
        if title is None:
            title = "Visualizing temporal alignment of TS1, TS2"
        plt.plot(TS1[:], '-bo')
        plt.plot(TS2[:]+gap, '-ro')
        for i in range(P.shape[0]):
            plt.plot(P[i,:], [TS1[P[i,0]], TS2[P[i,1]]+gap], 'k', linewidth=.5)
        ax.set_title(title, fontsize=plots.titleFontsize)
    