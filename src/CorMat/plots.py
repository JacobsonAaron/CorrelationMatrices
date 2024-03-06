from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.patheffects as pe
import numpy as np


class plots():
    """A collection of helpful plots, built on matplotlib. These functions do not output plots; 
    instead, they accept axes as arguments and populate or modify those axes."""
    
    def timeseries(ax, TS, title: str = None):
        """A standard timeseries plot."""
        ax.imshow(TS, cmap="inferno")
        ax.set_xlabel("Time")
        ax.set_ylabel("Region Index")
        if title is not None:
            ax.set_title(title)
        
    
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