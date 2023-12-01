from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.patheffects as pe
import numpy as np


class Plots():
    pass


    def image(ax,arr,xy,zoom,cmap=None):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=zoom, cmap=cmap)
        im.image.axes = ax
        ab = offsetbox.AnnotationBbox(im, xy, #xybox=(-50., 50.),
                            #xycoords='data', 
                            boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
        

    def outlineAnnotate(ax, text, location, inColor="black", outColor="white", linewidth=1, size=10):
        # Note: [0,0] is lower left, [1,1] is upper right.
        ax.text(location[0]+1, location[1]+1, text,
                size=size,
                color=inColor,
                path_effects=[pe.withStroke(linewidth=linewidth, foreground=outColor)])