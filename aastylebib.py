#!/usr/bin/env python

# helper module to define common styles for plots and figures
import os
import sys
import numpy as np
from scipy.constants import h,c,e,pi,N_A,physical_constants
from scipy import interpolate
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator,LogLocator,MultipleLocator,\
                    FormatStrFormatter,FuncFormatter,ScalarFormatter
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.patches import Rectangle,ConnectionPatch
import matplotlib.gridspec as gridspec
from PIL import Image
#import matplotlib.transforms as transforms
from glob import glob
import re

class Plotstyle(object):
    def __init__(self):
        self.pcolorlist = [
                            #(153/255., 153/255., 153/255.),
                            (228/255.,  26/255.,  28/255.),
                            ( 55/255., 126/255., 184/255.),
                            ( 77/255., 175/255.,  74/255.),
                            (152/255.,  78/255., 163/255.),
                            (255/255., 127/255.,   0/255.),
                            (255/255., 255/255.,  51/255.),
                            (166/255.,  86/255.,  40/255.),
                            (247/255., 129/255., 191/255.)]
        self.datpoints = {'marker':'o', 'edge_c':'grey', 'edge_lw':'0.3'}
    def setdefaults(self):
        # custom settings
        pass

def fig_golden_ratio(width):
    """Takes the width of an image (in centimetres) and returns a (width, height)
    tuple (in inches) with Golden ratio between width and height."""
    return (width/2.54, width/2.54/1.618)

def gen_figname(ext='pdf'):
    """Generate the figure filename from the script filename. The optional
    argument (default: pdf) states the file extension (without dot).))"
    return os.path.splitext(sys.argv[0])[0] + "." + ext
    """
    return os.path.splitext(sys.argv[0])[0] + "." + ext


use_seaborn = False
#try:
    #import seaborn as sns
    #use_seaborn = True
    #print(" > Seaborn loaded <")

    #sns.set_context("paper")
    #sns.set_style("ticks",{'xtick.direction': 'in','ytick.direction': 'in'})
    #sns.set_palette("colorblind") # deep,muted,pastel,bright,dark,colorblind
    ##sns.despine()

    #pcolorlist = sns.color_palette()


#except:
    #print(" > Seaborn NOT loaded <")

