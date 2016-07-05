"""
Plotting functions to generate Scatter plots with
Matplotlib and Bokeh frameworks
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

from matplotlib import pyplot as plt
from bokeh.charts import Scatter, output_file, show

# ================
# ImageNet Classes
# ================
RASPBERRY = 'raspberry'
STRAWBERRY = 'strawberry'
BLACKBERRY = 'blackberry'
RED_CURRANT = 'redcurrant'
WHITE_CURRANT ='whitecurrant'
BLUEBERRY = 'blueberry'
CHERRY = 'cherry'
PLUM = 'plum'
APRICOT = 'apricot'
GOOSEBERRY = 'Gooseberry'

MARKERS = {RASPBERRY: 'o',
           STRAWBERRY: '+',
           BLACKBERRY: '*',
           RED_CURRANT: 'D',
           WHITE_CURRANT: 'h',
           BLUEBERRY: 's',
           CHERRY: 'd',
           PLUM: '8',
           APRICOT: 'p',
           GOOSEBERRY: '<'}

COLOURS = {RASPBERRY: '#ff6666',
           STRAWBERRY: '#794044',
           BLACKBERRY: '#000000',
           RED_CURRANT: '#f03939',
           WHITE_CURRANT: '#f0f688',
           BLUEBERRY: '#3a539b',
           CHERRY: '#f688b4',
           PLUM: '#9615f0',
           APRICOT: '#f0b015',
           GOOSEBERRY: '#15f024'}

CLASSES_OF_INTEREST = [APRICOT, PLUM, CHERRY, BLUEBERRY]

# =======================================================

RASPBERRY_BASE_CLASSES = ['E', 'G', 'L']

def make_plot(X, Y, colours, classes, fig_filename, title,
              s=10, annotate=False, sample_names=None):
    """
    generates and shows a scatter plot

    Parameters
    ----------
    X : numpy.ndarray
        values of x

    Y : numpy.ndarray
        values of y

    s : int (default=10)
        dimension of markers

    colours : list
        colour of the markers of the different classes

    classes : list
        predicted classes of the values

    sample_names : list
        list of labels for each dot (only if annotate=True)

    fig_filename : str
        name of the image file of the graph saved

    title : str
        title of the plot

    annotate : bool
        title of the plot
    """
    plt.figure()
    plt.title(title)
    for (i, cla) in enumerate(set(classes)):
        xc = [p for (j, p) in enumerate(X) if classes[j] == cla]
        yc = [p for (j, p) in enumerate(Y) if classes[j] == cla]
        if sample_names:
            nc = [p for (j, p) in enumerate(sample_names) if classes[j] == cla]
        else:
            nc = None
        cols = [c for (j, c) in enumerate(colours) if classes[j] == cla]
        if cla in MARKERS:
            marker = MARKERS[cla]
        else:
            print('\t WARNING: Class {} not found in MARKERS'.format(cla))
            marker = 'o'  # default marker value
        plt.scatter(xc, yc, s=s, marker=marker, c=cols, label=cla)

        if annotate and nc:
            for j, txt in enumerate(nc):
                plt.annotate(txt, (xc[j], yc[j]))

    plt.legend(loc=0)
    plt.savefig(fig_filename)
    plt.show()
    plt.clf()


def make_interactive_plot(data, fig_filename="tsne_plot.html",
                          title="t-SNE for Raspberries"):

    scatter = Scatter(data, x='X', y='Y',
                      color='classes', marker='classes',
                      title='t-SNE',
                      legend=True)
    output_file(fig_filename, title=title)
    show(scatter)


