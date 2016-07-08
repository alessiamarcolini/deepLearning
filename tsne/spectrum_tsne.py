"""Calculates t-SNE on Spectra data files
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

import os
import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt

DATA_FILE = os.path.join(os.path.abspath(os.path.curdir), '..', 'data',
                         'spectrum', 'raspberry_SO4_mean_box.dat')

SO4_IMAGES_FOLDER = os.path.join(os.path.abspath(os.path.curdir), '..', 'data',
                                 'images', 'so4')

COLORS_MAP = {'e': 'orange',
              'g': 'red',
              'l': 'purple'}

MARKERS_MAP = {'l': 'o',
               't': 's'}

def compose_output_filename(classes, filename_prefix,
                            file_ext='.txt', **additional_args):
    """
    Compose the name of file name including a short name for classes, according
    to the given `filename_prefix` template name.


    Parameters
    ----------
    classes: list
        the list of the classes for all the collected images

    filename_prefix: str (default: `OUTPUT_IMAGES_FILE_PREFIX`)
        The reference prefix for the composed file name.

    file_ext: str (default: .txt)
        The extension of the output filename

    additional_args: dict (optional)
        Dictionary containing additional parameters to include in the output filename.

    Returns
    -------
    matrix_filename: str
        the name of the output matrix file

    """
    classes_set = sorted([cl.lower() for cl in set(classes)])
    matrix_filename = filename_prefix + '_'
    for class_name in classes_set:
        matrix_filename += class_name[:5].lower() + "_"

    if additional_args:
        for name in sorted(additional_args):
            value = additional_args[name]
            matrix_filename += '{}_{}_'.format(name, value)

    matrix_filename += file_ext
    return matrix_filename


def collect_info_from_punnets(dataset_folder):
    """Collect class labels for each variety of raspberries
    in each punnet. These information are embedded in the
    names of files, whose naming pattern is:

        dataset_variety_label_conf_punnetNo_photoNo.jpg

    Parameters
    ----------
    dataset_folder : str
        Path to the folder containing all the datasets

    Returns
    -------
    dict
        Dictionary containing information of corresponding
        labels for each punnet (numbered from 1 to 16 in so4)
        and for each variety of raspberry contained in the
        dataset.

        This dictionary is structured as follows:
          punnet_number --> { 'variety': .. , 'label': "e|g|l" }
    """

    punnets_info = dict()
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            if filename.endswith('.jpg'):
                fname, _ = os.path.splitext(filename)
                _, variety, label, _, punnet_no, _ = fname.split('_')

                punnet_no = int(punnet_no)
                punnet_no -= 1  # since we want to map punnets numbers to rows in the matrix
                if punnet_no in punnets_info:
                    continue  # information for this punnet already taken! Skip it!
                    # Please note: maturity (aka labels) for each punnet is
                    # exactly the same of each raspberry extracted from that punnet!

                punnets_info.setdefault(punnet_no, dict())
                punnets_info[punnet_no]['variety'] = variety.lower()
                punnets_info[punnet_no]['class'] = label

    return punnets_info

def make_plot(X, Y, punnets_info, fig_filename, title, s=50, ):
    # annotate=False, sample_names=None):
    """
    generates and shows a scatter plot

    Parameters
    ----------
    X : numpy.ndarray
        values of x

    Y : numpy.ndarray
        values of y

    punnets_info : dict
        The dictionary containing metadata for each punnets (i.e. variety and labels)

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
    classes = set(COLORS_MAP.keys())
    for variety in MARKERS_MAP:
        marker = MARKERS_MAP[variety]
        X_sub = [r for (j, r) in enumerate(X) if punnets_info[j]['variety'] == variety]
        Y_sub = [r for (j, r) in enumerate(Y) if punnets_info[j]['variety'] == variety]
        for (i, cla) in enumerate(classes):
            xc = [p for (j, p) in enumerate(X_sub) if punnets_info[j]['class'] == cla]
            yc = [p for (j, p) in enumerate(Y_sub) if punnets_info[j]['class'] == cla]
            cols = [COLORS_MAP[cla] for _ in range(len(xc))]
            plt.scatter(xc, yc, s=s, marker=marker, c=cols, label=cla)


            # if sample_names:
            #     nc = [p for (j, p) in enumerate(sample_names) if classes[j] == cla]
            # else:
            #     nc = None
            # if annotate and nc:
            #     for j, txt in enumerate(nc):
            #         plt.annotate(txt, (xc[j], yc[j]))

    plt.legend(loc=0)
    plt.savefig(fig_filename)
    plt.show()
    plt.clf()


if __name__ == '__main__':

    X = np.loadtxt(DATA_FILE)
    SpectrumTSNE = TSNE(n_components=2, init='pca', random_state=0,
                        perplexity=30.0)

    X_tsne = SpectrumTSNE.fit_transform(X)

    # tsne_output_filename = compose_output_filename(classes=['red'],
    #                                                filename_prefix='tsne_spectra',
    #                                                perpl=SpectrumTSNE.perplexity,
    #                                                ncomp=SpectrumTSNE.n_components,
    #                                                instr=SpectrumTSNE.init)
    # if not os.path.exists(tsne_output_filename):
    #     X_tsne = SpectrumTSNE.fit_transform(X_scaled)
    #     np.savetxt(tsne_output_filename, X_tsne)
    # else:
    #     X_tsne = np.loadtxt(tsne_output_filename)

    SpectrumMDS = MDS(n_components=2)

    # mds_output_filename = compose_output_filename(classes=['red'],
    #                                                filename_prefix='tsne_spectra',
    #                                                perpl=SpectrumTSNE.perplexity,
    #                                                ncomp=SpectrumTSNE.n_components,
    #                                                instr=SpectrumTSNE.init)

    X_mds = SpectrumMDS.fit_transform(X)

    punnets_info = collect_info_from_punnets(SO4_IMAGES_FOLDER)

    make_plot(X_tsne[:, 0], X_tsne[:, 1], punnets_info,
              fig_filename='tsne.pdf',
              title='tSNE on Spectrum - Init: {} Perplexity: {}'.format(
                  SpectrumTSNE.init, SpectrumTSNE.perplexity))

    make_plot(X_mds[:, 0], X_mds[:, 1], punnets_info,
              fig_filename='mds.pdf',
              title='MDS on Spectrum')