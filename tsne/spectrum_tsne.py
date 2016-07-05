"""Calculates t-SNE on Spectra data files
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from tsne.vgg16_imagenet_tsne import compose_output_filename
from tsne.plotting import make_plot

DATA_FILE = os.path.join(os.path.abspath(os.path.curdir), 'data',
                         'spectrum', 'allspectra_red.dat')

if __name__ == '__main__':

    X = np.loadtxt(DATA_FILE)
    X_scaled = scale(X, axis=1)  # scale samples

    SpectrumTSNE = TSNE(n_components=2, init='random', random_state=0,
                        perplexity=30.0)

    tsne_output_filename = compose_output_filename(classes=['red'],
                                                   filename_prefix='tsne_spectra',
                                                   perpl=SpectrumTSNE.perplexity,
                                                   ncomp=SpectrumTSNE.n_components,
                                                   instr=SpectrumTSNE.init)
    if not os.path.exists(tsne_output_filename):
        X_tsne = SpectrumTSNE.fit_transform(X_scaled)
        np.savetxt(tsne_output_filename, X_tsne)
    else:
        X_tsne = np.loadtxt(tsne_output_filename)

    make_plot(X_tsne[:, 0], X_tsne[:, 1], classes=['red'],
              colours=['r' for i in range(X_tsne[:, 0].size())],
              fig_filename=tsne_output_filename.replace('txt', 'pdf'),
              title='tSNE on Spectrum - Init: {} Perplexity: {}'.format(
                  SpectrumTSNE.init, SpectrumTSNE.perplexity))