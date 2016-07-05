"""Generates the t-SNE feature reduction for
 the two Deep Learning Networks predicting
 features from several datasets.
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

import os
from tsne.plotting import make_plot
from sklearn.manifold import TSNE
import numpy as np
from joblib import Parallel, delayed

from argparse import ArgumentParser

PREDICTED_FEATURES_FOLDER = os.path.join(os.path.abspath(os.path.curdir), 'predicted_features')
VALIDATION_LABELS_FOLDER = os.path.join(os.path.abspath(os.path.curdir), 'validation_labels')

REFERENCE_CLASSES = ['EARLY', 'GOOD', 'LATE']


def collect_data_files(target_folder=PREDICTED_FEATURES_FOLDER,
                       split_pattern='_predicted_features_'):
    """Collect all the image matrix file saved after
    predictions of Deep Learning models

    Parameters
    ----------
    target_folder : str (default: `PREDICTED_FEATURES_FOLDER`)
        The folder to process to collect all the
        matrix files. These files will be named
        according to the DL model and the corresponding
        dataset to be used. Each collected file is assumed
        to be a textual file saved by `numpy.savetxt`
        function.

    split_pattern : str (default: '_predicted_features_'
        The string pattern to be used to split gathered
        filenames in order to collect model names and
        corresponding dataset.

    Returns
    -------
    dict
        A dictionary mapping dataset and models to files.
        The output dictionary has `dataset_names` as keys and
        an additional dictionary as value, whose keys
        will be the names of the models associated to
        the corresponding file
    """

    dataset_maps = dict()
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file.endswith('.txt'):
                model_name, dataset_name_w_ext = file.split(split_pattern)
                dataset_name, _ = os.path.splitext(dataset_name_w_ext)
                dataset_maps.setdefault(dataset_name, dict())
                dataset_maps[dataset_name][model_name] = os.path.join(root, file)
    return dataset_maps


def apply_tsne(X, store_result=True, filename='tnse_result.txt',
               perplexity=30.0, init='random', n_components=2):
    """Apply t-SNE (t-distributed Stochastic Neighbor Embedding.) to input data
    configured with passed parameters. The implementation of the t-SNE
    is the one included in scikit-learn.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix `NxM` where N is the number of samples and M is the number
        of features.

    store_result : bool (default: True)
        If true, resulting t-sne matrix will be stored in a local file
        for further use.

    filename : str (default: `tsne_result.txt`)
        Name of the output file used to store t-SNE results.
        This parameter is used only in case `store_result` will be True.

    perplexity : float (default: 30.0)
        The `perplexity` parameter of the `sklearn.manifold.TSNE` model

    init : string (default: 'random')
        The `init` parameter of the `sklearn.manifold.TSNE` model for the
        initialization strategy.

    n_components : int (default: 2)
        The `n_components` parameter of the `sklearn.manifold.TSNE` model.

    Returns
    -------
    numpy.ndarray
        The results of the `fit_transform` method of the `TNSE` model
        applied to input data.

    """

    RaspberryTSNE = TSNE(n_components=n_components, perplexity=perplexity,
                         init=init, random_state=0)
    X_tsne = RaspberryTSNE.fit_transform(X)
    if store_result:
        np.savetxt(filename, X_tsne)
    return X_tsne


def load_and_apply_tsne(matrix_filepath, tsne_filepath):
    """Load the matrix from the `matrix_filepath` and apply
    the t-SNE algorithm, storing results to the `tsne_filepath`

    Parameters
    ----------
    matrix_filepath : str
        Path to the data matrix file

    tsne_filepath : str
        Path to the file where to store t-SNE computation
    """
    print('Executing t-SNE on {}'.format(matrix_filepath))
    X = np.loadtxt(matrix_filepath)
    apply_tsne(X, store_result=True, filename=tsne_filepath)


if __name__ == '__main__':

    parser = ArgumentParser(usage='''This scripts may run in two modes: 'tsne' and 'plot'.
                                    Use the --mode option to decide''')
    parser.add_argument('--mode', help='Execution Mode', choices=['tsne', 'plot'], dest='exec_mode')
    args = parser.parse_args()

    if args.exec_mode == 'tsne':
        print('Execution Mode: t-SNE')
        dataset_features_map = collect_data_files(target_folder=PREDICTED_FEATURES_FOLDER,
                                                  split_pattern='_predicted_features_')

        dl_models = set(mname for ds in dataset_features_map for mname in dataset_features_map[ds])
        print('Found a total of {} datasets for {} models'.format(len(dataset_features_map),
                                                                  len(dl_models)))
        print('\t Deep Learning Models: {}'.format(dl_models))
        # Preparing data to load data and apply t-SNE
        process_data = list()
        for dataset_name in dataset_features_map:
            for model_name in dataset_features_map[dataset_name]:
                matrix_filepath = dataset_features_map[dataset_name][model_name]
                tsne_filename = 'tsne_{}_{}.txt'.format(model_name, dataset_name)
                tsne_filepath = os.path.join(os.path.abspath(os.path.curdir),
                                             'tsne_data', tsne_filename)
                process_data.append((matrix_filepath, tsne_filepath))
        Parallel(n_jobs=-1, backend='threading')(
            delayed(load_and_apply_tsne)(mfp, tsnefp) for mfp, tsnefp in process_data)
    else:
        print('Execution Mode: Plot')
        labels_features_map = collect_data_files(target_folder=VALIDATION_LABELS_FOLDER,
                                                 split_pattern='_validation_labels_')



