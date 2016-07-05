"""Generates the t-SNE feature reduction for
 the two Deep Learning Networks predicting
 features from several datasets.
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

import os
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
try:
    from .plotting import make_interactive_plot, RASPBERRY_BASE_CLASSES
except (ImportError, SystemError):
    from plotting import make_interactive_plot, RASPBERRY_BASE_CLASSES

from argparse import ArgumentParser

PREDICTED_FEATURES_FOLDER = os.path.join(os.path.abspath(os.path.curdir), 'predicted_features')
VALIDATION_LABELS_FOLDER = os.path.join(os.path.abspath(os.path.curdir), 'validation_labels')
TSNE_FEATURES_FOLDER = os.path.join(os.path.abspath(os.path.curdir), 'tsne_data')

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
                dataset_maps.setdefault(model_name, dict())
                dataset_maps[model_name][dataset_name] = os.path.join(root, file)
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


def load_tsne_data_for_plots(validation_filepath, model_name, dataset_name):
    """Load labels from input `validation_filepath` and transform them in a
    proper np.ndarray for further processing. Moreover, corresponding t-SNE
    data saved in a file are collected and returned as well.

    Parameters
    ----------
    validation_filepath : str
        The path to the validation file containing labels for the prediction
        for the corresponding model_name and dataset_name

    model_name: str
        The name of the Deep Learning model to which lables refer to.

    dataset_name: str
        The name of the corresponding dataset to which labels refer to.

    Returns
    -------
    labels : numpy.ndarray
        A numpy array containing labels (encoded as string) corresponding to
         different classes properly modified according to the input model
         and dataset

    classes: numpy.ndarray
        A numpy array containing only class labels (encoded as string)
        corresponding to different classes of samples loaded from
        input `validation_filepath`.

    X_tsne : numpy.ndarray
        The t-SNE data loaded from matrix file retrieved according to the
        `validation_filepath`.
    """

    # load labels from file (i.e. validation_filepath)
    labels_map = np.loadtxt(validation_filepath)

    # Generate Labels depending on the specific model and dataset
    class_label = '{label}_{mname}_{dsname}'
    ds_labels = np.array([class_label.format(label=label,
                                              mname=model_name,
                                              dsname=dataset_name)
                           for label in RASPBERRY_BASE_CLASSES])
    labels = ds_labels[np.argmax(labels_map, axis=1)]
    labels = labels.reshape(labels.shape[0], 1)  # reshaping to allow future np.vstack

    # Generate ONLY Class labels accordingly (used for markers in plots)
    class_only = '{label}'
    ds_classes = np.array([class_only.format(label=label)
                          for label in RASPBERRY_BASE_CLASSES])
    classes = ds_classes[np.argmax(labels_map, axis=1)]
    classes = classes.reshape(classes.shape[0], 1)  # reshaping to allow future np.vstack

    tsne_filepath = validation_filepath.replace(VALIDATION_LABELS_FOLDER, TSNE_FEATURES_FOLDER)
    tsne_filepath = tsne_filepath.replace('validation_labels', 'tsne')

    if not os.path.exists(tsne_filepath):
        print('ERROR: {} does not exist!'.format(tsne_filepath))
        return
    X_tsne = np.loadtxt(tsne_filepath)

    return labels, classes, X_tsne


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
        for model_name in dataset_features_map:
            for dataset_name in dataset_features_map[model_name]:
                matrix_filepath = dataset_features_map[model_name][dataset_name]
                tsne_filename = '{}_tsne_{}.txt'.format(model_name, dataset_name)
                tsne_filepath = os.path.join(os.path.abspath(os.path.curdir),
                                             'tsne_data', tsne_filename)
                process_data.append((matrix_filepath, tsne_filepath))
        Parallel(n_jobs=-1, backend='threading')(
            delayed(load_and_apply_tsne)(mfp, tsnefp) for mfp, tsnefp in process_data)
    else:
        print('Execution Mode: Plot')
        labels_features_map = collect_data_files(target_folder=VALIDATION_LABELS_FOLDER,
                                                 split_pattern='_validation_labels_')
        for model_name in labels_features_map:
            X_all = None
            labels_all = None
            classes_all = None
            for dataset_name in labels_features_map[model_name]:
                validation_filepath = labels_features_map[dataset_name][model_name]
                # Load lables and t-SNE data
                labels, classes, X_tsne = load_tsne_data_for_plots(validation_filepath,
                                                                   model_name, dataset_name)
                # Stack data accordingly
                if X_all is None:
                    X_all = X_tsne
                    labels_all = labels
                    classes_all = classes
                else:
                    X_all = np.vstack((X_all, X_tsne))
                    labels_all = np.vstack((labels_all, labels))
                    classes_all = np.vstack((classes_all, classes))
            # Compose the expected Pandas DataFrame
            data_dict = {'X': X_all[:, 0], 'Y': X_all[:, 1]}
            data_dict['labels'] = labels_all.ravel()
            data_dict['classes'] = classes_all.ravel()
            data = pd.DataFrame(data=data_dict)
            make_interactive_plot(data, fig_filename='tsne_plot_{}.html'.format(model_name))



