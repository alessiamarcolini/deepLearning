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
from joblib import Parallel, delayed
from argparse import ArgumentParser

PREDICTED_FEATURES_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'predicted_features')
TSNE_FEATURES_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tsne_data')
TSNE_SPLIT_PATTERN = 'tsne-init-{}-perp-{}'


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
                dataset_filepath = os.path.join(root, file)
                model_name, dataset_name_w_ext = file.split(split_pattern)
                dataset_name, _ = os.path.splitext(dataset_name_w_ext)
                dataset_maps.setdefault(model_name, dict())
                dataset_maps[model_name][dataset_name] = dataset_filepath
    return dataset_maps


def apply_tsne(X, store_result=True, filename='tnse_result.txt',
               perplexity=30.0, init='random', n_components=2,
               TSNE_model=None):
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

    TSNE_model: sklearn.manifold.TSNE (default: None)
        The instance of an existing t-SNE model already
        fitted to generate the corresponding embedding.
        If None (default), a new t-SNE model will be created
        and provided data (i.e. `X`) will be
        used to fit the model, and finally transformed in the
        resulting embedding.

    Returns
    -------
    sklearn.manifold.TSNE
        The t-SNE model created if `TSNE_model=None` parameter was provided
        (default value). Otherwise, a `None` is returned instead

    numpy.ndarray
        The results of the `fit_transform` method of the `TNSE` model
        applied to input data.

    Notes
    -----
    TSNE_model and (init, perplexity, n_components) parameters are
    mutually exclusive. In other words, if an existing TSNE_model
    is passed to the function, the other three parameters are
    discarded because useless. Otherwise, they will be used to create
    a **new** model which will be `fit` and `transform` on the passed
    `X` data.
    """
    if TSNE_model is None:
        RaspberryTSNE = TSNE(n_components=n_components, perplexity=perplexity,
                             init=init, random_state=0)
    else:
        RaspberryTSNE = TSNE_model
    X_tsne = RaspberryTSNE.fit_transform(X)
    if store_result:
        np.savetxt(filename, X_tsne)
    return RaspberryTSNE, X_tsne


def load_and_apply_tsne(matrix_filepath, tsne_filepath,
                        tsne_init='random', tsne_perplexity=30.0,
                        TSNE_model=None):
    """Load the matrix from the `matrix_filepath` and apply
    the t-SNE algorithm, storing results to the `tsne_filepath`

    Parameters
    ----------
    matrix_filepath : str
        Path to the data matrix file

    tsne_filepath : str
        Path to the file where to store t-SNE computation

    tsne_init : str (default: random)
        The initialisation strategy to apply for t-SNE Model

    tsne_perplexity: float (default: 30.0)
        The perplexity parameter for the t-SNE Model

    TSNE_model: sklearn.manifold.TSNE

    Returns
    -------
    sklearn.manifold.TSNE
        The instance of the t-SNE model created is returned, **only if**
        the `TSNE_model=None` parameter has been provided.
        Otherwise, `None` is returned instead.

    See Also
    --------
    Please see the `apply_tsne` function notes for further details.
    """

    print('Executing t-SNE on {}'.format(matrix_filepath))
    X = np.loadtxt(matrix_filepath)
    if TSNE_model is None:
        RaspberryTSNE, _ = apply_tsne(X, store_result=True, filename=tsne_filepath,
                                      init=tsne_init, perplexity=tsne_perplexity)
        return RaspberryTSNE
    else:
        RaspberryTSNE, _ = apply_tsne(X, store_result=True, filename=tsne_filepath,
                                      TSNE_model=TSNE_model)
        return None


def compose_tsne_filepath(tsne_init, tsne_perplexity, model_name, dataset_name):
    """Utility function to compose the path to the
    tsne_filepath according to provided parameters."""

    tsne_split_pattern = TSNE_SPLIT_PATTERN.format(tsne_init,
                                                   tsne_perplexity)
    tsne_filename = '{}_{}_{}.txt'.format(model_name, tsne_split_pattern,
                                          dataset_name)
    tsne_filepath = os.path.join(os.path.abspath(os.path.curdir),
                                 'tsne_data', tsne_filename)
    return tsne_filepath


if __name__ == '__main__':

    parser = ArgumentParser(usage='''This scripts may run in two modes: 'tsne' and 'plot'.
                                    Use the --mode option to decide''')
    parser.add_argument('--init', help='t-SNE Initialisation Strategy',
                        choices=['random', 'pca'], dest='tsne_init',default='pca')
    parser.add_argument('--perplexity', type=float, dest='tsne_perplexity',
                        default=30.0, help='t-SNE Initialisation Strategy')
    parser.add_argument('--training_set', type=str, dest='training_dataset_name',
                        default='so2_t', help='t-SNE Embedding Dataset')
    args = parser.parse_args()


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
        # First of all, check that there is a Training set for the current
        # model, otherwise SKIP IT!
        if not args.training_dataset_name in dataset_features_map[model_name]:
            print('Skipping Model: ', model_name)
            continue
        # Create the t-SNE Embedding
        training_set_matrix_filepath = dataset_features_map[model_name][args.training_dataset_name]
        tsne_filepath = compose_tsne_filepath(args.tsne_init, args.tsne_perplexity,
                                              model_name, args.training_dataset_name)

        if os.path.exists(tsne_filepath):
            print('Skipping: ', tsne_filepath, ' Existing!')
            continue

        RaspberryTSNE = load_and_apply_tsne(training_set_matrix_filepath, tsne_filepath,
                                            args.tsne_init, args.tsne_perplexity)
        for dataset_name in dataset_features_map[model_name]:
            if dataset_name == 'training_set':
                continue  # skip training set
            matrix_filepath = dataset_features_map[model_name][dataset_name]
            tsne_filepath = compose_tsne_filepath(args.tsne_init, args.tsne_perplexity,
                                                  model_name, dataset_name)
            if os.path.exists(tsne_filepath):
                print('Skipping: ', tsne_filepath, ' Existing!')
                continue
            process_data.append((matrix_filepath, tsne_filepath, RaspberryTSNE))

    Parallel(n_jobs=-1, backend='threading')(
        delayed(load_and_apply_tsne)(matrix_filepath=mfp, tsne_filepath=tsnefp,
                                     TSNE_model=tsne_model) for mfp, tsnefp, tsne_model in process_data)



