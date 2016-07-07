"""Applies the VGG16 model and compute the t-SNE method for feature reduction
applied on a set of input images, corresponding to a specified set of
labelled classes.
"""

import os
import cv2
import h5py
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.manifold import TSNE
from argparse import ArgumentParser

# dimensions of our images (used in model building)
IMG_WIDTH, IMG_HEIGHT = 224, 224

# VGG16 Model weights filename
WEIGHTS_PATH = 'vgg16_first_training_raspberry_weights.h5'

IMAGE_PATH = '/data/webvalley/fruit_images/'

# Output filename templates
OUTPUT_TSNE_FILE_PREFIX = "tsne_matrix"
OUTPUT_IMAGES_FILE_PREFIX = "images_matrix"
DEFAULT_IMAGES_FILENAME = OUTPUT_IMAGES_FILE_PREFIX + ".txt"

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

CLASSES_OF_INTEREST = [RASPBERRY]


def VGG_16(weights_path=None, add_fully_connected=False):
    """
    generate the ANN model.
    It can have the fully connected layer

    Parameters
    ----------
    weights_path : string
        if True adds the fully connected layer to the model

    add_fully_connected : bool (default=False)
        if True adds the fully connected layer to the model

    Returns
    -------
    model : Sequential
        The ANN model

    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    if add_fully_connected:
        model.add(Dense(1000, activation='softmax'))

    if weights_path:
        # model.load_weights(weights_path)
        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)

        print('loading weights into the model')
        with h5py.File(weights_path) as weights_file:
            for k in range(weights_file.attrs['nb_layers']):
                if k >= len(model.layers):
                    # we don't look at the last (fully-connected) layers in the savefile
                    break
                g = weights_file['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                model.layers[k].set_weights(weights)
        print('loading weights completed')

    return model


def collect_images(path=IMAGE_PATH):
    """
    Collect the images contained in a given path. 
    The input path must correspond to the main folder containing all images
    organized into multiple folders, one per each class.

    Each sub-folder will be matched against the list of accepted classes.

    Returns
    -------
    input_images: list
        A list containing all the different input images gathered from folders
    sample_names: list
        A list containing names of each image.
    classes: list
        The list of all the classes associated to corresponding images
    colors: list
        The list of colors of each image (used in plots)

    """
    input_images = []
    classes = []
    colors = []
    sample_names = []
    for root, dirs, files in os.walk(path):
        _, class_name = os.path.split(root)
        if class_name in CLASSES_OF_INTEREST:
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() in ['.jpeg']:
                    input_images.append(os.path.join(root, file))
                    classes.append(class_name)
                    colors.append(COLOURS[class_name])
                    sample_names.append(name)

    return input_images, sample_names, classes, colors


def compose_output_filename(classes, filename_prefix=OUTPUT_IMAGES_FILE_PREFIX,
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


def predict_images(input_images, save_txt=True, filepath=DEFAULT_IMAGES_FILENAME):
    """
    Takes the images from dir_name and predicts them.
    It saves the output if save_txt=True

    Parameters
    ----------
    input_images : list
        dir_name of the images

    save_txt : bool (default=True)
        if True saves the prediction into a .txt file

    filepath : str
        the name of the output file

    Returns
    -------
    output_images : numpy.ndarray

    """
    output_images = []
    for image_file_path in input_images:
        #print('Processing Image: ', image_file_path)
        im = cv2.resize(cv2.imread(image_file_path), (224, 224)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        output = model.predict(im)
        output_images.append(list(output[0]))
    print('loading and predicting completed for a total of %d images' % (len(input_images)))
    output_images = np.array(output_images)
    output_images = output_images.astype(np.float32)

    if save_txt:
        np.savetxt(filepath, output_images)

    return output_images


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--with-tsne', type=bool, default=False,
                        help='Apply also t-SNE after predict', dest='with_tsne')
    args = parser.parse_args()
    # Preamble
    print('\n')
    print('='*80)
    
    # Step 1: Collect Fruit Images
    print('Step 1: Collecting images:')
    input_images, sample_names, classes, colours = collect_images()
    print('Collected %d images' % len(input_images))
    if not len(input_images):
        print('no image')
        exit()
    print('Step 1: Done!', end='\n\n')
    
    # First of all, instantiate t-SNE model and check if the matrix file already exists
    # (with the SAME configuration parameters)
    # Instantiate t-SNE Model
    
    # Step 2: Load Image Matrices from File OR Generate Image Matrices
    print('Step 2: Generate Image Matrices')
    images_fruit_filename = compose_output_filename(classes)
    images_fruit_filepath = os.path.join(os.path.abspath(os.path.curdir), images_fruit_filename)
    if os.path.exists(images_fruit_filepath):
        print("\t Image file exists, loading from file: ", images_fruit_filepath)
        output_images_fruits = np.loadtxt(images_fruit_filename)
        print("\t Load completed")
        print("\t Output images matrix shape: ", output_images_fruits.shape)
    else:
        print("\t Image file not exists, predicting: ", images_fruit_filepath)

        # build the VGG16 network
        print('\t Building VGG_16 network')
        model = VGG_16(weights_path=WEIGHTS_PATH, add_fully_connected=False)

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        print('\t Stochastic Gradient Descent Done')
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        # predicting images
        print('\t Starting Prediction!')
        output_images_fruits = predict_images(input_images, filepath=images_fruit_filepath)

    print("\t Prediction Completed for all %d Fruit Images" % (len(input_images)))
    print("\t Output Images Matrix Shape: ", output_images_fruits.shape)

    print('Step 2: Done!', end='\n\n')

    if args.with_tsne:
        # Step 3: t-SNE
        print('Step 3: t-SNE')

        Y_tsne_model = TSNE(n_components=2, init='random', random_state=0)

        tsne_matrix_filename = compose_output_filename(classes,
                                                       filename_prefix=OUTPUT_TSNE_FILE_PREFIX,
                                                       perpl=Y_tsne_model.perplexity,
                                                       ncomp=Y_tsne_model.n_components,
                                                       init_strategy=Y_tsne_model.init)
        tsne_matrix_filepath = os.path.join(os.path.abspath(os.path.curdir), tsne_matrix_filename)

        print("\t Image file not exists, calculating t-SNE: ", tsne_matrix_filepath)

        print('\t t-SNE model to output images')
        Y_tsne_fruits = Y_tsne_model.fit_transform(output_images_fruits)

        # Saving t-SNE resulting matrix
        np.savetxt(tsne_matrix_filepath, Y_tsne_fruits)

