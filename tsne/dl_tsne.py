#importiamo cose
import os
import cv2
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from matplotlib import pyplot
from sklearn.manifold import TSNE

# path to the model weights files.
WEIGHTS_PATH = 'vgg16_weights.h5'

#general path
IMAGE_PATH = '/data/webvalley/fruit_images/'
#fruits paths
RASPBERRY = 'BerryImageNet'
STRAWBERRY = 'strawberry'
BLACKBERRY = 'blackberry'
RED_CURRANT = 'redcurrant'
WHITE_CURRANT ='whitecurrant'
BLUEBERRY = 'blueberry'
CHERRY = 'cherry'
PLUM = 'susine'
APRICOT = 'apricot'
GOOSEBERRY = 'Gooseberry'

MARKERS = {RASPBERRY: 'o',
           STRAWBERRY: '+',
           BLACKBERRY: '*',
           RED_CURRANT: 'D',
           WHITE_CURRANT: 'h',
           BLUEBERRY: 's',
           CHERRY: 'b',
           PLUM: '8',
           APRICOT: 'p',
           GOOSEBERRY: '<'}

COLOURS = {RASPBERRY: '#ff6666',
           STRAWBERRY: '#794044',
           BLACKBERRY: '#000',
           RED_CURRANT: '#f03939',
           WHITE_CURRANT: '#f0f688',
           BLUEBERRY: '#3a539b',
           CHERRY: '#f688b4',
           PLUM: '#9615f0',
           APRICOT: '#f0b015',
           GOOSEBERRY: '#15f024'}

# dimensions of our images.
IMG_WIDTH, IMG_HEIGHT = 224, 224

#output array file
OUTPUT_IMAGES_FILE_PREFIX = "images_matrix"
DEFAULT_IMAGES_FILEMANE = OUTPUT_IMAGES_FILE_PREFIX + ".txt"

def make_plot(X, Y, colours, classes, sample_names,
              fig, title, s=10, marker='o', annotate=False):
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

    marker : str (default='o')
        shape of the marker

    colours : list
        colour of the markers of the different classes

    classes : list
        predicted classes of the values

    sample_names : list
        list of labels for each dot (only if annotate=True)

    fig : str
        name of the image file of the graph saved

    title : str
        title of the plot

    annotate : bool
        title of the plot


    """
    pyplot.title(title)
    for (i, cla) in enumerate(set(classes)):
        xc = [p for (j, p) in enumerate(X) if classes[j] == cla]
        yc = [p for (j, p) in enumerate(Y) if classes[j] == cla]
        nc = [p for (j, p) in enumerate(sample_names) if classes[j] == cla]
        cols = [c for (j, c) in enumerate(colours) if classes[j] == cla]
        pyplot.scatter(xc, yc, s=s, marker=marker, c=cols, label=cla)

        if annotate:
            for j, txt in enumerate(nc):
                pyplot.annotate(txt, (xc[j], yc[j]))
    pyplot.legend(loc=4)
    pyplot.savefig(fig)
    pyplot.show()
    pyplot.clf()


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


def collect_images(path):
    """


    Returns
    -------

    """
    input_images = []
    classes = []
    markers = []
    colors = []
    sample_names = []
    for root, dirs, files in os.walk(path):
        _, class_name = os.path.split(root)
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in ['.jpg', '.png', '.gif', '.jpeg']:
                input_images.append(os.path.join(root, file))
                classes.append(class_name)
                markers.append(MARKERS[class_name])
                colors.append(COLOURS[class_name])
                sample_names.append(name)

    return input_images, sample_names, classes, markers, colors

def compose_matrix_filename(classes):
    """
    compose the name of file name with the classes

    Parameters
    ----------
    classes: list
        the list of the classes for all the collected images

    Returns
    -------
    matrix_filename: str
        the name of the output matrix file

    """
    classes_set = set(classes)
    matrix_filename = OUTPUT_IMAGES_FILE_PREFIX
    for class_name in classes_set:
        matrix_filename += class_name[:5]+"_"
    matrix_filename += ".txt"
    return matrix_filename


def predict_images(input_images, save_txt=True, file_name=DEFAULT_IMAGES_FILEMANE):
    """
    Takes the images from directory and predicts them.
    It saves the output if save_txt=True

    Parameters
    ----------
    input_images : str
        directory of the images

    save_txt : bool (default=True)
        if True saves the prediction into a .txt file

    file_name : str
        the name of the output file

    Returns
    -------
    output_images : list

    """
    output_images = []
    for image_file_path in input_images:
        im = cv2.resize(cv2.imread(image_file_path), (224, 224)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        output = model.predict(im)
        output_images.append(list(output[0]))
    print('loading and predicting completed')
    output_images = np.array(output_images)
    output_images = output_images.astype(np.float64)

    if save_txt:
        np.savetxt(file_name, output_images)

    return output_images


if __name__ == '__main__':
    #collect raspberries images


    # collect grapes images
    print('collecting images:')
    input_images, sample_names, classes, markers, colours = collect_images(IMAGE_PATH)
    print('Collected %d images' % len(input_images))
    if not len(input_images):
        print('no image')
        exit()


    images_fruit_filename = compose_matrix_filename(classes)
    if os.path.exists(images_fruit_filename):
        print("image file exists, loading from file")
        output_images_fruits = np.loadtxt(images_fruit_filename)
        print("load completed")
        print("output images matrix shape: ", output_images_fruits.shape)
    else:
        print("image file not exists, predicting")

        # build the VGG16 network
        print('building VGG_16 network')
        model = VGG_16(weights_path=WEIGHTS_PATH, add_fully_connected=False)

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        print('sgd done')
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        # predicting images
        print('prediction')
        output_images_fruits = predict_images(input_images, file_name=images_fruit_filename)

        print("prediction completed")
        print("output images matrix shape: ", output_images_fruits.shape)


    print('Doing tsne model')
    Y_tsne_model = TSNE(n_components=2, random_state=0)

    print('apply tsne model to output_image of the fruits')
    Y_tsne_fruits = Y_tsne_model.fit_transform(output_images_fruits)
    print('tsne completed')


    #plottiamo roba
    plot_name= "tsne_comp%d_perplexity%d.pdf" % (Y_tsne_model.n_components, Y_tsne_model.perplexity)
    make_plot(X=Y_tsne_fruits[:, 0], Y=Y_tsne_fruits[:, 1], s=10, marker=markers,
              colours=colours,
              classes=classes, sample_names=sample_names, fig=plot_name,
              title="Tsne of different fruits")

