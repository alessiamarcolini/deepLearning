import os
import h5py
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from os.path import join, getsize
import sys
from mcc_multiclass import multimcc
# import keras.backend.tensorflow_backend as K
import argparse
import random
from keras.utils import np_utils
from keras.utils.visualize_util import plot


def vgg16_predict(weights_path=None, img_width=224, img_height=224, fc_model=None, f_type=None, n_labels=None):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())

    print('#Model loaded.')

    loss = None
    optimizer = None
    top_model = Sequential()
    if fc_model == 'cal':
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        top_model.add(Dense(n_labels, input_dim=model.output_shape[1], activation='sigmoid'))
        loss = 'categorical_crossentropy'
        optimizer = optimizers.Adam(lr=1e-4, epsilon=1e-08)
    elif fc_model == 'tom':
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        top_model.add(Dense(n_labels, input_dim=model.output_shape[1], activation='softmax'))
        loss = 'categorical_crossentropy'
        optimizer = optimizers.Adam(lr=1e-4, epsilon=1e-08)
    elif fc_model == 'am':
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        top_model.add(Dense(n_labels, input_dim=model.output_shape[1], activation='sigmoid'))
        loss = 'binary_crossentropy'
        optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    # model.load_weights(weights_path)
    f = h5py.File(weights_path)
    for k in range(len(f.attrs['layer_names'])):
        g = f[f.attrs['layer_names'][k]]
        weights = [g[g.attrs['weight_names'][p]] for p in range(len(g.attrs['weight_names']))]
        if k >= len(model.layers):
            top_model.layers[k - len(model.layers)].set_weights(weights)
        else:
            model.layers[k].set_weights(weights)
    f.close()

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    if f_type == 'f31':
        for layer in model.layers[:32]:
            layer.trainable = False
    elif f_type == 'f24':
        for layer in model.layers[:25]:
            layer.trainable = False
    elif f_type == 'f17':
        for layer in model.layers[:18]:
            layer.trainable = False
    elif f_type == 'f10':
        for layer in model.layers[:11]:
            layer.trainable = False
    elif f_type == 'f5':
        for layer in model.layers[:6]:
            layer.trainable = False
    elif f_type == 'f0':
        pass

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    top_model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

    return model, top_model


def load_im2(paths):
    l = []
    for name in paths:
        im2 = cv2.resize(cv2.imread(name), (224, 224)).astype(np.float32)
        im2[:, :, 0] -= 103.939
        im2[:, :, 1] -= 116.779
        im2[:, :, 2] -= 123.68
        im2 = im2.transpose((2, 0, 1))
        # im2 = np.expand_dims(im2, axis=0)
        # print(im2.shape)
        l.append(im2)
    return np.array(l)


def parse_validation_list(validation_file_list=None):
    map = np.loadtxt(validation_file_list, dtype=str)
    images_path = []

    for record in map:
        images_path.append(record.strip())

    images_array = load_im2(images_path)

    return images_array, images_path


class MyArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(MyArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg


parser = MyArgumentParser(
    description='Run a training experiment using pretrained VGG16, specified on the Raspberry DataSet.',
    fromfile_prefix_chars='@')
parser.add_argument('--vgg16_weights', type=str, default='vgg16_weights.h5', help='VGG16 PreTrained weights')
parser.add_argument('--output_dir', dest='OUTDIR', type=str, default="./experiment_output/", help='Output directory')
parser.add_argument('--validation_file_list', dest='VALIDATION_FILE_LIST', type=str,
                    help='File list of validation images (with path)')
parser.add_argument('--validation_labels', dest='VALIDATION_LABELS', type=str,
                    help='List of validation labels of the images')
parser.add_argument('--n_labels', dest='N_LABELS', type=int, default=2, help='Number of labels to predict')
parser.add_argument('--fc_model', dest='FC_MODEL', type=str, choices=['tom', 'cal', 'am'], default='tom',
                    help='Fully connected model on top (default: %(default)s)')
parser.add_argument('--f_type', dest='F_TYPE', type=str, choices=['f0', 'f5', 'f10', 'f17', 'f24', 'f31'],
                    default='f24',
                    help='Layers to freeze: F0 = Freeze 0 layers, F24 = Freeze 24 layers (default: %(default)s)')
parser.add_argument('--plot', dest='PLOT', action='store_true', help='Produce plots of the model')

args = parser.parse_args()
VGG_WEIGHTS = args.vgg16_weights
FC_MODEL = args.FC_MODEL
VALIDATION_FILE_LIST = args.VALIDATION_FILE_LIST
VALIDATION_LABELS = args.VALIDATION_LABELS
N_LABELS = args.N_LABELS
F_TYPE = args.F_TYPE
PLOT = args.PLOT
OUTDIR = args.OUTDIR + "/"

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

if VALIDATION_FILE_LIST is not None:
    validation, validation_images = parse_validation_list(validation_file_list=VALIDATION_FILE_LIST)
else:
    print >> sys.stderr, "\nFile with path images to predict is mandatory."
    exit(1)

if VALIDATION_LABELS is not None:
    validation_labels = np.loadtxt(VALIDATION_LABELS)
    validation_labels = np_utils.to_categorical(validation_labels, nb_classes=N_LABELS)

print "\n\n#\tExperiment Setup"
print "#VGG_WEIGHTS:", VGG_WEIGHTS
print "#FC_MODEL:", FC_MODEL
print "#F_TYPE:", F_TYPE
print "#VALIDATION_FILE_LIST:", VALIDATION_FILE_LIST
print "#VALIDATION_LABELS:", VALIDATION_LABELS
print "#N_LABELS:", N_LABELS
print "#PLOT:", PLOT
print "#OUTDIR:", OUTDIR

print "\n\n\n"
print "#\tPerforming Predict on Validation"
#### PREDICT
model, top_model = vgg16_predict(weights_path=VGG_WEIGHTS, img_width=224, img_height=224, fc_model=FC_MODEL,
                                 f_type=F_TYPE, n_labels=N_LABELS)
if PLOT:
    plot(model, to_file=OUTDIR + "V_L_So_" + F_TYPE + "_" + FC_MODEL + "_plot.png", show_shapes=True)
predicted_features_train = model.predict(validation)
np.savetxt(OUTDIR + "V_L_So_" + F_TYPE + "_" + FC_MODEL + "_bottleneck_validation.txt", predicted_features_train)

predicted_labels_train = top_model.predict(predicted_features_train)
print "#Writing Prediction Output"
prediction_summary_train = open(OUTDIR + "V_L_So_" + F_TYPE + "_" + FC_MODEL + "_validation_summary.txt", "w")
prediction_summary_train.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

predicted_labels_linear = []

for i in range(len(predicted_labels_train)):
    cls_prob = [str(el) for el in predicted_labels_train[i]]
    if VALIDATION_LABELS is not None:
        real_label = np.argmax(validation_labels[i])
    else:
        real_label = 'NA'
    line = [validation_images[i], str(real_label), ";".join(cls_prob)]
    predicted_labels_linear.append(np.argmax(predicted_labels_train[i]))
    prediction_summary_train.write("\t".join(line) + "\n")
    prediction_summary_train.flush()

if VALIDATION_LABELS is not None:
    train_labels_linear = []

    for lbl in validation_labels:
        train_labels_linear.append(np.argmax(lbl))

    train_labels_linear = np.array(train_labels_linear)
    predicted_labels_linear = np.array(predicted_labels_linear)

    MCC = multimcc(train_labels_linear, predicted_labels_linear)
    print("#MCC Val:", MCC)
    prediction_summary_train.write("MCC: " + str(round(MCC, 3)))
prediction_summary_train.close()
