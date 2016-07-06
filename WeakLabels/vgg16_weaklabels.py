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


def vgg16_train(weights_path = None, img_width = 224, img_height = 224, fc_model = None,f_type = None, n_labels = None ):

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


    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')


    loss = None
    optimizer = None
    if fc_model == 'cal':
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        model.add(Dense(n_labels, activation='sigmoid'))
        loss = 'categorical_crossentropy'
        optimizer = optimizers.Adam(lr=1e-4, epsilon=1e-08)
        batch_size = 128
    elif fc_model == 'tom':
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        model.add(Dense(n_labels, activation='softmax'))
        loss = 'categorical_crossentropy'
        optimizer = optimizers.Adam(lr=1e-4, epsilon=1e-08)
        batch_size = 128
    elif fc_model == 'am':
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_labels, activation='sigmoid'))
        loss = 'binary_crossentropy'
        optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)
        batch_size = 16

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

    return model, batch_size

def vgg16_finetuning(weights_path = None, img_width = 224, img_height = 224, fc_model = None,f_type = None, n_labels = None ):

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

    loss = None
    optimizer = None
    last_layer = None
    if fc_model == 'cal':
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        last_layer = Dense(n_labels, activation='sigmoid')
        loss = 'categorical_crossentropy'
        optimizer = optimizers.Adam(lr=1e-4, epsilon=1e-08)
        batch_size = 128
    elif fc_model == 'tom':
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        last_layer = Dense(n_labels, activation='softmax')
        loss = 'categorical_crossentropy'
        optimizer = optimizers.Adam(lr=1e-4, epsilon=1e-08)
        batch_size = 128
    elif fc_model == 'am':
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        last_layer = Dense(n_labels, activation='sigmoid')
        loss = 'binary_crossentropy'
        optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)
        batch_size = 16

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    #model.load_weights(weights_path)
    f = h5py.File(weights_path)
    for k in range(len(f.attrs['layer_names'])):
       g = f[f.attrs['layer_names'][k]]
       weights = [g[g.attrs['weight_names'][p]] for p in range(len(g.attrs['weight_names']))]
       if k >= len(model.layers):
           break
       else:
           model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(last_layer)

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

    return model, batch_size

def vgg16_predict(weights_path = None, img_width = 224, img_height = 224, fc_model = None,f_type = None, n_labels = None ):
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

    print('Model loaded.')


    loss = None
    optimizer = None
    top_model = Sequential()
    if fc_model == 'cal':
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        top_model.add(Dense(n_labels, input_dim = model.output_shape[1], activation='sigmoid'))
        loss = 'categorical_crossentropy'
        optimizer = optimizers.Adam(lr=1e-4, epsilon=1e-08)
    elif fc_model == 'tom':
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        top_model.add(Dense(n_labels, input_dim = model.output_shape[1],activation='softmax'))
        loss = 'categorical_crossentropy'
        optimizer = optimizers.Adam(lr=1e-4, epsilon=1e-08)
    elif fc_model == 'am':
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        top_model.add(Dense(n_labels, input_dim = model.output_shape[1],activation='sigmoid'))
        loss = 'binary_crossentropy'
        optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    #model.load_weights(weights_path)
    f = h5py.File(weights_path)
    for k in range(len(f.attrs['layer_names'])):
       g = f[f.attrs['layer_names'][k]]
       weights = [g[g.attrs['weight_names'][p]] for p in range(len(g.attrs['weight_names']))]
       if k >= len(model.layers):
           top_model.layers[k-len(model.layers)].set_weights(weights)
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
        im2[:,:,0] -= 103.939
        im2[:,:,1] -= 116.779
        im2[:,:,2] -= 123.68
        im2 = im2.transpose((2,0,1))
        #im2 = np.expand_dims(im2, axis=0)
        #print(im2.shape)
        l.append(im2)
    return np.array(l)

def parse_mapping(mapping_file=None):
    map = np.loadtxt(mapping_file, dtype=str)
    labels = []
    images_path = []

    for record in map:
        images_path.append(record[0])
        labels.append(int(record[1]))

    images_array = load_im2(images_path)
    labels = np_utils.to_categorical(labels,nb_classes=np.max(labels)+1)

    return images_array, labels, images_path


class myArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(myArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg

parser = myArgumentParser(description='Run a training experiment using pretrained VGG16, specified on the Raspberry DataSet.',
        fromfile_prefix_chars='@')
# parser.add_argument('--gpu', type=int, default=0, help='GPU Device (default: %(default)s)')
parser.add_argument('--nb_epochs', type=int, default=10, help='Number of Epochs during training (default: %(default)s)')
# parser.add_argument('--random', action='store_true', help='Run with random sample labels')
parser.add_argument('--vgg16_weights', type=str, default='vgg16_weights.h5',help='VGG16 PreTrained weights')
parser.add_argument('--output_dir', dest='OUTDIR',type=str, default="./experiment_output/",help='Output directory')
# parser.add_argument('--input_dir', type=str, default="./",help='Input directory')
parser.add_argument('--weaklbl_training_map', dest='WEAKLABEL_TRAINING_MAP', type=str,help='Mapping file of training images (with path) and weak label')
parser.add_argument('--weaklbl_validation_map', dest='WEAKLABEL_VALIDATION_MAP', type=str,help='Mapping file of validation images (with path) and weak label')
parser.add_argument('--hard_training_map', dest='HARD_TRAINING_MAP', type=str,help='Mapping file of training images (with path) and hard label')
parser.add_argument('--hard_validation_map', dest='HARD_VALIDATION_MAP', type=str,help='Mapping file of validation images (with path) and hard label')
parser.add_argument('--fc_model', dest='FC_MODEL', type=str, choices=['tom', 'cal', 'am'], default='tom', help='Fully connected model on top (default: %(tom)s)')
parser.add_argument('--f_type', dest='F_TYPE', type=str, choices=['f0','f5', 'f10', 'f17' ,'f24','f31'], default='f24', help='Layers to freeze: F0 = Freeze 0 layers, F24 = Freeze 24 layers (default: %(tom)s)')

args = parser.parse_args()
# RANDOM_LABELS = args.random
NB_EPOCHS = args.nb_epochs
VGG_WEIGHTS = args.vgg16_weights
FC_MODEL = args.FC_MODEL
F_TYPE = args.F_TYPE
WEAK_TRAINING_MAP = args.WEAKLABEL_TRAINING_MAP
WEAK_VALIDATION_MAP = args.WEAKLABEL_VALIDATION_MAP
HARD_TRAINING_MAP = args.HARD_TRAINING_MAP
HARD_VALIDATION_MAP = args.HARD_VALIDATION_MAP
OUTDIR = args.OUTDIR +"/"

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

if WEAK_TRAINING_MAP is not None:
    weak_train, weak_train_labels, weak_train_images = parse_mapping(mapping_file = WEAK_TRAINING_MAP)
else:
    print >> sys.stderr, "\nWeakLabel Training map is mandatory."
    exit(1)

if WEAK_VALIDATION_MAP is not None:
    weak_validation, weak_validation_labels, weak_validation_images = parse_mapping(mapping_file = WEAK_VALIDATION_MAP)

if HARD_TRAINING_MAP is not None:
    hard_train, hard_train_labels, hard_train_images = parse_mapping(mapping_file = HARD_TRAINING_MAP)
else:
    print >> sys.stderr, "\nHardLabel Training map is mandatory."
    exit(1)

if HARD_VALIDATION_MAP is not None:
    hard_validation, hard_validation_labels, hard_validation_images = parse_mapping(mapping_file = HARD_VALIDATION_MAP)


nb_epochs = NB_EPOCHS

print "\n\n#\tExperiment Setup"
print "#NB_EPOCHS:", NB_EPOCHS
print "#VGG_WEIGHTS:",VGG_WEIGHTS
print "#FC_MODEL:",FC_MODEL
print "#F_TYPE:",F_TYPE
print "#WEAK_TRAINING_MAP:",WEAK_TRAINING_MAP
print "#WEAK_VALIDATION_MAP:",WEAK_VALIDATION_MAP
print "#HARD_TRAINING_MAP:",HARD_TRAINING_MAP
print "#HARD_VALIDATION_MAP:",HARD_VALIDATION_MAP
print "#OUTDIR:",OUTDIR

print "\n\n\n"
print "#\tStarting Training on Weak Labels"
#### TRAIN
model, batch_size = vgg16_train(weights_path=VGG_WEIGHTS, img_width=224, img_height=224, fc_model=FC_MODEL, f_type=F_TYPE, n_labels= weak_train_labels.shape[1])
model.fit(weak_train, weak_train_labels, nb_epoch=nb_epochs, batch_size=batch_size)
weak_weights_file = OUTDIR + "V_L_So_"+F_TYPE+"_"+FC_MODEL+"_weaklabels_weights.h5"
model.save_weights(weak_weights_file, overwrite=True)

print "\n#\tPerforming Predict on Training Weak Labels"
#### PREDICT
model, top_model = vgg16_predict(weights_path=weak_weights_file, img_width=224, img_height=224, fc_model=FC_MODEL, f_type=F_TYPE,n_labels= weak_train_labels.shape[1])
predicted_features_train = model.predict(weak_train)
np.savetxt(OUTDIR + "V_L_So_"+F_TYPE+"_"+FC_MODEL+"_weaklabels_bottleneck_train.txt", predicted_features_train)

predicted_labels_train = top_model.predict(predicted_features_train)
prediction_summary_train = open(OUTDIR + "V_L_So_" + F_TYPE + "_" + FC_MODEL + "_weaklabels_training_summary.txt", "w")
prediction_summary_train.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

predicted_labels_linear = []

for i in range(len(predicted_labels_train)):
    cls_prob = [str(el) for el in predicted_labels_train[i]]
    real_label = np.argmax(weak_train_labels[i])
    line = [weak_train_images[i], str(real_label), ";".join(cls_prob)]
    predicted_labels_linear.append(np.argmax(predicted_labels_train[i]))
    prediction_summary_train.write("\t".join(line) + "\n")
    prediction_summary_train.flush()

train_labels_linear = []

for lbl in weak_train_labels:
    train_labels_linear.append(np.argmax(lbl))

train_labels_linear = np.array(train_labels_linear)
predicted_labels_linear = np.array(predicted_labels_linear)

MCC = multimcc(train_labels_linear, predicted_labels_linear)
print("#MCC Val:", MCC)
prediction_summary_train.write("MCC: " + str(round(MCC, 3)))
prediction_summary_train.close()


if WEAK_VALIDATION_MAP is not None:
    print "\n"
    print "#\tPerforming Predict on Validation Weak Labels"
    predicted_features_validation = model.predict(weak_validation)
    np.savetxt(OUTDIR + "V_L_So_"+F_TYPE+"_"+FC_MODEL+"_weaklabels_bottleneck_validation.txt", predicted_features_validation)

    predicted_labels_validation = top_model.predict(predicted_features_validation)
    prediction_summary_validation = open(OUTDIR + "V_L_So_" + F_TYPE + "_" + FC_MODEL + "_weaklabels_validation_summary.txt", "w")
    prediction_summary_validation.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

    predicted_labels_linear = []

    for i in range(len(predicted_labels_validation)):
        cls_prob = [str(el) for el in predicted_labels_validation[i]]
        real_label = np.argmax(weak_validation_labels[i])
        line = [weak_validation_images[i], str(real_label), ";".join(cls_prob)]
        predicted_labels_linear.append(np.argmax(predicted_labels_validation[i]))
        prediction_summary_validation.write("\t".join(line) + "\n")
        prediction_summary_validation.flush()

    validation_labels_linear = []

    for lbl in weak_validation_labels:
        validation_labels_linear.append(np.argmax(lbl))

    validation_labels_linear = np.array(validation_labels_linear)
    predicted_labels_linear = np.array(predicted_labels_linear)

    MCC = multimcc(validation_labels_linear, predicted_labels_linear)
    print("#MCC Val:", MCC)
    prediction_summary_validation.write("MCC: " + str(round(MCC, 3)))
    prediction_summary_validation.close()

print "\n\n\n"
print "#\tStarting Fine Tuning on Hard Labels"
#### FINE TUNING
model, batch_size = vgg16_finetuning(weights_path=weak_weights_file, img_width=224, img_height=224, fc_model=FC_MODEL, f_type=F_TYPE, n_labels= hard_train_labels.shape[1])
model.fit(hard_train, hard_train_labels, nb_epoch=nb_epochs, batch_size=batch_size)
hard_weights_file = OUTDIR + "V_L_So_"+F_TYPE+"_"+FC_MODEL+"_hardlabels_weights.h5"
model.save_weights(hard_weights_file, overwrite=True)

print "\n#\tPerforming Predict on Training Hard Labels"
#### PREDICT on FINE TUNING
model, top_model = vgg16_predict(weights_path=hard_weights_file, img_width=224, img_height=224, fc_model=FC_MODEL, f_type=F_TYPE, n_labels= hard_train_labels.shape[1])
predicted_features_train = model.predict(hard_train)
np.savetxt(OUTDIR + "V_L_So_"+F_TYPE+"_"+FC_MODEL+"_hardlabels_bottleneck_train.txt", predicted_features_train)

predicted_labels_train = top_model.predict(predicted_features_train)
prediction_summary_train = open(OUTDIR + "V_L_So_" + F_TYPE + "_" + FC_MODEL + "_hardlabels_training_summary.txt", "w")
prediction_summary_train.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

predicted_labels_linear = []

for i in range(len(predicted_labels_train)):
    cls_prob = [str(el) for el in predicted_labels_train[i]]
    real_label = np.argmax(hard_train_labels[i])
    line = [hard_train_images[i], str(real_label), ";".join(cls_prob)]
    predicted_labels_linear.append(np.argmax(predicted_labels_train[i]))
    prediction_summary_train.write("\t".join(line) + "\n")
    prediction_summary_train.flush()

train_labels_linear = []

for lbl in hard_train_labels:
    train_labels_linear.append(np.argmax(lbl))

train_labels_linear = np.array(train_labels_linear)
predicted_labels_linear = np.array(predicted_labels_linear)

MCC = multimcc(train_labels_linear, predicted_labels_linear)
print("#MCC Val:", MCC)
prediction_summary_train.write("MCC: " + str(round(MCC, 3)))
prediction_summary_train.close()

if HARD_VALIDATION_MAP is not None:
    print "\n"
    print "#\tPerforming Predict on Validation Hard Labels"
    predicted_features_validation = model.predict(hard_validation)
    np.savetxt(OUTDIR + "V_L_So_"+F_TYPE+"_"+FC_MODEL+"_hardlabels_bottleneck_validation.txt", predicted_features_validation)

    predicted_labels_validation = top_model.predict(predicted_features_validation)
    prediction_summary_validation = open(OUTDIR + "V_L_So_" + F_TYPE + "_" + FC_MODEL + "_hardlabels_validation_summary.txt", "w")
    prediction_summary_validation.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

    predicted_labels_linear = []

    for i in range(len(predicted_labels_validation)):
        cls_prob = [str(el) for el in predicted_labels_validation[i]]
        real_label = np.argmax(hard_validation_labels[i])
        line = [hard_validation_images[i], str(real_label), ";".join(cls_prob)]
        predicted_labels_linear.append(np.argmax(predicted_labels_validation[i]))
        prediction_summary_validation.write("\t".join(line) + "\n")
        prediction_summary_validation.flush()

    validation_labels_linear = []

    for lbl in hard_validation_labels:
        validation_labels_linear.append(np.argmax(lbl))

    validation_labels_linear = np.array(validation_labels_linear)
    predicted_labels_linear = np.array(predicted_labels_linear)

    MCC = multimcc(validation_labels_linear, predicted_labels_linear)
    print("#MCC Val:", MCC)
    prediction_summary_validation.write("MCC: " + str(round(MCC, 3)))
    prediction_summary_validation.close()