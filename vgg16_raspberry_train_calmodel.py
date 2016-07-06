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
parser.add_argument('--gpu', type=int, default=0, help='GPU Device (default: %(default)s)')
parser.add_argument('--nb_epochs', type=int, default=10, help='Number of Epochs during training (default: %(default)s)')
parser.add_argument('--random', action='store_true', help='Run with random sample labels')
parser.add_argument('--vgg16_weights', type=str, default='vgg16_weights.h5',help='VGG16 PreTrained weights')
parser.add_argument('--output_dir', type=str, default="./experiment_output/",help='Output directory')
parser.add_argument('--input_dir', type=str, default="./",help='Input directory')

args = parser.parse_args()
GPU = args.gpu
RANDOM_LABELS = args.random
NB_EPOCHS = args.nb_epochs
OUTDIR = args.output_dir+"/"
INDIR = args.input_dir+"/"
VGG_WEIGHTS = args.vgg16_weights

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)


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
    return l


# path to the model weights files.
weights_path = VGG_WEIGHTS

# dimensions of our images.
img_width, img_height = 224, 224
nb_epochs = NB_EPOCHS

train_data_dir = INDIR+'BerryPhotos/train'
validation_data_dir = INDIR+'BerryPhotos/validation'

random.seed(0)

# with K.tf.device('/gpu:'+str(GPU)):
#     K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
    # build the VGG16 network
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

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)

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

# build a classifier model to put on top of the convolutional model
model.add(Flatten())
model.add(Dense(768, activation='sigmoid'))
model.add(Dropout(0.0))
model.add(Dense(768, activation='sigmoid'))
model.add(Dropout(0.0))
model.add(Dense(3, activation='sigmoid'))
# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4, epsilon=1e-08),
              metrics=['accuracy'])

train_images = []
train_labels = []
train_path_e = train_data_dir + "/early/"
train_path_g = train_data_dir + "/good/"
train_path_l = train_data_dir + "/late/"
train_paths = [train_path_e, train_path_g, train_path_l]

train_filenames_e = os.listdir(train_path_e)
train_filenames_g = os.listdir(train_path_g)
train_filenames_l = os.listdir(train_path_l)

for path in train_paths:
    if path == train_path_e:
        labels_array = [1, 0, 0]
        for name in train_filenames_e:
            train_images.append(path + name)
            if RANDOM_LABELS:
                labels_array = [0, 0, 0]
                rnd_cls = random.randint(0, 2)
                labels_array[rnd_cls] = 1
            train_labels.append(labels_array)
    elif path == train_path_g:
        labels_array = [0, 1, 0]
        for name in train_filenames_g:
            train_images.append(path + name)
            if RANDOM_LABELS:
                labels_array = [0, 0, 0]
                rnd_cls = random.randint(0, 2)
                labels_array[rnd_cls] = 1
            train_labels.append(labels_array)
    elif path == train_path_l:
        labels_array = [0, 0, 1]
        for name in train_filenames_l:
            train_images.append(path + name)
            if RANDOM_LABELS:
                labels_array = [0, 0, 0]
                rnd_cls = random.randint(0, 2)
                labels_array[rnd_cls] = 1
            train_labels.append(labels_array)

validation_images = []
validation_labels = []
val_path_e = validation_data_dir + "/early/"
val_path_g = validation_data_dir + "/good/"
val_path_l = validation_data_dir + "/late/"
val_paths = [val_path_e, val_path_g, val_path_l]

val_filenames_e = os.listdir(val_path_e)
val_filenames_g = os.listdir(val_path_g)
val_filenames_l = os.listdir(val_path_l)

for path in val_paths:
    if path == val_path_e:
        for name in val_filenames_e:
            validation_images.append(path + name)
            validation_labels.append([1, 0, 0])
    elif path == val_path_g:
        for name in val_filenames_g:
            validation_images.append(path + name)
            validation_labels.append([0, 1, 0])
    elif path == val_path_l:
        for name in val_filenames_l:
            validation_images.append(path + name)
            validation_labels.append([0, 0, 1])

train = np.array(load_im2(train_images))
# print(train)
validation = np.array(load_im2(validation_images))

# fit the model
model.fit(train, train_labels, nb_epoch=nb_epochs, batch_size=128)
model.save_weights(OUTDIR + "vgg16_first_training_raspberry_weights_calmodel.h5", overwrite=True)
predicted_labels = model.predict(validation)

prediction_summary = open(OUTDIR + "vgg16_first_train_raspberry_prediction_summary_calmodel.txt", "w")
prediction_summary.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

predicted_labels_linear = []

for i in range(len(predicted_labels)):
    cls_prob = predicted_labels[i]
    for j in range(len(validation_labels[i])):
        cl = validation_labels[i][j]
        if cl == 1 and j == 0:
            real_label = "Early"

        elif cl == 1 and j == 1:
            real_label = "Good"

        elif cl == 1 and j == 2:
            real_label = "Late"

    line = [validation_images[i], real_label,
            "Early:" + str(round(cls_prob[0], 3)) + ";Good:" + str(round(cls_prob[1], 3)) + ";Late:" + str(
                round(cls_prob[2], 3))]
    predicted_labels_linear.append(np.argmax(cls_prob))
    prediction_summary.write("\t".join(line) + "\n")
    prediction_summary.flush()

prediction_summary.close()

validation_labels_linear = []

for lbl in validation_labels:
    if lbl[0] == 1:
        validation_labels_linear.append(0)
    if lbl[1] == 1:
        validation_labels_linear.append(1)
    if lbl[2] == 1:
        validation_labels_linear.append(2)

validation_labels_linear = np.array(validation_labels_linear)
predicted_labels_linear = np.array(predicted_labels_linear)

MCC = multimcc(validation_labels_linear, predicted_labels_linear)
print(MCC)
