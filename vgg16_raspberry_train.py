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
weights_path = 'vgg16_weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224
nb_epochs = int(sys.argv[1])

train_data_dir = 'BerryPhotos/train'
validation_data_dir = 'BerryPhotos/validation'
'''
nb_train_samples = 466
nb_train_early = nb_train_late = 112
nb_train_good = 242

nb_validation_samples = 198
nb_validation_early = nb_validation_late = 48
nb_validation_good = 102
nb_epoch = 50
'''
'''
l_images = []
for root, dirs, files in os.walk("/home/a-marcolini/deepLearning/BerryPhotos/V/Good/Individual/Cropped"):
    for name in files:
        l_images.append(os.path.join(root, name))

#path to the images of rasperries directory
#image_path = '/home/a-marcolini/Downloads/BerryImageNet'

classes_path = 'classes_final.csv'
'''

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
model.add(Flatten(input_shape=model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
#top_model.load_weights(top_model_weights_path)


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
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
        for name in train_filenames_e:
            train_images.append(path + name)
            train_labels.append([1,0,0])
    elif path == train_path_g:
        for name in train_filenames_g:
            train_images.append(path + name)
            train_labels.append([0,1,0])
    elif path == train_path_l:
        for name in train_filenames_l:
            train_images.append(path + name)
            train_labels.append([0,0,1])

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
            validation_labels.append([1,0,0])
    elif path == val_path_g:
        for name in val_filenames_g:
            validation_images.append(path + name)
            validation_labels.append([0,1,0])
    elif path == val_path_l:
        for name in val_filenames_l:
            validation_images.append(path + name)
            validation_labels.append([0,0,1])


train = np.array(load_im2(train_images))
#print(train)
validation = np.array(load_im2(validation_images))

# fit the model
model.fit(train, train_labels, nb_epoch=nb_epochs, batch_size=16)
model.save_weights("vgg16_first_training_raspberry_weights.h5", overwrite=True)
predicted_labels = model.predict(validation)

prediction_summary = open("vgg16_first_train_raspberry_prediction_summary.txt", "w")
prediction_summary.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS'])+'\n')

predicted_labels_linear = []

for i in range(len(predicted_labels)):
    cls_prob = predicted_labels[i]
    for j in range(len(validation_labels[i])):
        cl = validation_labels[i][j]
        if cl == 1 and j == 0:
            real_label = "Early"
            predicted_labels_linear.append(0)
        elif  cl == 1 and j == 1:
            real_label = "Good"
            predicted_labels_linear.append(1)
        elif  cl == 1 and j == 2:
            real_label = "Late"
            predicted_labels_linear.append(2)
    line = [validation_images[i], real_label, "Early:"+str(round(cls_prob[0],3))+";Good:"+str(round(cls_prob[1],3))+";Late:"+str(round(cls_prob[2],3))]
    prediction_summary.write("\t".join(line)+"\n")
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
