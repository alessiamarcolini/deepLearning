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
from mcc_multiclass import multimcc, confusion_matrix


def load_im2(paths):
    l = []
    for name in paths:
        result = cv2.imread(name)
        im2 = cv2.resize(result, (224, 224)).astype(np.float32)
        im2[:,:,0] -= 103.939
        im2[:,:,1] -= 116.779
        im2[:,:,2] -= 123.68
        im2 = im2.transpose((2,0,1))
        #im2 = np.expand_dims(im2, axis=0)
        #print(im2.shape)
        l.append(im2)
    return l


# path to the model weights files.
weights_path = 'vgg16_first_training_raspberry_weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224


validation_data_dir = 'datasets/so3/'


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

model.add(Flatten(input_shape=model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)

assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
model.load_weights(weights_path)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model

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


validation_images = os.listdir(validation_data_dir)
validation_images.sort()

for i in range(len(validation_images)):
    validation_images[i] = validation_data_dir + validation_images[i]


validation = np.array(load_im2(validation_images))

predicted_labels = model.predict(validation)
predicted_labels_linear = []

prediction_summary = open("vgg16_first_train_raspberry_prediction_new.csv", "w")
prediction_summary.write("\t".join(['FILENAME', 'PREDICTED_LABELS', 'E', 'G', 'L'])+'\n')


for i in range(len(predicted_labels)):
    cls_prob = predicted_labels[i]
    predicted_labels_linear.append(np.argmax(cls_prob))
    if predicted_labels_linear[i] == 0:
        predicted_label = "Early"
    elif predicted_labels_linear[i] == 1:
        predicted_label = "Good"
    elif predicted_labels_linear[i] == 2:
        predicted_label = "Late"
    line = [validation_images[i], predicted_label, str(round(cls_prob[0],3)), str(round(cls_prob[1],3)), str(round(cls_prob[2],3))]

    prediction_summary.write(";".join(line)+"\n")
    prediction_summary.flush()

prediction_summary.flush()
prediction_summary.close()
