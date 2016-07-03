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
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
weights_path = 'vgg16_first_training_raspberry_weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224


validation_data_dir = 'BerrySamples_Blue'


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

#make arrays for MCC plotting
######################

mcc_list = []
perc_white = []

val_path_e = validation_data_dir + "/early/"
val_path_g = validation_data_dir + "/good/"
val_path_l = validation_data_dir + "/late/"
val_paths = [val_path_e, val_path_g, val_path_l]

val_filenames_e = os.listdir(val_path_e)
val_filenames_g = os.listdir(val_path_g)
val_filenames_l = os.listdir(val_path_l)

perc = list(range(0,16))
for i in range(len(perc)):
    perc[i] = str(perc[i])


p2 = ["16", "32", "64", "90", "128", "256", "512", "1024", "2048"]
perc.extend(p2)
print(perc)

for p in perc:

    validation_images = []
    validation_labels = []


    for path in val_paths:
        if path == val_path_e:
            for name in val_filenames_e:
                ls = name.split("_")
                if p == ls[2]:
                    validation_images.append(path + name)
                    validation_labels.append([1,0,0])
        elif path == val_path_g:
            for name in val_filenames_g:
                ls = name.split("_")
                if p == ls[2]:
                    validation_images.append(path + name)
                    validation_labels.append([0,1,0])
        elif path == val_path_l:
            for name in val_filenames_l:
                ls = name.split("_")
                if p == ls[2]:
                    validation_images.append(path + name)
                    validation_labels.append([0,0,1])



    validation = np.array(load_im2(validation_images))




    predicted_labels = model.predict(validation)

    prediction_summary = open("EXP/blue/vgg16_first_train_raspberry_prediction_summary_blue_" + p + ".csv", "w")
    prediction_summary.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS'])+'\n')

    predicted_labels_linear = []

    for i in range(len(predicted_labels)):
        cls_prob = predicted_labels[i]
        for j in range(len(validation_labels[i])):
            cl = validation_labels[i][j]
            if cl == 1 and j == 0:
                real_label = "Early"

            elif  cl == 1 and j == 1:
                real_label = "Good"

            elif  cl == 1 and j == 2:
                real_label = "Late"

        line = [validation_images[i], real_label, str(round(cls_prob[0],3)), str(round(cls_prob[1],3)), str(round(cls_prob[2],3))]
        predicted_labels_linear.append(np.argmax(cls_prob))
        prediction_summary.write(";".join(line)+"\n")
        prediction_summary.flush()



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
    mcc_list.append(MCC)
    perc_white.append(p)
    prediction_summary.write("MCC=" + str(MCC) + "\n")
    prediction_summary.flush()

    prediction_summary.write(str(confusion_matrix(validation_labels_linear, predicted_labels_linear)))
    prediction_summary.flush()
    prediction_summary.close()

mcc_list = np.array(mcc_list)
perc_white = np.array(perc_white)

fig = plt.figure()
ax = fig.add_subplot(111)
mcc_list_round = [round(i,3) for i in mcc_list]
j_old = 0.0
for i,j in zip(perc_white,mcc_list_round):
    if j_old != j:
        ax.annotate(str(j),xy=(i,j+0.01))
    j_old = j

plt.plot(perc_white, mcc_list, "-", color="red")
plt.plot(perc_white, mcc_list, "o", color="blue")
plt.ylabel("MCC")
plt.xlabel("Blue")
plt.ylim((-0.3,1))



#plt.xscale("log")
plt.savefig("plot_blue.png")
#plt.show()
