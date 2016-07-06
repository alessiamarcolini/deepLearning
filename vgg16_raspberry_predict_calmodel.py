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
import argparse

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
parser.add_argument('--fc_model', dest='FC_MODEL', type=str, choices=['tom', 'cal'], default='tom', help='Fully connected model on top (default: %(tom)s)')
parser.add_argument('--fc_weights', dest='FC_WEIGHTS',type=str, default=None, help='Fully Connected on top Weights')
parser.add_argument('--dataset', dest='DATASET',type=str, default=None, help='Input directory of the images')
args = parser.parse_args()
FC_MODEL = args.FC_MODEL
FC_WEIGHTS = args.FC_WEIGHTS
DATASET = args.DATASET


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


def dataset_to_parameters(dataset):
    validation_data_dir = ("/".join(["datasets", dataset.lower()]))
    validation_data_dir += "/"
    if len(os.listdir(validation_data_dir)) == 3:
        predict_mcc = True
    else:
        predict_mcc= False

    return predict_mcc, validation_data_dir

def vgg16(weights_path=None, add_fully_connected=True, fc_model = 'cal'):
    img_width, img_height = 224, 224

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

    if fc_model == 'cal':
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
        model.add(Dense(768, activation='sigmoid'))
        model.add(Dropout(0.0))
    elif fc_model == 'tom':
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.0))
    else:
        print ("fc_model option invalid. Exiting")
        sys.exit(1)

    top_model = None
    if add_fully_connected:
        top_model = Sequential()
        if fc_model == 'cal':
            top_model.add(Dense(3, input_dim = model.output_shape[1], activation='sigmoid'))
        elif fc_model == 'tom':
            top_model.add(Dense(3, input_dim = model.output_shape[1], activation='softmax'))
        else:
            print ("fc_model option invalid. Exiting")
            sys.exit(1)

    return model, top_model




def create_validationImg_validationLabel_list(predict_mcc, validation_data_dir):
    validation_images = []

    if predict_mcc:

        validation_labels = []

        val_path_e = validation_data_dir + "early/"
        val_path_g = validation_data_dir + "good/"
        val_path_l = validation_data_dir + "late/"
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

    else:
        validation_images = os.listdir(validation_data_dir)
        validation_images.sort()
        for i in range(len(validation_images)):
            validation_images[i] = validation_data_dir + validation_images[i]
        validation_labels = None


    return validation_images, validation_labels

def file_generator(predict_mcc, validation_images, validation_labels, predicted_labels):

    lines = []
    predicted_labels_linear = []
    validation_labels_linear = []

    for i in range(len(predicted_labels)):
        cls_prob = predicted_labels[i]

        predicted_labels_linear.append(np.argmax(cls_prob))

        if predicted_labels_linear[i] == 0:
            predicted_label = "Early"
        elif predicted_labels_linear[i] == 1:
            predicted_label = "Good"
        elif predicted_labels_linear[i] == 2:
            predicted_label = "Late"

        line = [validation_images[i], predicted_label, str(round(cls_prob[0],3)),
                str(round(cls_prob[1],3)), str(round(cls_prob[2],3))]

        if predict_mcc:

            for j in range(len(validation_labels[i])):
                cl = validation_labels[i][j]

                if cl == 1 and j == 0:
                    real_label = "Early"
                    validation_labels_linear.append(j)

                elif  cl == 1 and j == 1:
                    real_label = "Good"
                    validation_labels_linear.append(j)

                elif  cl == 1 and j == 2:
                    real_label = "Late"
                    validation_labels_linear.append(j)

            line.append(real_label)

        lines.append(";".join(line)+"\n")


    validation_labels_linear = np.array(validation_labels_linear)
    predicted_labels_linear = np.array(predicted_labels_linear)

    return lines, validation_labels_linear, predicted_labels_linear


def MCC_CM_calculator(validation_labels_linear, predicted_labels_linear):
    #Return MCC and confusion matrix

    MCC = multimcc(validation_labels_linear, predicted_labels_linear)
    MCC = round(MCC,3)
    MCC_line = "MCC=" + str(MCC)

    CM = confusion_matrix(validation_labels_linear, predicted_labels_linear)

    CM_lines = ";p_E;p_G;p_L\n"

    for i in range(len(CM[0])):
        if i == 0:
            l = "r_E"
        elif i == 1:
            l = "r_G"
        elif i == 2:
            l = "r_L"

        CM_lines += l + ";"
        for j in CM[0][i]:
            CM_lines += str(j) + ";"
        CM_lines += "\n"

    return MCC_line, CM_lines


def main():
    file_lines = []
    dataset = DATASET
    # path to the model weights files.
    # weights_path = 'weights/vgg16_weights_calmodel.h5'
    weights_path = FC_WEIGHTS

    # dimensions of our images.

    predict_mcc, validation_data_dir = dataset_to_parameters(dataset)

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename
    model, top_model = vgg16(weights_path, fc_model=FC_MODEL)

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(len(f.attrs['layer_names'])):
        g = f[f.attrs['layer_names'][k]]
        weights = [g[g.attrs['weight_names'][p]] for p in range(len(g.attrs['weight_names']))]
        if k >= len(model.layers):
            top_model.layers[k-len(model.layers)].set_weights(weights)
        else:
            model.layers[k].set_weights(weights)
    f.close()
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
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4, epsilon=1e-08),
                  metrics=['accuracy'])


    validation_images, validation_labels = create_validationImg_validationLabel_list(predict_mcc, validation_data_dir)
    validation = np.array(load_im2(validation_images))

    np.savetxt("tsne/validation_labels/cal1_validation_labels_{}.txt".format(dataset), validation_labels)

    predicted_features = model.predict(validation)
    np.savetxt("tsne/predicted_features/cal1_predicted_features_{}.txt".format(dataset), predicted_features)

    top_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4, epsilon=1e-08),
                  metrics=['accuracy'])

    predicted_labels = top_model.predict(predicted_features)

    prediction_summary = open("results/vgg16_calmodel_prediction_summary_{}.csv".format(dataset), "w")



    lines, validation_labels_linear, predicted_labels_linear = file_generator(predict_mcc, validation_images, validation_labels, predicted_labels)


    if predict_mcc:
        MCC_line, CM_lines = MCC_CM_calculator(validation_labels_linear, predicted_labels_linear)
        file_lines.append(MCC_line)
        file_lines.append(CM_lines)

    file_lines.append("\t".join(['FILENAME', 'PREDICTED_LABEL', 'E', 'G', 'L', 'REAL_LABEL'])+'\n')


    file_lines.extend(lines)

    for line in file_lines:
        prediction_summary.write(line)
    prediction_summary.close()


if __name__ == "__main__":
    main()
