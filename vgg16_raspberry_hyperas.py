from __future__ import division

__author__ = 'zarbo'


def data():
    import numpy as np
    import os
    from images_utils import load_im2
    import pickle

    inputs = pickle.load(open('inputs.pickle', 'rb'))

    GPU = inputs['GPU']
    NB_EPOCHS = inputs['NB_EPOCHS']
    INDIR = inputs['INDIR']
    VGG_WEIGHTS = inputs['VGG_WEIGHTS']

    train_data_dir = INDIR + 'BerryPhotos/train'
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
                train_labels.append([1, 0, 0])
        elif path == train_path_g:
            for name in train_filenames_g:
                train_images.append(path + name)
                train_labels.append([0, 1, 0])
        elif path == train_path_l:
            for name in train_filenames_l:
                train_images.append(path + name)
                train_labels.append([0, 0, 1])
    train = np.array(load_im2(train_images))

    validation_data_dir = INDIR + 'BerryPhotos/validation'
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

    validation = np.array(load_im2(validation_images))

    return train, train_labels, validation, validation_labels, GPU, NB_EPOCHS, VGG_WEIGHTS


def model(train, train_labels, validation, validation_labels, GPU, NB_EPOCHS, VGG_WEIGHTS):
    import os
    import h5py
    from hyperas.distributions import choice
    from mcc_multiclass import multimcc
    #import keras.backend.tensorflow_backend as K
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
    from keras.layers import Dropout, Flatten, Dense
    from mcc_multiclass import multimcc
    from hyperopt import STATUS_OK
    from keras.optimizers import SGD, RMSprop, Adam

    # path to the model weights files.

    weights_path = VGG_WEIGHTS

    img_width, img_height = 224, 224
    nb_epochs = NB_EPOCHS
    print ("Entering GPU Model")
    #with K.tf.device('/gpu:' + str(GPU)):
    with open('FAKELOG',"w"):
        #K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
        #session = K.get_session()
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
        activation_function = 'relu'
        print "\n\t#Chosen Activation:", activation_function
        dense_size = 512
        print "\t#Chosen Dense Size:", dense_size
        dropout_rate = {{choice([0.0,0.25,0.5,0.75])}}
        print "\t#Chosen Dropout Rate:", dropout_rate
        model.add(Flatten())
        model.add(Dense(dense_size, activation=activation_function))
        model.add(Dropout(dropout_rate))
        if 'two' == 'two':
            print "\t#Chosen FC Size: Double"
            model.add(Dense(dense_size, activation=activation_function))
            model.add(Dropout(dropout_rate))
        else:
            print "\t#Chosen FC Size: Single"
        final_classifier = 'softmax'
        print "\t#Chosen Final Classifier:", final_classifier
        model.add(Dense(3, activation=final_classifier))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        # top_model.load_weights(top_model_weights_path)


        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in model.layers[:25]:
            layer.trainable = False

        trial_model_optimizer_dict = {}
        #trial_model_optimizer_list = {{choice(['rmsprop', 'adam', 'sgd','adagrad','adadelta','adamax'])}}
        trial_model_optimizer_list = {{choice([ 'adam', 'sgd'])}}
        print "\t#Chosen Optimizer: ", trial_model_optimizer_list
        epsilon = 1e-08
        lr = {{choice([1e-1, 1e-2,1e-3,1e-4,1e-5,1e-6,1e-7])}}
        momentum={{choice([0.7,0.8,0.9,1.0])}}
        nesterov = {{choice([True,False])}}
        if trial_model_optimizer_list == 'adam':
            print "\t\t#Chosen Epsilon:", epsilon

            print "\t\t#Chosen Learning Rate:", lr
            # beta_1 = {{uniform(0.5, 1)}}
            # beta_2 = {{uniform(0.6, 1)}}
            #trial_model_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon )
            trial_model_optimizer = Adam(lr=lr,epsilon=epsilon )
            trial_model_optimizer_dict['adam'] = {'lr': lr,
                                              'epsilon': epsilon}

        elif trial_model_optimizer_list == 'rmsprop':
            #epsilon={{choice([0,1e-04, 1e-05,1e-06,1e-07,1e-08, 1e-09, 1e-10])}}
            print "\t\t#Chosen Epsilon:", epsilon
            #lr = {{choice([0.1,0.5,0.01,0.05,0.001,0.005,0.0001,0.0005])}}

            print "\t\t#Chosen Learning Rate:", lr
            # rho = {{uniform(0.5, 1)}}
            #trial_model_optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon)
            trial_model_optimizer = RMSprop(lr=lr, epsilon=epsilon)
            trial_model_optimizer_dict['rmsprop'] = {'lr': lr,
                                              'epsilon': epsilon}

        elif trial_model_optimizer_list == 'sgd':

            print "\t\t#Chosen Nesterov:", nesterov
            #lr = {{choice([0.1,0.5,0.01,0.05,0.001,0.005,0.0001,0.0005])}}

            print "\t\t#Chosen Learning Rate:", lr

            print "\t\t#Chosen Momentum:", momentum
            # decay={{uniform(0, 0.5)}}
            #trial_model_optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
            trial_model_optimizer = SGD(lr=lr, momentum=momentum, nesterov=nesterov)
            trial_model_optimizer_dict['sgd'] = {'lr': lr,
                                              'momentum': momentum,
                                              'nesterov': nesterov}
        elif trial_model_optimizer_list == 'adagrad':
            print "\t\t#Chosen Epsilon:", epsilon

            print "\t\t#Chosen Learning Rate:", lr
            # beta_1 = {{uniform(0.5, 1)}}
            # beta_2 = {{uniform(0.6, 1)}}
            #trial_model_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon )
            trial_model_optimizer = Adagrad(lr=lr,epsilon=epsilon )
            trial_model_optimizer_dict['adagrad'] = {'lr': lr,
                                              'epsilon': epsilon}
        elif trial_model_optimizer_list == 'adamax':
            print "\t\t#Chosen Epsilon:", epsilon

            print "\t\t#Chosen Learning Rate:", lr
            # beta_1 = {{uniform(0.5, 1)}}
            # beta_2 = {{uniform(0.6, 1)}}
            #trial_model_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon )
            trial_model_optimizer = Adamax(lr=lr,epsilon=epsilon )
            trial_model_optimizer_dict['adamax'] = {'lr': lr,
                                              'epsilon': epsilon}
        elif trial_model_optimizer_list == 'adadelta':
            print "\t\t#Chosen Epsilon:", epsilon

            print "\t\t#Chosen Learning Rate:", lr
            # beta_1 = {{uniform(0.5, 1)}}
            # beta_2 = {{uniform(0.6, 1)}}
            #trial_model_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon )
            trial_model_optimizer = Adadelta(lr=lr,epsilon=epsilon )
            trial_model_optimizer_dict['adadelta'] = {'lr': lr,
                                              'epsilon': epsilon}
        # elif trial_model_optimizer_list == 'nadam':
        #     print "\t\t#Chosen Epsilon:", epsilon
        #     lr = 1e-4
        #     print "\t\t#Chosen Learning Rate:", lr
        #     # beta_1 = {{uniform(0.5, 1)}}
        #     # beta_2 = {{uniform(0.6, 1)}}
        #     #trial_model_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon )
        #     trial_model_optimizer = Nadam(lr=lr,epsilon=epsilon )
        #     trial_model_optimizer_dict['nadam'] = {'lr': lr,
        #                                       'epsilon': epsilon}




        saved_clean_model = model.to_json()

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='categorical_crossentropy',
                      optimizer=trial_model_optimizer,
                      metrics=['accuracy'])

        # fit the model
        batch_size = 128
        print "\t#Chosen batch size:", batch_size,"\n"
        model.fit(train, train_labels, nb_epoch=nb_epochs, batch_size=batch_size)
        predicted_labels = model.predict(validation)
        predicted_labels_linear = []
        for i in range(len(predicted_labels)):
            cls_prob = predicted_labels[i]
            predicted_labels_linear.append(np.argmax(cls_prob))

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
        output_model = {
            'model': saved_clean_model,
            'optimizer': trial_model_optimizer_dict,
            'batch_size': batch_size
        }
    #session.close()
    return {'loss': -MCC, 'status': STATUS_OK, 'model': output_model}


from hyperas import optim
from hyperopt import Trials, tpe
import argparse
import pickle
import numpy as np
import os
from images_utils import load_im2
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, Adamax
from mcc_multiclass import multimcc
from keras.models import model_from_json


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


parser = myArgumentParser(
    description='Run a training experiment using pretrained VGG16, specified on the Raspberry DataSet.',
    fromfile_prefix_chars='@')
parser.add_argument('--gpu', type=int, default=0, help='GPU Device (default: %(default)s)')
parser.add_argument('--nb_epochs', type=int, default=10, help='Number of Epochs during training (default: %(default)s)')
parser.add_argument('--n_trials', type=int, default=10,
                    help='Number of Trials during the HyperParameter Space Search (default: %(default)s)')
parser.add_argument('--vgg16_weights', type=str, default='vgg16_weights.h5', help='VGG16 PreTrained weights')
parser.add_argument('--output_dir', type=str, default="./experiment_output/", help='Output directory')
parser.add_argument('--input_dir', type=str, default="./", help='Input directory')
args = parser.parse_args()

GPU = args.gpu
NB_EPOCHS = args.nb_epochs
OUTDIR = args.output_dir + "/"
INDIR = args.input_dir + "/"
VGG_WEIGHTS = args.vgg16_weights
N_TRIALS = args.n_trials

try:
    os.makedirs(OUTDIR)
except OSError:
    if not os.path.isdir(OUTDIR):
        raise

inputs = {
    'GPU': GPU,
    'NB_EPOCHS': NB_EPOCHS,
    'VGG_WEIGHTS': VGG_WEIGHTS,
    'INDIR': INDIR
}

with open('inputs.pickle', 'wb') as handle:
    pickle.dump(inputs, handle)

best_run, best_model_dict = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=N_TRIALS,
                                      trials=Trials())

train_data_dir = INDIR + 'BerryPhotos/train'
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
            train_labels.append([1, 0, 0])
    elif path == train_path_g:
        for name in train_filenames_g:
            train_images.append(path + name)
            train_labels.append([0, 1, 0])
    elif path == train_path_l:
        for name in train_filenames_l:
            train_images.append(path + name)
            train_labels.append([0, 0, 1])
train = np.array(load_im2(train_images))

validation_data_dir = INDIR + 'BerryPhotos/validation'
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

validation = np.array(load_im2(validation_images))

OPTIMIZER_dict = best_model_dict['optimizer']

if OPTIMIZER_dict.keys()[0] == 'adam':
    OPTIMIZER = Adam(lr=OPTIMIZER_dict['adam']['lr'], epsilon=OPTIMIZER_dict['adam']['epsilon'])
elif OPTIMIZER_dict.keys()[0] == 'rmsprop':
    OPTIMIZER = RMSprop(lr=OPTIMIZER_dict['rmsprop']['lr'], epsilon=OPTIMIZER_dict['rmsprop']['epsilon'])
elif OPTIMIZER_dict.keys()[0] == 'sgd':
    OPTIMIZER = SGD(lr=OPTIMIZER_dict['sgd']['lr'], momentum=OPTIMIZER_dict['sgd']['momentum'],
                    nesterov=OPTIMIZER_dict['sgd']['nesterov'])
elif OPTIMIZER_dict.keys()[0] == 'adamax':
    OPTIMIZER = Adamax(lr=OPTIMIZER_dict['adamax']['lr'], epsilon=OPTIMIZER_dict['adamax']['epsilon'])
elif OPTIMIZER_dict.keys()[0] == 'adagrad':
    OPTIMIZER = Adagrad(lr=OPTIMIZER_dict['adagrad']['lr'], epsilon=OPTIMIZER_dict['adagrad']['epsilon'])
elif OPTIMIZER_dict.keys()[0] == 'adadelta':
    OPTIMIZER = Adadelta(lr=OPTIMIZER_dict['adadelta']['lr'], epsilon=OPTIMIZER_dict['adadelta']['epsilon'])
#elif OPTIMIZER_dict.keys()[0] == 'nadam':
#    OPTIMIZER = Nadam(lr=OPTIMIZER_dict['nadam']['lr'], epsilon=OPTIMIZER_dict['nadam']['epsilon'])


best_model = model_from_json(best_model_dict['model'])
best_model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
print("\n\n#########EXECUTING RETRAIN OF THE BEST MODEL TO SAVE WEIGHTS")
best_model.fit(train, train_labels,
                    batch_size=best_model_dict['batch_size'],
                    nb_epoch=NB_EPOCHS)

predicted_labels = best_model.predict(validation)

prediction_summary = open(OUTDIR + "vgg16_first_train_raspberry_prediction_summary.txt", "w")
prediction_summary.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

predicted_labels_linear = []

for i in range(len(predicted_labels)):
    cls_prob = predicted_labels[i]
    real_label = "NotFound"
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
print ("MCC of the best model:",MCC)
prediction_summary.write("\n\nMCC:" + str(MCC))
prediction_summary.close()
best_model.save_weights(OUTDIR + "vgg16_first_training_raspberry_weights_hyperas.h5", overwrite=True)


##RANDOM LABEL
train_labels_linear = []

for lbl in train_labels:
    if lbl[0] == 1:
        train_labels_linear.append(0)
    if lbl[1] == 1:
        train_labels_linear.append(1)
    if lbl[2] == 1:
        train_labels_linear.append(2)
random_train_labels_linear = np.copy(train_labels_linear)
np.random.shuffle(random_train_labels_linear)
random_train_labels = np_utils.to_categorical(random_train_labels_linear, max(random_train_labels_linear) + 1)

random_model = model_from_json(best_model_dict['model'])
random_model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
print("\n\n#########EXECUTING RANDOM LABEL OF THE BEST MODEL")
random_model.fit(train, random_train_labels,
                    batch_size=best_model_dict['batch_size'],
                    nb_epoch=NB_EPOCHS)

predicted_labels = random_model.predict(validation)
prediction_summary = open(OUTDIR + "vgg16_first_train_raspberry_prediction_summary_RANDOM_LABELS.txt", "w")
prediction_summary.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

predicted_labels_linear = []

for i in range(len(predicted_labels)):
    cls_prob = predicted_labels[i]
    real_label = "NotFound"
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

predicted_labels_linear = np.array(predicted_labels_linear)

MCC = multimcc(validation_labels_linear, predicted_labels_linear)
print ("MCC of the best model in RANDOM LABELS:",MCC)
prediction_summary.write("\n\nRandom Labels MCC:" + str(MCC))
prediction_summary.close()

