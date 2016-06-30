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


# path to the model weights files.
weights_path = 'vgg16_weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224

#train_data_dir = 'data/train'
#validation_data_dir = 'data/validation'
#nb_train_samples = 2000
#nb_validation_samples = 800
#nb_epoch = 50

l_images = []
for root, dirs, files in os.walk("/home/a-marcolini/deepLearning/BerryPhotos/V/Good/Individual/Cropped"):
    for name in files:
        l_images.append(os.path.join(root, name))

#path to the images of rasperries directory
#image_path = '/home/a-marcolini/Downloads/BerryImageNet'

classes_path = 'classes_final.csv'


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
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))


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

# add the model on top of the convolutional base
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)



# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
#model.compile(loss='binary_crossentropy',
#              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#              metrics=['accuracy'])
model.compile(optimizer=sgd, loss='categorical_crossentropy')

#filenames = os.listdir(path=image_path)

f = open(classes_path, "r")
fout = open("out.txt", "w")
classes = f.readlines()
index_classes = {}
for i in range(1,len(classes)):
    l = classes[i].split(";")
    index = l[0]
    clas = l[1]
    clas = clas[:-1]
    index_classes[index] = clas
#   print(index_classes)

sout=""

j=0
n_str = 0
for i,name in enumerate(l_images):
    sout += str(i+1) + ") name = " + name + "\n"
    j+=1
    im = cv2.resize(cv2.imread(name), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    #print(im)
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained mode
    out = model.predict(im)
    best5 = np.argsort(out[0])[::-1][:5] #to extract the top5 class indexes
    best5_prob = []
    for k in best5:
        best5_prob.append(out[0][k])
    best5_prob = np.array(best5_prob)

    for t,num in enumerate(best5):
        classification = index_classes[str(num)]
        if classification == "strawberry":
            classification = "____STRAWBERRY!!____"
            if t==0:
                n_str += 1
        probability = best5_prob[t]
        probability = round(probability, 5)
        sout+="\t" + str(t+1) + ". class = " + classification
        sout+= "\n\t   probability = " + str(probability*100) + "%\n"

    #print(best5)
    #print(best5_prob)
    #print("\n")
    #if j==8:
    #    break
    print (j)

sout += "Detection percentage: " + str((n_str*1.0/len(l_images))*100) + "%"


fout.write(sout)




'''
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

'''

'''
# fine-tune the model
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
'''
