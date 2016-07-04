"""List of settings for the t-SNE
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

WEIGHTS_PATH = 'vgg16_weights.h5'

IMAGE_PATH = '/data/webvalley/fruit_images/'

# dimensions of our images (used in model building)
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Class Names
RASPBERRY = 'raspberry'
STRAWBERRY = 'strawberry'
BLACKBERRY = 'blackberry'
RED_CURRANT = 'redcurrant'
WHITE_CURRANT ='whitecurrant'
BLUEBERRY = 'blueberry'
CHERRY = 'cherry'
PLUM = 'plum'
APRICOT = 'apricot'
GOOSEBERRY = 'Gooseberry'

# Plot Configuration for each class
MARKERS = {RASPBERRY: 'o',
           STRAWBERRY: '+',
           BLACKBERRY: '*',
           RED_CURRANT: 'D',
           WHITE_CURRANT: 'h',
           BLUEBERRY: 's',
           CHERRY: 'd',
           PLUM: '8',
           APRICOT: 'p',
           GOOSEBERRY: '<'}

COLOURS = {RASPBERRY: '#ff6666',
           STRAWBERRY: '#794044',
           BLACKBERRY: '#000000',
           RED_CURRANT: '#f03939',
           WHITE_CURRANT: '#f0f688',
           BLUEBERRY: '#3a539b',
           CHERRY: '#f688b4',
           PLUM: '#9615f0',
           APRICOT: '#f0b015',
           GOOSEBERRY: '#15f024'}

CLASSES_OF_INTEREST = [APRICOT, PLUM, CHERRY, BLUEBERRY]

OUTPUT_IMAGES_FILE_PREFIX = "images_matrix"
DEFAULT_IMAGES_FILENAME = OUTPUT_IMAGES_FILE_PREFIX + ".txt"

OUTPUT_TSNE_FILE_PREFIX = "tsne_matrix"