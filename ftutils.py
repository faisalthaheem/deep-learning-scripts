"""
   Copyright 2019 Faisal Thaheem

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import time
import json
import os
import cv2
import io
import tensorflow as tf

from pprint import pprint
from random import shuffle
from PIL import Image

from keras.backend.tensorflow_backend import set_session
from keras.utils import np_utils
from keras.models import Model, load_model, model_from_json
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from skimage.color import rgb2gray

def tf_new_session(device_id = "0", memory_fraction = 1.0):
    
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = device_id
    sess = tf.Session(config=config)
    # see https://github.com/keras-team/keras/issues/4780
    sess.run(tf.global_variables_initializer())
    
    return sess



def set_tf_session_for_keras(device_id = "0", memory_fraction = 1.0):
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = device_id
    sess = tf.Session(config=config)
    # see https://github.com/keras-team/keras/issues/4780
    sess.run(tf.global_variables_initializer())
    set_session(sess)

def load_image_into_numpy_array(path_to_image, imgdim=(96,96,1), grayScale = True):

    if None != imgdim:
        img = image.load_img(path_to_image, grayscale = grayScale, target_size=(imgdim[0], imgdim[1]))
    else:
        img = image.load_img(path_to_image, grayscale = grayScale, target_size=None)

    x = image.img_to_array(img).astype(np.uint8)

    return x

def bytes_to_image_for_classification(imgbytes, mode="RGB"):
    img = Image.open(io.BytesIO(imgbytes))
    img = img.convert(mode)
    ret = np.array(img)
    return ret,img

def load_image_for_classification(path_to_image, imgdim=(96,96,1),expandDims=True,grayScale = True):
    
    if imgdim != None:
        img = image.load_img(path_to_image, grayscale = grayScale, target_size=(imgdim[0], imgdim[1]))
    else:
        img = image.load_img(path_to_image, grayscale = grayScale)

    x = image.img_to_array(img).astype(np.uint8)

    if expandDims is True:
        x = np.expand_dims(x, axis=0)

    x = x / 255

    return x

def load_images_for_classification(path_to_images, imgdim=(96,96,1)):

    h,w,c = imgdim
    loaded_images = np.empty((len(path_to_images), 1, h,w,c), np.float)

    for i in range(0,len(path_to_images)):
        path = path_to_images[i]
        loaded_image = load_image_for_classification(path, imgdim, True)
        loaded_images[i] = loaded_image
    return loaded_images

def convertToGrayscaleForClassification(img):

    imgDim = img.shape

    img = rgb2gray(img)
    img = np.reshape(img, (imgDim[0],imgDim[1],1))

    return img

def overlayImageOnBlackCanvas(img, canvas_shape = (400,400,3)):

    h,w,c = img.shape

    computed_canvas_shape = canvas_shape
    resizeAtEnd = False

    if h>canvas_shape[0] or w>canvas_shape[1]:
        max_dim = max(h,w)
        computed_canvas_shape = (max_dim,max_dim,c)
        
        resizeAtEnd = True

    canvas = np.zeros(computed_canvas_shape).astype(np.uint8)

    insert_y = (computed_canvas_shape[0] - h) //2
    insert_x = (computed_canvas_shape[1] - w) //2
    
    canvas[insert_y: insert_y+h , insert_x:insert_x+w] = img

    if resizeAtEnd is True:
        canvas = resize(canvas, canvas_shape, preserve_range=True).astype(np.uint8)

    return canvas

    #print("Inserted image of dimension {} at y {} and x {} on canvas of dimension {}".format(img.shape, (insert_y,insert_y+h), (insert_x,insert_x+w), black_bg.shape))


def make_dataset(loc, split = 0.2, imgdim=(96,96,1), grayScale = True, max_test_files = 4096):
    #the path contains sub folders, name of folder is the label whereas
    t_start = time.time()

    #dictionary of foldername -> list
    train_files = {}
    
    for root, directory, files in os.walk(loc):
        if root != loc:
            label = os.path.basename(root)
            train_files[label] = [ os.path.join(root,x) for x in os.listdir(root)]
            shuffle(train_files[label])
    
    tmp_keys = list(train_files.keys())
    
    #print(len(train_files[tmp_keys[0]]), split_index)
    
    #split the data into train and dev
    num_train_files = 0
    num_dev_files = 0

    max_test_files_per_class = max_test_files // len(tmp_keys)
    print("Max X_test size is [{}] - per class [{}]".format(max_test_files, max_test_files_per_class))

    train_files_list = []
    dev_files_list = []
    dev_files = {}
    for k in tmp_keys:

        print("Processing class [{}]".format(k), end='')
        
        split_index = int(len(train_files[k]) * float(split))
        
        #take only max_test_files as test samples.. big enough
        if split_index >  max_test_files_per_class:
            split_index = max_test_files_per_class
        
        num_train_files += (len(train_files[k]) - split_index)
        num_dev_files += split_index

        dev_files[k] = train_files[k][:split_index]
        train_files[k] = train_files[k][split_index:]

        #add train files to the list to be returned
        for f in train_files[k]:
            train_files_list.append((k,f))

        for f in dev_files[k]:
            dev_files_list.append((k,f))

        print("| train_files [{}] & dev_files [{}]".format(len(train_files[k]), len(dev_files[k])))
    
    unique_classes = np.unique(tmp_keys)
    unique_classes.sort()

    t_end = time.time()
    print("Took [{}] s. to make dataset".format((t_end-t_start)))
    
    return num_train_files, num_dev_files, tmp_keys, train_files_list, dev_files_list, list(unique_classes)

def load_minibatch(classes, train_files_list, batch_size, batch_number,imgdim=(96,96,1), grayScale = True):
    batch_start_index = batch_size * batch_number

    X_index = 0
    X = np.empty((batch_size,imgdim[0],imgdim[1],imgdim[2]),np.uint8)
    Y = []

    for i in range(batch_start_index, batch_start_index+batch_size):

        train_item = train_files_list[i]
        X[X_index] = load_image_into_numpy_array(train_item[1], imgdim, grayScale = grayScale)
        Y.append(train_item[0])

        X_index += 1
    
    
    #ensure we have len(classes) = len(np.unique(Y))
    Y_unique = np.unique(Y)
    missing_classes = list(set(classes) - set(Y_unique))

    
    for itm_class, itm_path in train_files_list:
        if itm_class in missing_classes:
            X = np.append(X, np.expand_dims(load_image_into_numpy_array(itm_path, imgdim, grayScale = grayScale),axis=0), axis=0)
            Y.append(itm_class)

            missing_classes.remove(itm_class)
    
    return X, Y

def convert_to_one_hot(classes, labels):
    encoder = LabelEncoder()
    encoder.fit(classes)
    encoded_labels = encoder.transform(labels)
    one_hot_labels = np_utils.to_categorical(encoded_labels)
    return encoded_labels, one_hot_labels

def write_file_contents(file_name, content):
    with open(file_name, 'w') as file_to_write:
        file_to_write.write(content)

def read_file_contents(file_name):
    with open(file_name,'r') as file_to_read:
        return file_to_read.read()

def saveModel(mdl, filename, classes):

    classes_json = json.dumps(classes)
    write_file_contents(filename + ".classes.json", classes_json)

    model_json = mdl.to_json()
    write_file_contents(filename + ".model.json", model_json)

    mdl.save_weights(filename + ".weights.h5")


def loadModel(fname, loss_metric = 'categorical_crossentropy'):
    print("Loading model from json: ", fname + ".model.json")

    # see https://github.com/keras-team/keras/issues/8538
    # for details on loading
        
    graph = tf.Graph()
    with graph.as_default():
        
        sess = tf_new_session('0',0.1)
        with sess.as_default():

            model_json = read_file_contents(fname+".model.json")
            model = model_from_json(model_json)
            
            #load weights
            print("Loading weights from: ", fname + ".weights.h5")
            model.load_weights(fname + ".weights.h5")
            model.compile(optimizer='adam', loss=loss_metric, metrics=['accuracy'])

            #load classes
            classes_json = read_file_contents(fname + ".classes.json")
            classes = json.loads(classes_json)

            return model, classes, graph, sess
    
    #should never come to this
    return None, None, None, None