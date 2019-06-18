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

from PIL import Image
import pprint
import tkinter as tk
from tkinter import filedialog

import sys
import numpy as np
import pprint
import time
import os
import argparse as argparse
import json
import ftutils as ft
import cv2
import math
from skimage.transform import resize
from skimage.color import rgb2gray

from keras import layers
from keras.models import Model, Sequential
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.utils import np_utils
from keras.callbacks import History
from keras.initializers import glorot_uniform

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import obj_detector

from label_map_util import get_label_map_dict

ap = argparse.ArgumentParser()
ap.add_argument("-mp", "--model.path", required=True,
        help="folder containing the model")
ap.add_argument("-lp", "--label.path", required=True,
        help="path to labels")
ap.add_argument("-iw", "--img.width", default=384,
        help="width of the image")
ap.add_argument("-ih", "--img.height", default=288,
    help="height of the image.")
ap.add_argument("-ic", "--img.chan", default=3,
        help="channels in the image")

args = vars(ap.parse_args())

imgdim = (int(args["img.height"]), int(args["img.width"]), int(args["img.chan"]))

detector = obj_detector.ObjDetector()
detector.init(
    args["model.path"],
    args["label.path"])

label_mapping = get_label_map_dict(args['label.path'], use_display_name=True)
label_mapping = {v:k for k,v in label_mapping.items()}
#pprint.pprint(label_mapping)

def detectObject(filePath):

        start=time.time()
        
        img_np = ft.load_image_into_numpy_array(filePath, imgdim=None, grayScale=False).astype(np.uint8)
        image_np_expanded = np.expand_dims(img_np, axis=0)

        (boxes, scores, classes, num) = detector.detect(image_np_expanded)
        pprint.pprint(scores[0][0])
        confidence = round((scores[0][0])*100.0)
        end=time.time()
        
        # pprint.pprint(result)

        if len(boxes) > 0: # and classes[0][0] == 1:
                print("Object found in [{}]s  conf[{}] class[{}]........".format((end-start), confidence, classes[0][0]))

                height, width, _ = img_np.shape

                first_result = boxes[0][0]

                y1 = int(height * first_result[0])
                x1 = int(width * first_result[1])
                y2 = int(height * first_result[2])
                x2 = int(width * first_result[3])

                first_result_image = img_np[y1:y2, x1:x2].copy().astype(np.uint8)

                plt.ion()
                fig,ax = plt.subplots(1)
                ax.imshow(img_np)

                # Create a Rectangle patch over the detected area
                rect = patches.Rectangle(
                        (x1,y1),
                        x2-x1,
                        y2-y1,
                        linewidth=1,edgecolor='r',facecolor='none')
        
                # Add the patch to the Axes
                ax.add_patch(rect)
                ax.text(10,10, 
                        '{} @{}'.format(label_mapping[int(classes[0][0])],confidence),
                        bbox=dict(facecolor='red'),
                        fontsize=12 
                )
                plt.show()

                first_result_image = ft.overlayImageOnBlackCanvas(first_result_image)
                

                return first_result_image
        else:
                print("No object of interest found")

        return None


def detectionLoop():

    tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

    while True:
        filePath = filedialog.askopenfilename(filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        if len(filePath) is 0:
            break

        print("Processing file: ", filePath)

        t_start = time.time()

        rootPath = os.path.dirname(filePath)
        filename = os.path.basename(filePath)

        _ = detectObject(filePath)
        t_1 = time.time()
        
        print("File processed in [{}] s.\n".format((t_1-t_start)))

detectionLoop()
