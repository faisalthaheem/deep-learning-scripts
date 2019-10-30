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

import argparse
import os
import sys
import sqlite3
import time
import pprint
import traceback
from tqdm import tqdm
import random
import json
import numpy as np
import cv2

import hashlib
import io
import logging
import random
import re
from collections import defaultdict

import PIL.Image
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("-ln", "--lite.names", required=True,
	help="comma seperated list of sqlite db files to look up image info from")
ap.add_argument("-dp", "--data.path", required=True,
	help="comma seperated list of folders containing images to process")
ap.add_argument("-op", "--output.path", required=True,
	help="file path containing the examples")    
ap.add_argument("-th", "--target.height", default=288,
    help="height of the resized image")
ap.add_argument("-tw", "--target.width", default=384,
    help="width of the resized image")    
ap.add_argument("-ch", "--crop.height", default=700,
    help="height of the resized image")
ap.add_argument("-cw", "--crop.width", default=700,
    help="width of the resized image")    
ap.add_argument("-crop", "--source.crop", default=False,
    help="If true, th and tw is used to crop around the roi before copying")    
ap.add_argument("-tc", "--coords.translate", default=0,
    help="0 disabled, 1 enabled")    


args = vars(ap.parse_args())

target_height = int(args["target.height"])
target_width = int(args["target.width"])

crop_height = int(args["crop.height"])
crop_width = int(args["crop.width"])

crop_enabled = args["source.crop"]

pad_x = 0
pad_y = 0

num_processed_from_db = 0

#create logger
logger = logging.getLogger('lite2tfrec.anpr')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('lite2tfrec.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)



def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


sqlitedbs = args["lite.names"].split(',')
datapaths = args["data.path"].split(',')

logger.info("Will be looking up following databases for info...")
logger.info(str(sqlitedbs))

logger.info("And, will be creating tf records from files under")
logger.info(str(datapaths))

dbconns = {}
dbcursors = {}

def loadsqlitedbs():
    
    logger.info("Instantiating sqlite dbs")

    for db in sqlitedbs:
        logger.info("Opening database " + db)

        if os.path.exists(db):
            conn = sqlite3.connect(db)
            dbconns[db] = conn

            cursor = conn.cursor()
            dbcursors[db] = cursor

            logger.info("database [{}] opened".format(db))
        else:
            logger.warn("database [{}] does not exist, skipping".format(db))
            dbcursors[db] = None
            dbconns[db] = None

    logger.info("dbs opened")

def closedbs():

    logger.info("Closing dbs")

    for db in sqlitedbs:
        if dbcursors[db] is not None:
            dbcursors[db].close()

        if dbconns[db] is not None:
            dbconns[db].close()

    logger.info("DBs closed")

def lookupFile(nameWithoutExtension):

    imheight,imwidth,dbName,isbackground,imgareas = None,None,None,None,None

    for db in sqlitedbs:
        try:
            cursor = dbcursors[db]
            if cursor is not None:
                #look up plate information for the requested name
                query = "SELECT imheight, imwidth, isbackground, imgareas FROM annotations WHERE filename = '{}' and isdeleted = 0".format(nameWithoutExtension)
                cursor.execute(query)
                row = cursor.fetchone()
                if row is not None:
                    imheight,imwidth,isbackground,imgareas = row[0],row[1],row[2],json.loads(row[3])
                    dbName = db
                    break

        except:
            logger.error(traceback.format_exc())

    return imheight,imwidth,dbName,isbackground,imgareas

def translateCoord(imheight, imwidth, y1,x1,y2,x2):

    ratioY = (target_height / imheight)
    ratioX = (target_width / imwidth)

    y1 = int(y1 * ratioY)
    y2 = int(y2 * ratioY)
    x1 = int(x1 * ratioX)
    x2 = int(x2 * ratioX)

    return y1,x1,y2,x2

def createExampleCropped( filename, filepath,
                        imheight,imwidth,dbName,isbackground,imgareas):
  
    #global targetWidth, targetHeight, crop_height, crop_width, crop_enabled, num_processed_from_db

    #case 1 when the image is smaller than target size
    # align the image to the top left of the dest size
    # coordinates do not require translation

    #case 2 when the image is larger than target size
    # align the mid point of the roi to the mid point of the target
    # extract enough pixels horizontally and vertically
    # align the roi to the center of the target size
    # translate coords

    #init new image
    source = cv2.imread(filepath)
    h,w,_ = source.shape

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []


    for imgarea in imgareas:

        target = np.zeros((target_height, target_width, 3), np.uint8)

        y1 = int(float(imgarea['y']))
        x1 = int(float(imgarea['x']))
        y2 = y1 + int(float(imgarea['height']))
        x2 = x1 + int(float(imgarea['width']))
        area_id = imgarea['id']
        class_name = imgarea['lbltxt']
        classId = imgarea["lblid"]
    
        # ensure the roi is not > target dimensions
        if x2-x1 > target_width or y2-y1 > target_height:
            # source = cv2.resize(source, (target_width, target_height))
            # w = target_width
            # h = target_height
            logger.warn("roi > target dimension, skipping {}".format(filename))
            # we will not crop and instead use the whole image use default method
            # see caller function
            return None
        
        roi_y1 = int(y1)
        roi_x1 = int(x1)
        roi_y2 = int(y2)
        roi_x2 = int(x2)

        #check for case 1
        if (w <= target_width and h <= target_height):
            #logger.info("{} <= {} and {} <= {}".format(w, target_width, h, target_height))
            target[0:,0:] = source
        else:
            #target_y_mid = target_height//2
            #target_x_mid = target_width//2

            #
            # assuming roi area will fit inside the target dimensions
            #
            req_pad_y = (target_height - (y2 - y1)) // 2
            req_pad_x = (target_width  - (x2 - x1)) // 2

            # # case when the src exceeds target in both dimensions
            # if w > target_width and h > target_height:
            #req_pad_x = 0
            #req_pad_y = 0

            if x1 - req_pad_x < 0:
                src_x1 = 0
                #no change for roi_x1
            else:
                src_x1 = x1 - req_pad_x
                roi_x1 = x1 - src_x1
                roi_x2 = x2 - src_x1

            if x2 + req_pad_x >= w:
                src_x2 = w
            else:
                src_x2 = x2 + req_pad_x

            
            if y1 - req_pad_y < 0:
                src_y1 = 0
                #no change for roi_y1
            else:
                src_y1 = y1 - req_pad_y
                roi_y1 = y1 - src_y1
                roi_y2 = y2 - src_y1

            if y2 + req_pad_y >= h:
                src_y2 = h
            else:
                src_y2 = y2 + req_pad_y

            src_x1 = int(src_x1)
            src_x2 = int(src_x2)
            src_y1 = int(src_y1)
            src_y2 = int(src_y2)

            roi_x1 = int(roi_x1)
            roi_x2 = int(roi_x2)
            roi_y1 = int(roi_y1)
            roi_y2 = int(roi_y2)

            content_w = int(src_x2 - src_x1)
            content_h = int(src_y2 - src_y1)
            #logger.info("{} {} {} {} {} {} {}".format(filename, content_h, content_w, src_y1, src_y2, src_x1, src_x2))
            target[0:content_h, 0:content_w] = source[src_y1:src_y2, src_x1:src_x2]

            tosave = target.copy()
            coords = "y:x {}, {} : {} ,{}".format(roi_y1, roi_x1, roi_y2, roi_x2)
            tosave = cv2.rectangle(tosave, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,255,0), 1)
            cv2.putText(tosave, coords, (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv2.imwrite( "{}-{}.jpg".format(filename,area_id), tosave)
        
        # with tf.gfile.GFile(filepath, 'rb') as fid:
        #     encoded_jpg = fid.read()
        #     encoded_jpg_io = io.BytesIO(encoded_jpg)
        #     image = PIL.Image.open(encoded_jpg_io)
        #     if image.format != 'JPEG':
        #         raise ValueError('Image format not JPEG')
        #     key = hashlib.sha256(encoded_jpg).hexdigest()

        #     width,height = image.size
        
        encoded_jpg = cv2.imencode('.jpg', target)[1].tostring()
        #encoded_jpg_io = io.BytesIO(encoded_jpg)
        key = hashlib.sha256(encoded_jpg).hexdigest()

        
    
        exmin = float(roi_x1) / target_width
        eymin = float(roi_y1) / target_height
        exmax = float(roi_x2) / target_width
        eymax = float(roi_y2) / target_height

        xmin.append(exmin)
        ymin.append(eymin)
        xmax.append(exmax)
        ymax.append(eymax)

        classes_text.append(class_name.encode('utf8'))
        classes.append(int(classId))

        truncated.append(0)
        poses.append('unspecified'.encode('utf8'))
  
    # logger.info(json.dumps({
    #     'image/height': target_height,
    #     'image/width': target_width,
    #     'image/filename': filename,
    #     'image/source_id': filename,
    #     'image/object/bbox/xmin': xmin,
    #     'image/object/bbox/xmax': xmax,
    #     'image/object/bbox/ymin': ymin,
    #     'image/object/bbox/ymax': ymax,
    #     'image/object/class/text': class_name,
    # }))


    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(target_height),
        'image/width': int64_feature(target_width),
        'image/filename': bytes_feature(
            filename.encode('utf8')),
        'image/source_id': bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/difficult': int64_list_feature(difficult_obj),
        'image/object/truncated': int64_list_feature(truncated),
        'image/object/view': bytes_list_feature(poses),
    }))
    return example

def createExample( filename, filepath,
                        imheight,imwidth,dbName,isbackground,imgareas):
  

    with tf.gfile.GFile(filepath, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width,height = image.size

    for imgarea in imgareas:
        
        y1 = int(float(imgarea['y']))
        x1 = int(float(imgarea['x']))
        y2 = y1 + int(float(imgarea['height']))
        x2 = x1 + int(float(imgarea['width']))
        area_id = imgarea['id']
        class_name = imgarea['lbltxt']
        classId = imgarea["lblid"]

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
    
        exmin = float(x1) / width
        eymin = float(y1) / height
        exmax = float(x2) / width
        eymax = float(y2) / height

        xmin.append(exmin)
        ymin.append(eymin)
        xmax.append(exmax)
        ymax.append(eymax)
        
        classes_text.append(class_name.encode('utf8'))
        classes.append(int(classId))
        truncated.append(0)
        poses.append('unspecified'.encode('utf8'))
  
    # logger.info(json.dumps({
    #     'image/height': height,
    #     'image/width': width,
    #     'image/filename': filename,
    #     'image/source_id': filename,
    #     'image/object/bbox/xmin': xmin,
    #     'image/object/bbox/xmax': xmax,
    #     'image/object/bbox/ymin': ymin,
    #     'image/object/bbox/ymax': ymax,
    #     'image/object/class/text': class_name,
    # }))


    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(
            filename.encode('utf8')),
        'image/source_id': bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/difficult': int64_list_feature(difficult_obj),
        'image/object/truncated': int64_list_feature(truncated),
        'image/object/view': bytes_list_feature(poses),
    }))
    return example

def createBackgroundExample( filename, filepath ):
  

    with tf.gfile.GFile(filepath, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG ' + filepath)
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width,height = image.size

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(
            filename.encode('utf8')),
        'image/source_id': bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
    }))
    return example

def processDataDir(datapath, writer):

    logger.info("Processing data path [{}]".format(datapath))
    global targetWidth, targetHeight, crop_height, crop_width, crop_enabled, num_processed_from_db
    labelid = 1
    labeltext = "unknown" # very  bad.. should never be written!!

    for root, dirs, files in os.walk(datapath):
        
        totalFiles = len(files)
        random.shuffle(files)
        
        for i in tqdm(range(0,totalFiles)):
                fileName = files[i]        
                filePath = os.path.join(root,fileName)
                #logger.info("Processing.. {} [{} of {}]".format(filePath, i, totalFiles))

                imheight,imwidth,dbName,isbackground,imgareas = lookupFile(fileName)
                if imheight is None:
                    logger.warn("Could not get information for [{}]".format(filePath))
                    continue

                if isbackground:
                    #create tf record
                    tf_example = createBackgroundExample(fileName, filePath)
                    writer.write(tf_example.SerializeToString())
                else:
                    #create tf record
                    tf_example = None

                    if crop_enabled:
                        try:
                            tf_example = createExampleCropped(fileName, filePath, imheight,imwidth,dbName,isbackground,imgareas)
                        except:
                            logger.error("Error cropping [{}]".format(fileName))
                    else:
                    #in case we are not in crop mode or cropping failed because roi was larger than required cropped area
                        tf_example = createExample(fileName, filePath, imheight,imwidth,dbName,isbackground,imgareas)

                    if tf_example is not None:
                        writer.write(tf_example.SerializeToString())

                num_processed_from_db = num_processed_from_db + 1

time_start = time.time()

loadsqlitedbs()
writer = tf.python_io.TFRecordWriter(args["output.path"])

for path in datapaths:
    processDataDir(path, writer)

#clean up part
writer.close()
closedbs()

time_end = time.time()

logger.info("Took [{}] s to process request".format(time_end-time_start))
logger.info("[{}] files written to tfrec.".format(num_processed_from_db))