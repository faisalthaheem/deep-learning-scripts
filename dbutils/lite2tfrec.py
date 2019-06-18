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

import hashlib
import io
import logging
import random
import re
from objclass import getObjectClass
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

    y1,x1,y2,x2,width,height,imheight,imwidth,dbName,needscropping, labelid, labeltext, isbackground = None,None,None,None,None,None,None,None,None,None,1,"", None

    for db in sqlitedbs:
        try:
            cursor = dbcursors[db]
            if cursor is not None:
                #look up plate information for the requested name
                query = "SELECT y1,x1,y2,x2,width,height,imheight,imwidth,needscropping,labelid,labeltext,isbackground FROM plates WHERE filename = '{}' and isdeleted = 0".format(nameWithoutExtension)
                cursor.execute(query)
                row = cursor.fetchone()
                if row is not None:
                    y1,x1,y2,x2,width,height,imheight,imwidth,needscropping,labelid, labeltext, isbackground = row[0],row[1],row[2],row[3],row[4],row[5],int(row[6]),int(row[7]),int(row[8]),int(row[9]),row[10], row[11]
                    dbName = db
                    break

        except:
            logger.error(traceback.format_exc())

    return y1,x1,y2,x2,width,height,imheight,imwidth,dbName,needscropping,labelid, labeltext, isbackground

def translateCoord(imheight, imwidth, y1,x1,y2,x2):

    ratioY = (target_height / imheight)
    ratioX = (target_width / imwidth)

    y1 = int(y1 * ratioY)
    y2 = int(y2 * ratioY)
    x1 = int(x1 * ratioX)
    x2 = int(x2 * ratioX)

    return y1,x1,y2,x2

def createExample( filename, filepath,
                        y1,x1,y2,x2,w,h, classId, className):
  

    with tf.gfile.GFile(filepath, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width,height = image.size

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
    class_name = className
    classes_text.append(class_name.encode('utf8'))
    classes.append(int(classId))
    truncated.append(0)
    poses.append('unspecified'.encode('utf8'))
  


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

                y1,x1,y2,x2,width,height,imheight,imwidth,dbName,needscropping, labelid, labeltext, isbackground = lookupFile(fileName)
                if y1 is None:
                    #logger.warn("Could not get information for [{}]".format(filePath))
                    continue

                if isbackground:
                    num_processed_from_db = num_processed_from_db + 1
                                    
                    #create tf record
                    tf_example = createBackgroundExample(fileName, filePath)
                    writer.write(tf_example.SerializeToString())
                else:

                    # y1 -= pad_y
                    # y2 += pad_y
                    # x1 -= pad_x
                    # x2 += pad_x

                    # if y1 < 0:
                    #     y1 = 0
                    # if x1 < 0:
                    #     x1 = 0
                    # if x2 >= imwidth:
                    #     x2 = imwidth - 1
                    # if y2 >= imheight:
                    #     y2 = imheight - 1

                    cx1 = 0
                    cx2 = imwidth
                    cy1 = 0
                    cy2 = imheight

                    # print(db)
                    # print(imwidth,imheight)
                    # print(x1,y1, x2,y2)

                    if crop_enabled and needscropping:
                        if x2 + crop_width < imwidth:
                            cx2 = x2 + crop_width
                        if x1 - crop_width > 0:
                            cx1 = x1 - crop_width
                            width = x2 - x1
                            x1 = crop_width
                            x2 = x1 + width
                        
                        if y2 + crop_height < imheight:
                            cy2 = y2 + crop_height
                        if y1 - crop_height > 0:
                            cy1 = y1 - crop_height
                            height = y2 - y1
                            y1 = crop_height
                            y2 = y1 + height

                        
                        
                        imwidth = cx2 - cx1
                        imheight = cy2 - cy1
                        
                        # print("After")
                        # print(cx1,cy1,cx2,cy2)
                        # print(imwidth,imheight)
                        # print(x1,y1, x2,y2)
                    
                    if int(args['coords.translate']):
                        y1,x1,y2,x2 = translateCoord(imheight, imwidth, y1,x1,y2,x2)

                    #get class based on this db name
                    #classId, className = getObjectClass(dbName)

                    num_processed_from_db = num_processed_from_db + 1
                                    
                    #create tf record
                    tf_example = createExample(fileName, filePath, y1,x1,y2,x2,width,height, labelid, labeltext)
                    writer.write(tf_example.SerializeToString())

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