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
import sys
import os
import tensorflow as tf
import shutil as sh
from PIL import Image
from resizeimage import resizeimage
import logging
from joblib import Parallel, delayed

import sqlite3
import time
import pprint
import traceback
from tqdm import tqdm

import hashlib
import io
import random
import re


ap = argparse.ArgumentParser()

ap.add_argument("-sd", "--source.dir", required=True,
    help="Directory containing files to be resized")
ap.add_argument("-dd", "--destination.dir", required=True,
    help="destination where files have to be copied resized")

ap.add_argument("-ih", "--image.height", default=256,
    help="height of the image")
ap.add_argument("-iw", "--image.width", default=256,
    help="width of the image")
ap.add_argument("-ln", "--lite.names", required=True,
    help="comma seperated list of sqlite db files to look up image info from")

ap.add_argument("-ch", "--crop.height", default=700,
    help="height of the resized image")
ap.add_argument("-cw", "--crop.width", default=700,
    help="width of the resized image")    
ap.add_argument("-crop", "--source.crop", default=False,
    help="If true, th and tw is used to crop around the roi before copying")    

ap.add_argument("-limit", "--copy.limit", default=15000,
    help="limits the number of copies to provided value")    
ap.add_argument("-randomize", "--copy.random", default=False,
    help="files are shuffled before being processed if True")
ap.add_argument("-useref", "--copy.ref", default='',
    help="only files in ref folder are copied, negates randomize and limit")
ap.add_argument("-nodb", "--db.ignore", default=True,
    help="files are shuffled before being processed if True")
    
args = vars(ap.parse_args())

#create logger
logger = logging.getLogger('copyresize')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('copyresize.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

crop_height = int(args["crop.height"])
crop_width = int(args["crop.width"])

targetWidth = int(args["image.width"])
targetHeight = int(args["image.height"])

crop_enabled = args["source.crop"]

sqlitedbs = args["lite.names"].split(',')
datapaths = args["source.dir"].split(',')

logging.info("Will be looking up following databases for info...")
logging.info(str(sqlitedbs))

logging.info("And, will be creating tf records from files under")
logging.info(str(datapaths))

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

    y1,x1,y2,x2,width,height,imheight,imwidth,dbName = None,None,None,None,None,None,None,None,None

    for db in sqlitedbs:
        try:
            cursor = dbcursors[db]
            if cursor is not None:
                #look up plate information for the requested name
                query = "SELECT y1,x1,y2,x2,width,height,imheight,imwidth FROM plates WHERE filename = '{}'".format(nameWithoutExtension)
                cursor.execute(query)
                row = cursor.fetchone()
                if row is not None:
                    y1,x1,y2,x2,width,height,imheight,imwidth = row[0],row[1],row[2],row[3],row[4],row[5],int(row[6]),int(row[7])
                    dbName = db
                    break

        except:
            logger.error(traceback.format_exc())

    return y1,x1,y2,x2,width,height,imheight,imwidth, dbName

def resizeImage(fileName, filePath, dest_file):
    try:
        global targetWidth, targetHeight,crop_height, crop_width, crop_enabled, args

        y1,x1,y2,x2,width,height,imheight,imwidth, dbName = lookupFile(fileName)
        if y1 is None and not bool(args['db.ignore']):
            return False

        cx1 = 0
        cx2 = imwidth
        cy1 = 0
        cy2 = imheight

        if crop_enabled:
            if x2 + crop_width < imwidth:
                cx2 = x2 + crop_width
            if x1 - crop_width > 0:
                cx1 = x1 - crop_width
                x1 = x1 - crop_width
                x2 = x2 - crop_width
            
            if y2 + crop_height < imheight:
                cy2 = y2 + crop_height
            if y1 - crop_height > 0:
                cy1 = y1 - crop_height
                y1 = y1 - crop_height
                y2 = y2 - crop_height
            
            
        img = Image.open(filePath)

        if crop_enabled:
            img = img.crop((cx1,cy1,cx2,cy2))
        
        img = img.resize((targetWidth,targetHeight))
        if img.mode != 24:
            img = img.convert("RGB")
        
        img.save(dest_file)

        
    except Exception as e:
        logger.error(e)

    return True

def processDataDir(datapath):

    logger.info("Processing data path [{}]".format(datapath))
    global filesCopied

    items_to_process = []
    ref_items = []
    using_ref_items = False

    if len(args["copy.ref"]) > 0:
        ref_items = os.listdir(args["copy.ref"])
        using_ref_items = True
        logger.info("[{}] ref files indexed.".format(len(ref_items)))

    for root, dirs, files in os.walk(args["source.dir"]):
        
        totalFiles = len(files)
        logger.info("[{}] has [{}] files to process.".format(args["source.dir"],totalFiles))

        fyles = files
        if bool(args['copy.random']) == True:
            #logger.info("File list randomized...")
            random.shuffle(fyles)

        for i in tqdm(range(0,totalFiles)):
            fileName = fyles[i]

            if using_ref_items and (fileName not in ref_items):
                logger.info("[{}] not in ref".format(fileName))
                continue

            filePath = os.path.join(root,fileName)
            #logger.info("Processing.. {} [{} of {}]".format(filePath, i, totalFiles))
            dest_file = os.path.join(args["destination.dir"], fileName)
            if resizeImage(fileName, filePath, dest_file):
                filesCopied += 1

                if filesCopied >= int(args["copy.limit"]):
                    logger.info("copy.limit reached. exiting.")
                    break
            #items_to_process.append((fileName, filePath, dest_file))

    #logger.info("Commencing parallel processing jobs")			
    #results = Parallel(n_jobs=4, backend="threading")(map(delayed(resizeImage), items_to_process))
    #items_to_process = []
    logger.info("Done..")


if not os.path.exists(args["destination.dir"]):
    logger.warn("%s is not a directory or does not exist, will try to create." % args["destination.dir"])
    os.mkdir(args["destination.dir"])
    if not os.path.isdir(args["destination.dir"]):
        logger.error("Unable to create destination path. Will exit now")
        sys.exit()


filesCopied = 0
time_start = time.time()

loadsqlitedbs()

for path in datapaths:
    if filesCopied < int(args["copy.limit"]):
        processDataDir(path)

closedbs()

time_end = time.time()

logger.info("Took [{}] s to process request".format(time_end-time_start))