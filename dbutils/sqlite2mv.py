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
import logging
import shutil
from objclass import getObjectClass
from tqdm import tqdm


ap = argparse.ArgumentParser()
ap.add_argument("-ln", "--lite.names", required=True,
    help="comma seperated list of sqlite db files to look up image info from")
ap.add_argument("-sp", "--src.path", required=True,
    help="source folder containing files to move")
ap.add_argument("-dp", "--dst.path", required=True,
    help="dest folder to move files to")

args = vars(ap.parse_args())

#create logger
logger = logging.getLogger('sqlite2mv')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('sqlite2mv.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


sqlitedbs = args["lite.names"].split(',')
datapaths = args["src.path"].split(',')

logger.info("Will be looking up following databases for info...")
logger.info(str(sqlitedbs))

logger.info("And, will be moving files from..")
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


def lookupFile(nameWithoutExtension, filePath):

    y1,x1,y2,x2,width,height,classId,className = None,None,None,None,None,None,None,None
    global targetWidth, targetHeight, crop_height, crop_width, crop_enabled

    for db in sqlitedbs:
        try:
            cursor = dbcursors[db]
            if cursor is not None:
                #look up plate information for the requested name
                query = "SELECT y1,x1,y2,x2,width,height,imheight,imwidth FROM plates WHERE filename = '{}' and isdeleted = 0".format(nameWithoutExtension)
                cursor.execute(query)
                row = cursor.fetchone()
                if row is not None:
                    y1,x1,y2,x2,width,height,imheight,imwidth = int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7])
                    break

        except:
            logger.error(traceback.format_exc())

    return y1,x1,y2,x2,width,height,classId,className

def processDataDir(datapath):

    logger.error("Processing data path [{}]".format(datapath))

    for root, dirs, files in os.walk(datapath):

        totalFiles = len(files)
        logger.error("Processing [{}] file(s) in root [{}]".format(totalFiles, root))

        for i in tqdm(range(0,totalFiles)):
            fileName = files[i]        
            filePath = os.path.join(root,fileName)
            destPath = os.path.join(args['dst.path'], fileName)
            #logger.info("Processing.. {} [{} of {}]".format(filePath, i, totalFiles))

            #y1,x1,y2,x2,width,height = lookupFile(os.path.splitext(fileName)[0])
            y1,x1,y2,x2,width,height,classId,className = lookupFile(fileName, filePath)
            if y1 is None:
                logger.warn("Could not get information for [{}] @ [{}]".format(fileName,filePath))
                continue
            try:
                # move file from source to dest
                logger.info("Moving [{}] to [{}]".format(filePath, destPath))
                shutil.move(filePath, destPath)
            except:
                logger.error(traceback.format_exc())


time_start = time.time()

loadsqlitedbs()

for path in datapaths:
    processDataDir(path)

#clean up part
closedbs()
time_end = time.time()
logger.info("Took [{}] s to process request".format(time_end-time_start))