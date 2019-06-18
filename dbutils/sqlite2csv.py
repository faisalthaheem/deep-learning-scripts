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
import csv
import cv2
from objclass import getObjectClass
from tqdm import tqdm


ap = argparse.ArgumentParser()
ap.add_argument("-ln", "--lite.names", required=True,
    help="comma seperated list of sqlite db files to look up image info from")
ap.add_argument("-dp", "--data.path", required=True,
    help="comma seperated list of folders containing images to process")
ap.add_argument("-fn", "--file.name", required=True,
    help="name of csv file")
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
ap.add_argument("-annotate", "--source.annotate", default=False,
    help="If true, a copy of each processed image; annotated, is saved into the annotated folder")
ap.add_argument("-ad", "--annotate.dir", default="./annotated",
    help="width of the resized image")    

args = vars(ap.parse_args())

#create logger
logger = logging.getLogger('sqlite2csv.anpr')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('sqlite2csv.log')
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
datapaths = args["data.path"].split(',')

logger.info("Will be looking up following databases for info...")
logger.info(str(sqlitedbs))

logger.info("And, will be creating tf records from files under")
logger.info(str(datapaths))

dbconns = {}
dbcursors = {}

crop_height = int(args["crop.height"])
crop_width = int(args["crop.width"])

target_height = int(args["target.height"])
target_width = int(args["target.width"])

crop_enabled = args["source.crop"]

pad_x = 0
pad_y = 0

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

def translateCoord(imheight, imwidth, y1,x1,y2,x2):

    ratioY = (target_height / imheight)
    ratioX = (target_width / imwidth)

    y1 = int(y1 * ratioY)
    y2 = int(y2 * ratioY)
    x1 = int(x1 * ratioX)
    x2 = int(x2 * ratioX)

    return y1,x1,y2,x2

def lookupFile(nameWithoutExtension, filePath):

    y1,x1,y2,x2,width,height,classId,className = None,None,None,None,None,None,None,None
    global targetWidth, targetHeight, crop_height, crop_width, crop_enabled

    for db in sqlitedbs:
        try:
            cursor = dbcursors[db]
            if cursor is not None:
                #look up plate information for the requested name
                query = "SELECT y1,x1,y2,x2,width,height,imheight,imwidth FROM plates WHERE filename = '{}' and isdeleted = 0 and isbackground=0".format(nameWithoutExtension)
                cursor.execute(query)
                row = cursor.fetchone()
                if row is not None:
                    y1,x1,y2,x2,width,height,imheight,imwidth = int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7])

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

                    # print("db: " + db)
                    # print(imwidth,imheight)
                    # print(x1,y1, x2,y2)

                    if crop_enabled and (nameWithoutExtension.startswith("2017-") or nameWithoutExtension.startswith("2018-")):
                        if x2 + crop_width < imwidth:
                            cx2 = x2 + crop_width
                        if x1 - crop_width >= 0:
                            cx1 = x1 - crop_width
                            width = x2 - x1
                            x1 = crop_width
                            x2 = x1 + width
                        
                        if y2 + crop_height < imheight:
                            cy2 = y2 + crop_height
                        if y1 - crop_height >= 0:
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
                        
                    #translate the original coordinates into destination coordinates
                    y1,x1,y2,x2 = translateCoord(imheight, imwidth, y1,x1,y2,x2)

                    if args["source.annotate"]:
                        img = cv2.imread(filePath)
                        cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 5)
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0), 2)
                        savePath = os.path.join(args["annotate.dir"],nameWithoutExtension)
                        cv2.imwrite(savePath, img)

                    #get class based on this db name
                    classId, className = getObjectClass(db)

                    break

        except:
            logger.error(traceback.format_exc())

    return y1,x1,y2,x2,width,height,classId,className

def processDataDir(datapath, writer):

    logger.error("Processing data path [{}]".format(datapath))

    for root, dirs, files in os.walk(datapath):
        
        logger.error("Processing root [{}]".format(root))

        totalFiles = len(files)
        for i in tqdm(range(0,totalFiles)):
            fileName = files[i]        
            filePath = os.path.join(root,fileName)
            #logger.info("Processing.. {} [{} of {}]".format(filePath, i, totalFiles))

            #y1,x1,y2,x2,width,height = lookupFile(os.path.splitext(fileName)[0])
            y1,x1,y2,x2,width,height,classId,className = lookupFile(fileName, filePath)
            if y1 is None:
                logger.warn("Could not get information for [{}] @ [{}]".format(fileName,filePath))
                continue
            #objClass = 1
            # if width/height > 2.2:
            #     objClass = 2

            try:
                writer.writerow([fileName,x1,x2,y1,y2,classId])
            except:
                logger.error(traceback.format_exc())


time_start = time.time()

loadsqlitedbs()
csvfile = open(args['file.name'],'w',newline='',encoding='utf-8')
csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

csvwriter.writerow(['frame','xmin','xmax','ymin','ymax','class_id'])

if args["source.annotate"] is True:
    os.path.makedirs(args["annotate.dir"])

for path in datapaths:
    processDataDir(path, csvwriter)

#clean up part
csvfile.close()
closedbs()

time_end = time.time()

logger.info("Took [{}] s to process request".format(time_end-time_start))