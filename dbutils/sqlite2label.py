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


# Use this script to bulk label your images
# Place all similar images in the same folder and then run this script
# with the labelid and labeltext to update the database

ap = argparse.ArgumentParser()
ap.add_argument("-ln", "--lite.names", required=True,
    help="comma seperated list of sqlite db files to look up image info from")
ap.add_argument("-sp", "--src.path", required=True,
    help="source folder containing files to move")
ap.add_argument('-mode', '--label.mode', default='object',
    help='if mode is "background" then -li and -lt are ignored and db field isbackground is set to 1')
ap.add_argument("-li", "--label.id", default=999,
    help="label id")
ap.add_argument("-lt", "--label.text", default='deflabel',
    help="label text")

args = vars(ap.parse_args())

#create logger
logger = logging.getLogger('sqlite2label')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('sqlite2label.log')
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

logger.info("And, will be labelinig files from..")
logger.info(str(datapaths))

dbconns = {}
dbcursors = {}

isLabelingBackground=False

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


def updateLabel(nameWithoutExtension, filePath):

    for db in sqlitedbs:
        try:
            cursor = dbcursors[db]
            if cursor is not None:
                
                if isLabelingBackground:
                    query = "UPDATE plates set isbackground=1 where filename = '{}'".format(
                        nameWithoutExtension
                    )
                else:    
                    query = "UPDATE plates set labelid={}, labeltext='{}' where filename = '{}'".format(
                        args['label.id'], args['label.text'], nameWithoutExtension
                    )
                cursor.execute(query)
                rowsAffected = cursor.rowcount

                if rowsAffected > 0:
                    dbconns[db].commit()
                else:
                    if isLabelingBackground:
                        #try to insert
                        query = "INSERT INTO plates values('{}','background',0,0,0,0,0,0,'',0,0,1,0,0,0,100,'bg',1)".format(
                            nameWithoutExtension,
                        )
                        cursor.execute(query)
                        rowsAffected = cursor.rowcount

                        if rowsAffected > 0:
                            dbconns[db].commit()
                        else:
                            logger.error("Query failed [{}]".format(query))
                    else:
                        logger.error("Query failed [{}]".format(query))
        except:
            logger.error(traceback.format_exc())

def processDataDir(datapath):

    logger.error("Processing data path [{}]".format(datapath))

    for root, dirs, files in os.walk(datapath):

        totalFiles = len(files)
        logger.error("Processing [{}] file(s) in root [{}]".format(totalFiles, root))

        for i in tqdm(range(0,totalFiles)):
            fileName = files[i]        
            filePath = os.path.join(root,fileName)

            try:
                updateLabel(fileName, filePath)
            except:
                logger.error(traceback.format_exc())


if __name__ == '__main__':

    time_start = time.time()

    if args['label.mode'] == 'background':
        isLabelingBackground = True

    loadsqlitedbs()

    for path in datapaths:
        processDataDir(path)

    #clean up part
    closedbs()
    time_end = time.time()

    logger.info("Took [{}] s to process request".format(time_end-time_start))