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
import shutil as sh
import logging
import sqlite3
import time
import pprint
import traceback
import json

import hashlib
import io
import random
import re


ap = argparse.ArgumentParser()

ap.add_argument("-d1", "--db.v1", required=True,
    help="v1 database")
ap.add_argument("-d2", "--db.v2", required=True,
    help="v2 database")
   
args = vars(ap.parse_args())

#create logger
logger = logging.getLogger('v1tov2')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('v1tov2.log')
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

dbconns = {}
dbcursors = {}

def loadsqlitedbs():
    
    logger.info("Instantiating sqlite dbs")

    for db in [args["db.v1"], args["db.v2"]]:
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

    for db in [args["db.v1"], args["db.v2"]]:
        if dbcursors[db] is not None:
            dbcursors[db].close()

        if dbconns[db] is not None:
            dbconns[db].commit()
            dbconns[db].close()

    logger.info("DBs closed")

def readAllRecordsFromV1():

    rowsRet = []

    try:
        cursor = dbcursors[args["db.v1"]]
        if cursor is not None:
            #look up plate information for the requested name
            query = "SELECT * FROM plates"
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
                rowRet = {
                    "filename": row[0],
                    "y1": row[2],
                    "x1": row[3],
                    "y2": row[4],
                    "x2": row[5],
                    "width": row[6],
                    "height": row[7],
                    "imheight": row[9],
                    "imwidth": row[10],
                    "isreviewed": row[11],
                    "lastreviewedat": row[12],
                    "isdeleted": row[13],
                    "needscropping": row[14],
                    "labelid": row[15],
                    "labeltext": row[16],
                    "isbackground": row[17]
                }

                rowsRet.append(rowRet)

    except:
        logger.error(traceback.format_exc())

    return rowsRet

def saveAllRecordsToV2(rowsToSave):
    try:
        inserts = []
        cursor = dbcursors[args["db.v2"]]
        if cursor is None:
            print("Found closed cursor for v2")
        else:
            for row in rowsToSave:
                imgArea =  {
                    "id": 0,
                    "x": row["x1"],
                    "y": row["y1"],
                    "z": 100,
                    "height": row["height"],
                    "width": row["width"],
                    "lblid": row["labelid"],
                    "lbltxt": row["labeltext"]
                }
                inserts.append((row["filename"], row["imheight"], row["imwidth"], row["isreviewed"], row["lastreviewedat"], row["isdeleted"], row["needscropping"], row["isbackground"], json.dumps([imgArea])))

            query = "REPLACE INTO annotations(filename, imheight, imwidth, isreviewed, lastreviewedat, isdeleted, needscropping, isbackground, imgareas) VALUES(?,?,?,?,?,?,?,?,?)"
            cursor.executemany(query, inserts)
                
    except:
        logger.error(traceback.format_exc())

time_start = time.time()

loadsqlitedbs()

rows = readAllRecordsFromV1()
saveAllRecordsToV2(rows)

closedbs()

time_end = time.time()

logger.info("Took [{}]s to process request".format(time_end-time_start))