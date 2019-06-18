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

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import sqlite3
import time
import pprint
import traceback
import logging
import csv
import json
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import errno
import string

ap = argparse.ArgumentParser()
ap.add_argument("-ln", "--lite.names", required=True,
	help="comma seperated list of sqlite db files to look up image info from")
ap.add_argument("-dp", "--data.path", required=True,
	help="comma seperated list of folders containing images to process")
args = vars(ap.parse_args())

#create logger
logger = logging.getLogger('sqlite_updatedims.anpr')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('sqlite_updatedims.log')
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


sqlitedbs = args["lite.names"].split(',')
datapaths = args["data.path"].split(',')

logger.info("Will be looking up following databases for info...")
logger.info(str(sqlitedbs))

logger.info("And, will be processing files from...")
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
			dbconns[db].commit()
			dbcursors[db].close()

		if dbconns[db] is not None:
			dbconns[db].close()

	logger.info("DBs closed")

def lookupFile(nameWithoutExtension):

	for db in sqlitedbs:
		try:
			cursor = dbcursors[db]
			if cursor is not None:
				#look up plate information for the requested name
				query = "SELECT count(*) FROM plates WHERE filename = '{}'".format(nameWithoutExtension)
				cursor.execute(query)

				return cursor.fetchone()[0], db

		except:
			logger.error(traceback.format_exc())

	return 0, None

def updateDetails(whichdb, filename, height, width):
	
	try:
		query = "Update plates set imheight = {}, imwidth = {} WHERE filename = '{}'".format(height, width, filename)
		dbcursors[whichdb].execute(query)

	except:
		logger.error(traceback.format_exc())

def processDataDir(datapath):

	logger.info("Processing data path [{}]".format(datapath))

	for root, dirs, files in os.walk(datapath):
		
		logger.info("Processing root [{}]".format(root))

		totalFiles = len(files)
		for i in tqdm(range(0,totalFiles)):
			fileName = files[i]		   
			filePath = os.path.join(root,fileName)
			#logger.info("Processing.. {} [{} of {}]".format(filePath, i, totalFiles))

			found, whichdb = lookupFile(fileName)
			if found > 0:
				img = Image.open(filePath)
				width, height = img.size

				updateDetails(whichdb, fileName, height, width)

				continue
			
			#img_data = image.load_img(filePath, grayscale = False)
			
			#extract plate


time_start = time.time()

loadsqlitedbs()

for path in datapaths:
	processDataDir(path)

#clean up part
closedbs()

time_end = time.time()

logger.info("Took [{}] s to process request".format(time_end-time_start))