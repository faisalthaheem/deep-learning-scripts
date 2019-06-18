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


ap = argparse.ArgumentParser()
ap.add_argument("-sd", "--source.dir", required=True,
	help="Directory containing files to look for in dest")
ap.add_argument("-dd", "--destination.dir", required=True,
	help="destination where files have to be deleted from")
	
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("rmbyref.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)



def resizeImage(tuple):
	try:
		global targetWidth, targetHeight

		img = Image.open(tuple[0])
		img = img.resize((targetWidth,targetHeight))
		if img.mode != 24:
			img = img.convert("RGB")
		img.save(tuple[1])

	except Exception as e:
		logging.error(e)


# verify the directories exist
if not os.path.exists(args["source.dir"]) or not os.path.isdir(args["source.dir"]):
	logging.error("%s is not a directory or does not exist." % args["source.dir"])
	sys.exit()

if not os.path.exists(args["destination.dir"]):
	logging.warn("%s is not a directory or does not exist, will try to create." % args["destination.dir"])
	os.mkdir(args["destination.dir"])
	if not os.path.isdir(args["destination.dir"]):
		logging.error("Unable to create destination path. Will exit now")
		sys.exit()
		
if args["destination.dir"] == args["source.dir"]:
	logging.error("Source and destination cannot be the same")
	sys.exit()

#list containing dest images to process
items_to_process = []

logging.info("Processing files in " + args["source.dir"])
for root, dirs, files in os.walk(args["source.dir"]):
	
	totalFiles = len(files)
	logging.info("[{}] files to process".format(totalFiles))

	for i in range(0,totalFiles):
		fileName = files[i]        
		filePath = os.path.join(root,fileName)
		#logger.info("Processing.. {} [{} of {}]".format(filePath, i, totalFiles))
		dest_file = os.path.join(args["destination.dir"], fileName)
		items_to_process.append((filePath, dest_file))

logging.info("Commencing parallel processing jobs")			
results = Parallel(n_jobs=4, backend="threading")(map(delayed(resizeImage), items_to_process))
items_to_process = []
logging.info("Done..")
