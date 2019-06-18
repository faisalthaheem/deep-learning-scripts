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

import pprint
import sys
import os
import argparse as argparse
import logging
import shutil

#create logger
logger = logging.getLogger('flattenfolder')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('flattenfolder.log')
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

ap = argparse.ArgumentParser()
ap.add_argument("-sd", "--sources.dir", required=True,
        help="folder with nested folders")
ap.add_argument("-dd", "--dest.dir", required=True,
        help="folder where all files are moved")
args = vars(ap.parse_args())


def processImage(filePath, filename):

        dest_path = os.path.join(args["dest.dir"], filename)
        shutil.move(filePath, dest_path)

        #return False


def processSourceDir():
    filesList = os.listdir()
    for root, dirs, files in os.walk(args["sources.dir"]):
        for fileName in files:
            logger.info("Processing.. " + os.path.join(root,fileName))
            if False == processImage(os.path.join(root,fileName), fileName):
                    break

logger.info("")
logger.info("Commencing flattening....")
processSourceDir()