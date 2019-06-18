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
import random
import tqdm

#create logger
logger = logging.getLogger('randomcopy')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('randomcopy.log')
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
ap.add_argument("-sd", "--source.dir", required=True,
        help="folder to copy files from")
ap.add_argument("-dd", "--dest.dir", required=True,
        help="folder where all files are copied")
ap.add_argument("-num", "--num.copy", required=True,
        help="number of files to copy")
ap.add_argument("-move", dest="action.move", action='store_true',
        help="whether to move files instead of copying them")
args = vars(ap.parse_args())

fyles = os.listdir(args["source.dir"])
random.shuffle(fyles)

numToCopy = int(args["num.copy"])

if len(fyles) < numToCopy:
    logger.error("Discovered files @[{}] = [{}], while asked to copy [{}]".format(args["source.dir"], numToCopy, args["num.copy"]))
    sys.exit()

fyles = fyles[numToCopy:]

is_moving = False
if args['action.move'] is True:
    is_moving = True
    logger.warn("Moving files flag is set to {}".format(args['action.move']))

for i in tqdm.tqdm(range(0,numToCopy)):
    fyli = fyles[i]
    srcPath = os.path.join(args["source.dir"], fyli)
    dstPath = os.path.join(args["dest.dir"], fyli)

    if is_moving is True:
        shutil.move(srcPath, dstPath)
    else:
        shutil.copy(srcPath, dstPath)

logger.info('Done')