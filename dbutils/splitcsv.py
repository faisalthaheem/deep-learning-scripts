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
import time
import pprint
import traceback
import logging
import random

ap = argparse.ArgumentParser()
ap.add_argument("-fn", "--file.name", required=True,
	help="name of csv file")
ap.add_argument("-ft", "--file.train", required=True,
	help="name of csv file to contain training entries")
ap.add_argument("-fv", "--file.validate", required=True,
	help="name of csv file to contain validation entries")
ap.add_argument("-sr", "--split.ratio", default=0.8,
	help="how to split the data set, default is 80/20 where 80\% is train and 20\% is validate")
ap.add_argument("-vm", "--split.maxvalidate", default=10000,
	help="where to cap the validate list, if 20\% of training set is larger than this value, validation set will be restricted to this many samples")
args = vars(ap.parse_args())

#create logger
logger = logging.getLogger('splitcsv.anpr')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('splitcsv.log')
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

content = []
with open(args["file.name"],encoding='utf-8') as f:
    content = f.readlines()

if len(content) == 0:
    logger.info("no content in input file")

header = content[:1]
content = content[1:]


random.shuffle(content)

if (len(content) * (1.0-float(args["split.ratio"]))) > args["split.maxvalidate"]:
    train_end = len(content) - args["split.maxvalidate"]
else:
    train_end = int(len(content) * float(args["split.ratio"]))


train_list = content[:train_end]
val_list = content[train_end+1:]

#write files
with open(args["file.train"],"w",encoding='utf-8') as f:
    f.write("".join(header))
    f.write("".join(train_list))

with open(args["file.validate"],"w",encoding='utf-8') as f:
    f.write("".join(header))
    f.write("".join(val_list))

logger.info("Done with [{}] train samples and [{}] validation samples".format(len(train_list), len(val_list)))
# you may also want to remove whitespace characters like `\n` at the end of each line
#content = [x.strip() for x in content]