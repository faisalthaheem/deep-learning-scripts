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
import numpy as np
import cv2
import shutil
import random
import math
from tqdm import tqdm


ap = argparse.ArgumentParser()
ap.add_argument("-ln", "--lite.names", required=True,
    help="comma seperated list of sqlite db files to look up image info from")
ap.add_argument("-dp", "--data.path", required=True,
    help="comma seperated list of folders containing images to process")
ap.add_argument('-output_dir', default = 'out.anchors', type = str, 
    help='Output anchor directory\n' )  
ap.add_argument('-num_clusters', default = 2, type = int, 
    help='number of clusters\n' )  
args = vars(ap.parse_args())

#create logger
logger = logging.getLogger('sqlite2anchors')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('sqlite2anchors.log')
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

logger.info("And, will be deleting files from..")
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
                query = "SELECT y1,x1,y2,x2,width,height,imheight,imwidth FROM plates WHERE filename = '{}' and isdeleted = 0 and isbackground=0".format(nameWithoutExtension)
                cursor.execute(query)
                row = cursor.fetchone()
                if row is not None:
                    y1,x1,y2,x2,width,height,imheight,imwidth = int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7])
                    break

        except:
            logger.error(traceback.format_exc())

    return y1,x1,y2,x2,width,height,classId,className

width_in_cfg_file = 640.
height_in_cfg_file = 480.

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file):
    f = open(anchor_file,'w')
    
    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0]*=width_in_cfg_file/32.
        anchors[i][1]*=height_in_cfg_file/32.
         

    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])
        
    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    #there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))
    
    f.write('%f\n'%(avg_IOU(X,centroids)))
    print("")

def kmeans(X,centroids,eps,anchor_file):
    
    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        
        if (assignments == prev_assignments).all() :
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def processDataDir(datapath):

    logger.error("Processing data path [{}]".format(datapath))

    annotation_dims = []

    for root, dirs, files in os.walk(datapath):

        totalFiles = len(files)
        logger.error("Processing [{}] file(s) in root [{}]".format(totalFiles, root))

        for i in tqdm(range(0,totalFiles)):
            fileName = files[i]        
            filePath = os.path.join(root,fileName)
            #logger.info("Processing.. {} [{} of {}]".format(filePath, i, totalFiles))

            #y1,x1,y2,x2,width,height = lookupFile(os.path.splitext(fileName)[0])
            y1,x1,y2,x2,width,height,classId,className = lookupFile(fileName, filePath)
            if y1 is None:
                logger.warn("Could not get information for [{}] @ [{}]".format(fileName,filePath))
                continue
            try:
                w = x2-x1
                h = y2-y1
                #print(w,h)
                
                annotation_dims.append((w,h))
            except:
                logger.error(traceback.format_exc())
    
    annotation_dims = np.array(annotation_dims)
  
    eps = 0.005
    
    if args['num_clusters'] == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = os.path.join( args['output_dir'],'anchors%d.txt'%(num_clusters))

            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims,centroids,eps,anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = os.path.join( args['output_dir'],'anchors%d.txt'%(args['num_clusters']))
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(args['num_clusters'])]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims,centroids,eps,anchor_file)
        print('centroids.shape', centroids.shape)


time_start = time.time()

loadsqlitedbs()

if not os.path.exists(args['output_dir']):
    os.mkdir(args['output_dir'])

for path in datapaths:
    processDataDir(path)

#clean up part
closedbs()
time_end = time.time()
logger.info("Took [{}] s to process request".format(time_end-time_start))