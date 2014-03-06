#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Thu Jan 19 12:53:56 CET 2012

"""Calculates the frame accumulated and then averaged LBP histogram of the normalized faces in the videos in the REPLAY-ATTACK (or CASIA-FASD) database. The result is the average LBP histogram over all the frames of the video. Different types of LBP operators are supported. The histograms can be computed for a subset of the videos in the database (using the protocols in the database). The output is a single .hdf5 file for each video. The procedure is described in the paper: "On the Effectiveness of Local Binary patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
"""

import os, sys
import argparse
import bob
import numpy
import math
import string

import antispoofing.utils.faceloc as faceloc
from antispoofing.utils.db import *

def main():
  
  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'database')
  OUTPUT_DIR = os.path.join(basedir, 'lbp_features')

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the videos to be treated by this procedure (defaults to "%(default)s")')
  parser.add_argument('-d', '--directory', dest="directory", default=OUTPUT_DIR, help="This path will be prepended to every file output by this procedure (defaults to '%(default)s')")
  parser.add_argument('-n', '--normface-size', dest="normfacesize", default=64, type=int, help="this is the size of the normalized face box if face normalization is used (defaults to '%(default)s')")
  parser.add_argument('--ff', '--facesize_filter', dest="facesize_filter", default=0, type=int, help="all the frames with faces smaller then this number, will be discarded (defaults to '%(default)s')")
  parser.add_argument('-l', '--lbptype', metavar='LBPTYPE', type=str, choices=('regular', 'riu2', 'uniform'), default='uniform', dest='lbptype', help='Choose the type of LBP to use (defaults to "%(default)s")')
  parser.add_argument('--el', '--elbptype', metavar='ELBPTYPE', type=str, choices=('regular', 'transitional', 'direction_coded', 'modified'), default='regular', dest='elbptype', help='Choose the type of extended LBP features to compute (defaults to "%(default)s")')
  parser.add_argument('-b', '--blocks', metavar='BLOCKS', type=int, default=1, dest='blocks', help='The region over which the LBP is calculated will be divided into the given number of blocks squared. The histograms of the individial blocks will be concatenated.(defaults to "%(default)s")')
  parser.add_argument('-c', dest='circular', action='store_true', default=False, help='If set, circular LBP will be computed')
  parser.add_argument('-o', dest='overlap', action='store_true', default=False, help='If set, the blocks on which the image is divided will be overlapping')
  parser.add_argument('-e', '--enrollment', action='store_true', default=False, dest='enrollment', help='If True, will do the processing of the enrollment data of the database (defaults to "%(default)s")')
  
  #######
  # Database especific configuration
  #######
  #Database.create_parser(parser)
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()

  lbphistlength = {'regular':256, 'riu2':10, 'uniform':59} # hardcoding the number of bins for the LBP variants

  from .. import spoof

  ########################
  #Querying the database
  ########################
  #database = new_database(databaseName,args=args)
  database = args.cls(args)
  if args.enrollment:  
    process = database.get_enroll_data()
  else:  
    realObjects, attackObjects = database.get_all_data()
    process = realObjects + attackObjects 

  counter = 0
  # process each video
  for obj in process:
    counter += 1
    input = bob.io.VideoReader(obj.videofile(directory=args.inputdir))

    # loading the face locations
    if string.find(database.short_description(), "CASIA") != -1:
      locations = faceloc.read_face(obj.facefile())
    else:
      locations = faceloc.read_face(obj.facefile(args.inputdir))  
    locations = faceloc.expand_detections(locations, input.number_of_frames)
    sz = args.normfacesize # the size of the normalized face box
   
    sys.stdout.write("Processing file %s (%d frames) [%d/%d] " % (obj.make_path(),
      input.number_of_frames, counter, len(process)))

    # start the work here...
    vin = input.load() # load the video
    
    data = numpy.array(args.blocks * args.blocks * lbphistlength[args.lbptype] * [0.]) # initialize the accumulated histogram	for uniform LBP

    numvf = 0 # number of valid frames in the video (will be smaller then the total number of frames if a face is not detected or a very small face is detected in a frame when face lbp are calculated   

    for k in range(0, vin.shape[0]): 
      frame = bob.ip.rgb_to_gray(vin[k,:,:,:])
      sys.stdout.write('.')
      sys.stdout.flush()
      hist, vf = spoof.lbphist_facenorm(frame, args.lbptype, locations[k], sz, args.elbptype, numbl=args.blocks,  circ=args.circular, overlap=args.overlap, bbxsize_filter=args.facesize_filter) # vf = 1 if it was a valid frame, 0 otherwise
      numvf = numvf + vf
      if vf == 1: # if it is a valid frame, add this histogram into the accumulated histogram
        data = data + hist # accumulate the histograms of all the frames one by one
          
    data = data / numvf # averaging over the number of valied frames
    
    sys.stdout.write('\n')
    sys.stdout.flush()

    # saves the output
    obj.save(data.reshape([1,data.size]), directory = args.directory, extension='.hdf5')

  return 0

if __name__ == "__main__":
  main()
