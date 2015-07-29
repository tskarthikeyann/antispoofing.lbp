#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon Feb  6 17:41:18 CET 2012

"""Calculates the normalized LBP histogram of the faces in each of the frames of the videos in the REPLAY-ATTACK, CASIA_FASD and MSU-MFSD database. The result are LBP histograms for each frame of the video. Different types of LBP operators are supported. The histograms can be computed for a subset of the videos in the database (using the protocols in the database). The output is a single .hdf5 file for each video. The procedure is described in the paper: "On the Effectiveness of Local Binary patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
"""

import os, sys
import argparse
import bob.io.base
import bob.ip.color
import bob.io.video
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
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str,
      dest='inputdir', default=INPUT_DIR, help='Base directory containing the videos to be treated by this procedure (defaults to "%(default)s")')
  parser.add_argument('-d', '--directory', dest="directory", default=OUTPUT_DIR, help="This path will be prepended to every file output by this procedure (defaults to '%(default)s')")
  parser.add_argument('-n', '--normface-size', dest="normfacesize", default=64, type=int, help="this is the size of the normalized face box if face normalization is used (defaults to '%(default)s')")
  parser.add_argument('--ff', '--facesize_filter', dest="facesize_filter", default=0, type=int, help="all the frames with faces smaller then this number, will be discarded (defaults to '%(default)s')")
  parser.add_argument('-l', '--lbptype', metavar='LBPTYPE', type=str, choices=('regular', 'riu2', 'uniform'), default='uniform', dest='lbptype', help='Choose the type of LBP to use (defaults to "%(default)s")')
  parser.add_argument('--el', '--elbptype', metavar='ELBPTYPE', type=str, choices=('regular', 'transitional', 'direction_coded', 'modified'), default='regular', dest='elbptype', help='Choose the type of extended LBP features to compute (defaults to "%(default)s")')
  parser.add_argument('-b', '--blocks', metavar='BLOCKS', type=int, default=1, dest='blocks', help='The region over which the LBP is calculated will be divided into the given number of blocks squared. The histograms of the individial blocks will be concatenated.(defaults to "%(default)s")')
  parser.add_argument('-c', dest='circular', action='store_true', default=False, help='If set, circular LBP will be computed')
  parser.add_argument('-o', dest='overlap', action='store_true', default=False, help='If set, the blocks on which the image is divided will be overlapping')
  parser.add_argument('-e', '--enrollment', action='store_true', default=False, dest='enrollment', help='If True, will do the processing on the enrollment data of the database (defaults to "%(default)s")')
  parser.add_argument('--nn', '--nonorm', dest='nonorm', action='store_true', default=False, help='If True, normalization on the bounding box will NOT be perfomed. If False, normalization will be done depending on the -n parameter.')
  parser.add_argument('--bbx', '--boundingbox', action='store_true', default=False, dest='boundingbox', help='If True, will read the face locations using the bbx function of the File class of the database. If False, will use faceloc.read_face utility to read the faceloc. For MSU-MFSD only (defaults to "%(default)s")')

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
    input = bob.io.video.reader(obj.videofile(directory=args.inputdir))

    # loading the face locations
    if string.find(database.short_description(), "CASIA") != -1:
      locations = faceloc.read_face(obj.facefile())
    elif string.find(database.short_description(), "MSU") != -1:
      if args.boundingbox: # load the locations based on the bbx function of the File object
        locations = obj.bbx(directory=args.inputdir)
        locations = {x[0]:faceloc.BoundingBox(x[1], x[2], x[3], x[4]) for x in locations} # for MSU MFSD
      else:  
        locations = faceloc.read_face(obj.facefile(args.inputdir)) # for MSU MFSD
    else:  
      locations = faceloc.read_face(obj.facefile(args.inputdir)) # for Replay-Attack
    locations = faceloc.expand_detections(locations, input.number_of_frames)
    sz = args.normfacesize # the size of the normalized face box

    sys.stdout.write("Processing file %s (%d frames) [%d/%d] " % (obj.make_path(),
      input.number_of_frames, counter, len(process)))

    # start the work here...
    vin = input.load() # load the video

    histdata = numpy.ndarray((0,args.blocks * args.blocks * lbphistlength[args.lbptype]), 'float64') # the numpy.ndarray, each row is the histogram of one frame

    numvf = 0 # number of valid frames in the video (will be smaller then the total number of frames if a face is not detected or a very small face is detected in a frame when face lbp are calculated
    validframes = [] # list with the indices of the valid frames

    for k in range(0, vin.shape[0]):
      frame = bob.ip.color.rgb_to_gray(vin[k,:,:,:])
      if string.find(database.short_description(), "MSU") != -1 and obj.is_rotated(): # rotate the frame by 180 degrees if needed
        frame = numpy.rot90(numpy.rot90(frame))  
        #sys.stdout.write("File %s is rotated " % (obj.make_path()))
      
      sys.stdout.write('.')
      sys.stdout.flush()
      if args.nonorm:
        hist, vf = spoof.lbphist_face(frame, args.lbptype, locations[k], args.elbptype, numbl=args.blocks,  circ=args.circular, overlap=args.overlap, bbxsize_filter=args.facesize_filter) # vf = 1 if it was a valid frame, 0 otherwise  
      else:
        hist, vf = spoof.lbphist_facenorm(frame, args.lbptype, locations[k], sz, args.elbptype, numbl=args.blocks,  circ=args.circular, overlap=args.overlap, bbxsize_filter=args.facesize_filter) # vf = 1 if it was a valid frame, 0 otherwise
      numvf = numvf + vf
      validframes.append(vf) # add 0 if it is not a valid frame, 1 in contrary
      #if vf == 1: # if it is a valid frame, add its histogram into the list of frame feature vectors
      histdata = numpy.append(histdata, hist.reshape([1, hist.size]), axis = 0) # add the histogram into the list of frame feature vectors

    sys.stdout.write('\n')
    sys.stdout.flush()

    # saves the output
    obj.save(histdata, directory = args.directory, extension='.hdf5')
    obj.save(numpy.array(validframes), directory = os.path.join(args.directory, 'validframes'), extension='.hdf5')

  return 0

if __name__ == "__main__":
  main()
