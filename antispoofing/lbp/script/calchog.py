#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Fri 12 Jun 15:55:56 CEST 2015

"""Calculates the normalized HOG features of the faces in each of the frames of the videos in the REPLAY-ATTACK, CASIA_FASD and MSU-MFSD database. The histograms can be computed for a subset of the videos in the database (using the protocols in the database). The output is a single .hdf5 file for each video. The procedure is described in the paper: "On the Effectiveness of Local Binary patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
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


def cut_face_bbx(frame, bbx, sz, bbxsize_filter=0):
  """Calculates the normalized 3x3 LBP histogram over a given bounding box (bbx) in an image (around the detected face for example), using the bob LBP operator, after first rescaling bbx to a predefined size. If bbx is None or invalid, returns an empty histogram.

  Keyword Parameters:

  frame
    The frame as a gray-scale image
  lbptype
    The type of the LBP operator (regular, uniform or riu2)
  bbx
    the face bounding box
  sz
    The size of the rescaled face bounding box
  bbxsize_filter
    Considers as invalid all the bounding boxes with size smaller then this value
  """
  
  if bbx and bbx.is_valid() and bbx.height > bbxsize_filter:
    cutframe = frame[bbx.y:(bbx.y+bbx.height),bbx.x:(bbx.x+bbx.width)] # cutting the box region
    tempbbx = numpy.ndarray((sz, sz), 'float64')
    normbbx = numpy.ndarray((sz, sz), 'uint8')
    bob.ip.base.scale(cutframe, tempbbx) # normalization
    tempbbx_ = tempbbx + 0.5
    tempbbx_ = numpy.floor(tempbbx_)
    normbbx = numpy.cast['uint8'](tempbbx_)

    return normbbx, 1
  else:
    return None, 0  
    



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
  parser.add_argument('-e', '--enrollment', action='store_true', default=False, dest='enrollment', help='If True, will do the processing on the enrollment data of the database (defaults to "%(default)s")')
  parser.add_argument('-c', '--cell', dest='cell', type=int, default=16, help='The size of the cells (defaults to "%(default)s")')
  parser.add_argument('--co', '--cell-overlap', dest="cell_overlap", type=int, default=8, help='The overlap size of the cells (defaults to "%(default)s")')
  parser.add_argument('-b', '--block', dest='block', type=int, default=4, help='The size of the blocks (defaults to "%(default)s")')
  parser.add_argument('--bo', '--block-overlap', dest="block_overlap", type=int, default=1, help='The overlap size of the blocks (defaults to "%(default)s")')
  parser.add_argument('--nn', '--nonorm', dest='nonorm', action='store_true', default=False, help='If True, block normalization of the HOG featurs will NOT be perfomed.')
  parser.add_argument('--bbx', '--boundingbox', action='store_true', default=False, dest='boundingbox', help='If True, will read the face locations using the bbx function of the File class of the database. If False, will use faceloc.read_face utility to read the faceloc. For MSU-MFSD only (defaults to "%(default)s")')

  #######
  # Database especific configuration
  #######
  #Database.create_parser(parser)
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()

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

    hog = bob.ip.base.HOG((sz,sz), cell_size=(args.cell, args.cell), cell_overlap=(args.cell_overlap, args.cell_overlap), block_size=(args.block,args.block), block_overlap=(args.block_overlap, args.block_overlap))
    if args.nonorm:
      hog.disable_block_normalization()
    hog_feat_shape = hog.output_shape()
    
    import ipdb; ipdb.set_trace()
    histdata = numpy.ndarray((0, hog_feat_shape[0] * hog_feat_shape[1] * hog_feat_shape[2]), 'float64')

    numvf = 0 # number of valid frames in the video (will be smaller then the total number of frames if a face is not detected or a very small face is detected in a frame when face lbp are calculated
    validframes = [] # list with the indices of the valid frames

    for k in range(0, vin.shape[0]):
      frame = bob.ip.color.rgb_to_gray(vin[k,:,:,:])
      if string.find(database.short_description(), "MSU") != -1 and obj.is_rotated(): # rotate the frame by 180 degrees if needed
        frame = numpy.rot90(numpy.rot90(frame))  
        #sys.stdout.write("File %s is rotated " % (obj.make_path()))
      
      sys.stdout.write('.')
      sys.stdout.flush()
      
      normbbx, vf = cut_face_bbx(frame, locations[k], sz, bbxsize_filter=0)
      
      numvf = numvf + vf
      validframes.append(vf)
      if vf == 1:
        hogfeats = hog.extract(normbbx)
      else:
        numpy.array(hog_feat_shape[0] * hog_feat_shape[1] * hog_feat_shape[2] * [numpy.NaN])  
      

      histdata = numpy.append(histdata, hogfeats.reshape([1, hog_feat_shape[0] * hog_feat_shape[1] * hog_feat_shape[2]]), axis = 0) # add the histogram into the list of frame feature vectors

    sys.stdout.write('\n')
    sys.stdout.flush()

    # saves the output
    obj.save(histdata, directory = args.directory, extension='.hdf5')
    obj.save(numpy.array(validframes), directory = os.path.join(args.directory, 'validframes'), extension='.hdf5')

  return 0

if __name__ == "__main__":
  main()
