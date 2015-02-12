#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Thu Jan 19 15:24:14 CET 2012

"""This script makes a histogram models for the real accesses videos in REPLAY-ATTACK by averaging the LBP histograms of each real access video. The output is an hdf5 file with the computed model histograms. The procedure is described in the paper: "On the Effectiveness of Local Binary patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
"""

import os, sys
import argparse
import bob.io.base
import numpy

from antispoofing.utils.db import *

def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  OUTPUT_DIR = os.path.join(basedir, 'res')
  
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the histogram features of all the videos')
  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results (models).')
  
  from ..helpers import score_manipulate as sm
  
  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()

  if not os.path.exists(args.inputdir):
    parser.error("input directory does not exist")
  
  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.io.base.create_directories_safe(args.outputdir)
    
  print "Output directory set to \"%s\"" % args.outputdir
  print "Loading input files..."
  # loading the input files
  database = args.cls(args)
  process_train_real, process_train_attack = database.get_train_data()

  # create the full datasets from the file data
  train_real = sm.create_full_dataset(args.inputdir, process_train_real);
  
  print "Creating the model..."

  model_hist_real = [sum(train_real[:,i]) for i in range(0, train_real.shape[1])] # sum the histograms of the real access videos
  
  model_hist_real = [i / train_real.shape[0] for i in model_hist_real]  # average the model histogram for the real access videos

  print "Saving the model histograms..."
  histmodelsfile = bob.io.base.HDF5File(os.path.join(args.outputdir, 'histmodelsfile.hdf5'),'w')
  histmodelsfile.append('model_hist_real', numpy.array(model_hist_real))

  del histmodelsfile
 
if __name__ == '__main__':
  main()
