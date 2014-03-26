#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Thu Jan 19 15:56:55 CET 2012

"""This script calculates the chi2 difference between a model histogram and the data histograms, assigning scores to the data according to this. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and HTER for the test and development set are calculated. The script outputs a text file with the performance results.
The details about the procedure are described in the paper: "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
"""

import os, sys
import argparse
import bob
import numpy

from antispoofing.utils.db import *
from antispoofing.utils.ml import *
    
def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  INPUT_MODEL_DIR = os.path.join(basedir, 'res')
  OUTPUT_DIR = os.path.join(basedir, 'res')

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')
  parser.add_argument('-m', '--input-modeldir', metavar='DIR', type=str, dest='inputmodeldir', default=INPUT_MODEL_DIR, help='Base directory containing the histogram models to be loaded')
  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-s', '--score', dest='score', action='store_true', default=False, help='If set, the final classification scores of all the frames will be dumped in a file')

  from .. import spoof, helpers
  from ..spoof import chi2
  from ..helpers import score_manipulate as sm
   
  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()

  if not os.path.exists(args.inputdir) or not os.path.exists(args.inputmodeldir):
    parser.error("input directory does not exist")

  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)
    
  print "Output directory set to \"%s\"" % args.outputdir
  print "Loading input files..."

  # loading the input files
  database = args.cls(args)
  process_train_real, process_train_attack = database.get_train_data()
  process_devel_real, process_devel_attack = database.get_devel_data()
  process_test_real, process_test_attack = database.get_test_data()

  # create the full datasets from the file data
  devel_real = sm.create_full_dataset(args.inputdir, process_devel_real); devel_attack = sm.create_full_dataset(args.inputdir, process_devel_attack); 
  test_real = sm.create_full_dataset(args.inputdir, process_test_real); test_attack = sm.create_full_dataset(args.inputdir, process_test_attack); 
  train_real = sm.create_full_dataset(args.inputdir, process_train_real); train_attack = sm.create_full_dataset(args.inputdir, process_train_attack); 

  print "Loading the models..."
  # loading the histogram models
  histmodelsfile = bob.io.HDF5File(os.path.join(args.inputmodeldir, 'histmodelsfile.hdf5'),'r')
  model_hist_real = histmodelsfile.read('model_hist_real')
  del histmodelsfile
    
  model_hist_real = model_hist_real[0,:]

  print "Calculating the Chi-2 differences..."
  # calculating the comparison scores with chi2 distribution for each protocol subset   
  sc_devel_realmodel = chi2.cmphistbinschimod(model_hist_real, (devel_real, devel_attack))
  sc_test_realmodel = chi2.cmphistbinschimod(model_hist_real, (test_real, test_attack))
  sc_train_realmodel = chi2.cmphistbinschimod(model_hist_real, (train_real, train_attack))

  print "Saving the results in a file"
  # It is expected that the positives always have larger scores. Therefore, it is necessary to "invert" the scores by multiplying them by -1 (the chi-square test gives smaller scores to the data from the similar distribution)
  sc_devel_realmodel = (sc_devel_realmodel[0] * -1, sc_devel_realmodel[1] * -1)
  sc_test_realmodel = (sc_test_realmodel[0] * -1, sc_test_realmodel[1] * -1)
  sc_train_realmodel = (sc_train_realmodel[0] * -1, sc_train_realmodel[1] * -1)

  if args.score: # save the scores in a file
    score_dir = os.path.join(args.outputdir, 'scores') # output directory for the socre files
    sm.map_scores(args.inputdir, score_dir, process_devel_real, sc_devel_realmodel[0]) 
    sm.map_scores(args.inputdir, score_dir, process_devel_attack, sc_devel_realmodel[1])
    sm.map_scores(args.inputdir, score_dir, process_test_real, sc_test_realmodel[0])
    sm.map_scores(args.inputdir, score_dir, process_test_attack, sc_test_realmodel[1])
    sm.map_scores(args.inputdir, score_dir, process_train_real, sc_train_realmodel[0])
    sm.map_scores(args.inputdir, score_dir, process_train_attack, sc_train_realmodel[1])

  # calculation of the error rates
  thres = bob.measure.eer_threshold(sc_devel_realmodel[1].flatten(), sc_devel_realmodel[0].flatten())
  dev_far, dev_frr = bob.measure.farfrr(sc_devel_realmodel[1].flatten(), sc_devel_realmodel[0].flatten(), thres)
  test_far, test_frr = bob.measure.farfrr(sc_test_realmodel[1].flatten(), sc_test_realmodel[0].flatten(), thres)
  
  # writing results to a file
  tbl = []
  tbl.append(" ")
  tbl.append(" threshold: %.4f" % thres)
  tbl.append(" dev:  FAR %.2f%% (%d / %d) | FRR %.2f%% (%d / %d) | HTER %.2f%% " % \
      (100*dev_far, int(round(dev_far*len(devel_attack))), len(devel_attack), 
       100*dev_frr, int(round(dev_frr*len(devel_real))), len(devel_real),
       50*(dev_far+dev_frr)))
  tbl.append(" test: FAR %.2f%% (%d / %d) | FRR %.2f%% (%d / %d) | HTER %.2f%% " % \
      (100*test_far, int(round(test_far*len(test_attack))), len(test_attack),
       100*test_frr, int(round(test_frr*len(test_real))), len(test_real),
       50*(test_far+test_frr)))
  txt = ''.join([k+'\n' for k in tbl])
  print txt

  # write the results to a file 
  tf = open(os.path.join(args.outputdir, 'perf_table.txt'), 'w')
  tf.write(txt)
  
if __name__ == '__main__':
  main()
