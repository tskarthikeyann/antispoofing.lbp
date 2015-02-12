#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon Mar 24 19:52:59 CET 2014

"""This script can makes an SVM classification of data into two categories: real accesses and spoofing attacks. It reads the SVM file previously computed and stored in a file.

The probabilities obtained with the SVM are considered as scores for the data. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and HTER for the test and development set are calculated. The script outputs a text file with the performance results.

The details about the procedure are described in the paper: "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
"""

import os, sys
import argparse
import bob.io.base
import bob.learn.linear
import bob.learn.libsvm
import bob.measure
import numpy

from antispoofing.utils.db import *
from antispoofing.utils.ml import *


def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  OUTPUT_DIR = os.path.join(basedir, 'res')

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')
  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-i', '--infile', type=str, dest='infile', default='res/svm_machine.hdf5', help='File containing the SVM machine and parameters to be loaded')
  parser.add_argument('-s', '--score', dest='score', action='store_true', default=False, help='If set, the final classification scores of all the frames will be dumped in a file')

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

  print "Reading input file with SVM machine and parameters"
  fin = bob.io.base.HDF5File(os.path.join(args.infile), 'r')
  if fin.has_group('min-max-norm'):
    fin.cd('min-max-norm')
    mins = fin.get_attribute('mins')
    maxs = fin.get_attribute('maxs')
    fin.cd('..')
  else:
    mins = None; maxs = None  
    
  if fin.has_group('stdnorm'):
    fin.cd('stdnorm')
    mean = fin.get_attribute('mean')
    std = fin.get_attribute('std')
    fin.cd('..')
  else:
    mean = None; std = None  
  
    
  if fin.has_group('pca_machine'):
    fin.cd('pca_machine')
    pca_machine = bob.learn.linear.Machine(fin)
    fin.cd('..')
  else:
    pca_machine = None  
  fin.cd('svm_machine')      
  svm_machine = bob.learn.libsvm.Machine(fin)
  fin.cd('/')

  print "Loading input files..."
  # loading the input files
  database = args.cls(args)
  process_train_real, process_train_attack = database.get_train_data()
  process_devel_real, process_devel_attack = database.get_devel_data()
  process_test_real, process_test_attack = database.get_test_data()

  # create the full datasets from the file data
  train_real = sm.create_full_dataset(args.inputdir, process_train_real); train_attack = sm.create_full_dataset(args.inputdir, process_train_attack); 
  devel_real = sm.create_full_dataset(args.inputdir, process_devel_real); devel_attack = sm.create_full_dataset(args.inputdir, process_devel_attack); 
  test_real = sm.create_full_dataset(args.inputdir, process_test_real); test_attack = sm.create_full_dataset(args.inputdir, process_test_attack); 

  if mins != None and maxs != None:  # normalization in the range [-1, 1] (recommended by LIBSVM)
    train_real = norm.norm_range(train_real, mins, maxs, -1, 1); train_attack = norm.norm_range(train_attack, mins, maxs, -1, 1)
    devel_real = norm.norm_range(devel_real, mins, maxs, -1, 1); devel_attack = norm.norm_range(devel_attack, mins, maxs, -1, 1)
    test_real = norm.norm_range(test_real, mins, maxs, -1, 1); test_attack = norm.norm_range(test_attack, mins, maxs, -1, 1)
    
  if mean != None and std != None:  # standard normalization
    train_real = norm.zeromean_unitvar_norm(train_real, mean, std); train_attack = norm.zeromean_unitvar_norm(train_attack, mean, std)
    devel_real = norm.zeromean_unitvar_norm(devel_real, mean, std); devel_attack = norm.zeromean_unitvar_norm(devel_attack, mean, std)
    test_real = norm.zeromean_unitvar_norm(test_real, mean, std); test_attack = norm.zeromean_unitvar_norm(test_attack, mean, std)
  
  if pca_machine != None: # PCA dimensionality reduction of the data
    train_real = pca.pcareduce(pca_machine, train_real); train_attack = pca.pcareduce(pca_machine, train_attack)
    devel_real = pca.pcareduce(pca_machine, devel_real); devel_attack = pca.pcareduce(pca_machine, devel_attack)
    test_real = pca.pcareduce(pca_machine, test_real); test_attack = pca.pcareduce(pca_machine, test_attack)

  def svm_predict(svm_machine, data):
    labels = [svm_machine.predict_class_and_scores(x)[1][0] for x in data]
    return labels
  
  print "Computing devel and test scores..."
  devel_real_out = svm_predict(svm_machine, devel_real);
  devel_attack_out = svm_predict(svm_machine, devel_attack);
  test_real_out = svm_predict(svm_machine, test_real);
  test_attack_out = svm_predict(svm_machine, test_attack);
  train_real_out = svm_predict(svm_machine, train_real);
  train_attack_out = svm_predict(svm_machine, train_attack);

  # it is expected that the scores of the real accesses are always higher then the scores of the attacks. Therefore, a check is first made, if the average of the scores of real accesses is smaller then the average of the scores of the attacks, all the scores are inverted by multiplying with -1.
  if numpy.mean(devel_real_out) < numpy.mean(devel_attack_out):
    devel_real_out = devel_real_out * -1; devel_attack_out = devel_attack_out * -1
    test_real_out = test_real_out * -1; test_attack_out = test_attack_out * -1
    train_real_out = train_real_out * -1; train_attack_out = train_attack_out * -1
    
  if args.score: # save the scores in a file
    score_dir = os.path.join(args.outputdir, 'scores') # output directory for the socre files
    sm.map_scores(args.inputdir, score_dir, process_devel_real, numpy.reshape(devel_real_out, [len(devel_real_out), 1])) 
    sm.map_scores(args.inputdir, score_dir, process_devel_attack, numpy.reshape(devel_attack_out, [len(devel_attack_out), 1]))
    sm.map_scores(args.inputdir, score_dir, process_test_real, numpy.reshape(test_real_out, [len(test_real_out), 1]))
    sm.map_scores(args.inputdir, score_dir, process_test_attack, numpy.reshape(test_attack_out, [len(test_attack_out), 1]))
    sm.map_scores(args.inputdir, score_dir, process_train_real, numpy.reshape(train_real_out, [len(train_real_out), 1]))
    sm.map_scores(args.inputdir, score_dir, process_train_attack, numpy.reshape(train_attack_out, [len(train_attack_out), 1]))

  # calculation of the error rates
  thres = bob.measure.eer_threshold(devel_attack_out, devel_real_out)
  dev_far, dev_frr = bob.measure.farfrr(devel_attack_out, devel_real_out, thres)
  test_far, test_frr = bob.measure.farfrr(test_attack_out, test_real_out, thres)
  
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
