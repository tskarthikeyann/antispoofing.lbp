#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon Oct  7 14:35:23 CEST 2013

import pkg_resources

import os, sys
import bob
import numpy

"""
Utilitary functions to collect features in a matrix for a database and to map scores with the corresponding frame
"""

def create_full_dataset(indir, objects):
  """Creates a full dataset matrix out of all the specified files"""
  dataset = None
  for obj in objects:
    filename = os.path.expanduser(obj.make_path(indir, '.hdf5'))
    fvs = bob.io.load(filename)
    if dataset is None:
      dataset = fvs
    else:
      dataset = numpy.append(dataset, fvs, axis = 0)
  return dataset[~numpy.isnan(dataset).any(axis=1)]  # remove all the Nan elements 

def map_scores(indir, score_dir, objects, score_list):
  """Maps frame scores to frames of the objects. Writes the scores for each frame in a file, NaN for invalid frames

  Keyword parameters:

  indir: the directory with the feature vectors (needed to read which frames are invalid)

  score_dir: the directory where the score files are going to be written

  objects: list of objects

  score_list: list of scores for the given objects
  """
  num_scores = 0 # counter for how many valid frames have been processed so far in total of all the objects
  for obj in objects:
    filename = os.path.expanduser(obj.make_path(indir, '.hdf5'))
    feat = bob.io.load(filename)
    indices = ~numpy.isnan(feat).any(axis=1) #find the indices of invalid frames (they are set to False in the resulting array)
    scores = numpy.ndarray((len(indices), 1), dtype='float64') 
    scores[indices] = score_list[num_scores:num_scores + sum(indices)] # set the scores of the valid frames only
    scores[~indices] = numpy.NaN # set NaN for the scores of the invalid frames
    num_scores += sum(indices) # increase the number of valid scores that have been already maped
    obj.save(scores, score_dir, '.hdf5') # save the scores
