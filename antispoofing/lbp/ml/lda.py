#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 19 Sep 2011 15:01:44 CEST 

"""LDA training for the anti-spoofing library
"""

import bob
import numpy

def make_lda(train, verbose=False):
  """Creates a new linear machine and train it using LDA.

  Keyword Parameters:

  train
    An iterable (tuple or list) containing two arraysets: the first contains
    the real accesses and the second contains the attacks.

  verbose
    Makes the training more verbose
  """

  T = bob.trainer.FisherLDATrainer()
  machine, eig_vals = T.train(train)
  return machine

def get_scores(machine, data):
  """Gets the scores for the data.

  Keyword Parameters:

  machine
    bob.machine.LinearMachine

  data
    numpy.ndarray containing the data that need to be projected
"""

  return numpy.vstack([machine(d) for d in data])[:,0]
