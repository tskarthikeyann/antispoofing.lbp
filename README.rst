=============================================================================
 Counter-Measures to Face Spoofing Attacks using Local Binary Patterns (LBP)
=============================================================================

This package implements the LBP counter-measure to spoofing attacks to face
recognition systems as described at the paper `On the Effectiveness of Local
Binary Patterns in Face Anti-spoofing`, by Chingovska, Anjos and Marcel,
presented at the IEEE BioSIG 2012 meeting.

If you use this package and/or its results, please cite the following
publications:

1. The `original paper <http://publications.idiap.ch/downloads/papers/2012/Chingovska_IEEEBIOSIG2012_2012.pdf>`_ with the counter-measure explained in details::

    @INPROCEEDINGS{Chingovska_BIOSIG_2012,
    author = {Chingovska, Ivana and Anjos, Andr{\'{e}} and Marcel, S{\'{e}}bastien},
    keywords = {Attack, Counter-Measures, Counter-Spoofing, Face Recognition, Liveness Detection, Replay, Spoofing},
    month = sep,
    title = {On the Effectiveness of Local Binary Patterns in Face Anti-spoofing},
    journal = {IEEE BIOSIG 2012},
    year = {2012},
    }
 
2. Bob_ as the core framework used to run the experiments::

    @inproceedings{Anjos_ACMMM_2012,
        author = {A. Anjos AND L. El Shafey AND R. Wallace AND M. G\"unther AND C. McCool AND S. Marcel},
        title = {Bob: a free signal processing and machine learning toolbox for researchers},
        year = {2012},
        month = oct,
        booktitle = {20th ACM Conference on Multimedia Systems (ACMMM), Nara, Japan},
        publisher = {ACM Press},
    }

If you wish to report problems or improvements concerning this code, please
contact the authors of the above mentioned papers.

Raw data
--------

The data used in the paper is publicly available and should be downloaded and
installed **prior** to try using the programs described in this package. Visit
`the REPLAY-ATTACK database portal
<https://www.idiap.ch/dataset/replayattack>`_ for more information.

This satellite package can also work with the `CASIA_FASD database <http://www.cbsr.ia.ac.cn/english/FaceAntiSpoof%20Databases.asp>`_. 

Installation
------------

.. note:: 

  If you are reading this page through our GitHub portal and not through PyPI,
  note **the development tip of the package may not be stable** or become
  unstable in a matter of moments.

  Go to `http://pypi.python.org/pypi/antispoofing.lbp
  <http://pypi.python.org/pypi/antispoofing.lbp>`_ to download the latest
  stable version of this package. Then, extract the .zip file to a folder of your choice.

The ``antispoofing.lbp`` package is a satellite package of the free signal processing and machine learning library Bob_. This dependency has to be downloaded manually. This version of the package depends on Bob_ version 2 or greater. To install `packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_, please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`_. For Bob_ to be able to work properly, some dependent Bob packages are required to be installed. Please make sure that you have read the Dependencies for your operating system.

The most simple solution is to download and extract ``antispoofing.lbp`` package, then to go to the console and write::

  $ cd antispoofing.lbp
  $ python bootstrap-buildout.py
  $ bin/buildout

This will download all required dependent Bob_ and other packages and install them locally. 


User Guide
----------

This section explains how to use the package in order to: a) calculate the LBP
features on the REPLAY-ATTACK or CASIA_FASD database; b) perform classification using Chi-2,
Linear Discriminant Analysis (LDA) and Support Vector Machines (SVM). At the bottom of the page, you can find instructions how to reproduce the exact paper results.

It is assumed you have followed the installation instructions for the package,
and got the required database downloaded and uncompressed in a directory.
After running the ``buildout`` command, you should have all required utilities
sitting inside the ``bin`` directory. We expect that the video files of the database are installed in a sub-directory called
``database`` at the root of the package.  You can use a link to the location of
the database files, if you don't want to have the database installed on the
root of this package::

  $ ln -s /path/where/you/installed/the/database database

If you don't want to create a link, use the ``--input-dir`` flag (available in
all the scripts) to specify the root directory containing the database files.
That would be the directory that *contains* the sub-directories ``train``,
``test``, ``devel`` and ``face-locations``.

Calculate the LBP features
==========================

The first stage of the process is calculating the feature vectors, which are
essentially normalized LBP histograms. There are two types of feature vectors:

1. per-video averaged feature-vectors (the normalized LBP histograms for each
   frame, averaged over all the frames of the video. The result is a single
   feature vector for the whole video), or

2. a single feature vector for each frame of the video (saved as a multiple row
   array in a single file). 

The program to be used for the first case is ``./bin/calclbp.py``, and for the
second case ``./bin/calcframelbp.py``. They both use the utility script
``spoof/calclbp.py``. Depending on the command line arguments, they can compute
different types of LBP histograms over the normalized face bounding box.
Furthermore, the normalized face-bounding box can be divided into blocks or
not.

The following command will calculate the per-video averaged feature vectors of
all the videos in the REPLAY-ATTACK database and will put the resulting
``.hdf5`` files with the extracted feature vectors in the default output
directory ``./lbp_features``::

  $ ./bin/calclbp.py --ff 50 replay

In the above command, the face size filter is set to 50 pixels (as in the
paper), and the program will discard all the frames with detected faces smaller
then 50 pixels as invalid.

To calculate the feature vectors for each frame separately (and save them into a single file for the full video), you have to run::

$ ./bin/calcframelbp.py --ff 50 replay

To see all the options for the scripts ``calclbp.py`` and ``calcframelbp.py``,
just type ``--help`` at the command line. Change the default option in order to
obtain various features, as described in the paper. 

If you want to see all the options for a specific database (e.g. protocols, lighting conditions etc.), type the following command (for Replay-Attack)::
 
  $ ./bin/calclbp.py replay --help

Classification using Chi-2 distance
===================================

The clasification using Chi-2 distance consists of two steps. The first one is
creating the histogram model (average LBP histogram of all the real access
videos in the training set). The second step is comparison of the features of
development and test videos to the model histogram and writing the results.

The script to use for creating the histogram model is
``./bin/mkhistmodel.py``.  It expects that the LBP features of the videos are
stored in a folder ``./lbp_features``. The model histogram will be written
in the default output folder ``./res``. You can change this default features by
setting the input arguments. To execute this script for Replay-Attack, just run::

  $ ./bin/mkhistmodel.py replay

The script for performing Chi-2 histogram comparison is
``./bin/cmphistmodels.py``, and it assumes that the model histogram has been
already created. It makes use of the utility script ``spoof/chi2.py``. The default input directory is
``./lbp_features``, while the default input directory for the histogram model
as well as default output directory is ``./res``. To execute this script for Replay-Attack, just
run:: 

  $ ./bin/cmphistmodels.py -s replay

Do not forget the ``-s`` option if you want the scores for each video saved in a file.

To see all the options for the scripts ``mkhistmodel.py`` and
``cmphistmodels.py``, just type ``--help`` at the command line.

Classification with linear discriminant analysis (LDA)
======================================================

The classification with LDA is performed using the script
``./bin/ldatrain_lbp.py``. The default input and output directories
are ``./lbp_features`` and ``./res``. To execute the script with prior PCA
dimensionality reduction as is done in the paper (for Replay-Attack), call::

  $ ./bin/ldatrain_lbp.py -r -s replay

Do not forget the ``-s`` option if you want the scores for each video saved in a file.

To see all the options for this script, just type ``--help`` at the command
line.

Classification with support vector machine (SVM)
================================================

The classification with SVM is performed using the script
``./bin/svmtrain_lbp.py``. The default input and output directories are ``./lbp_features``
and ``./res``. To execute the script with prior normalization of the data in
the range ``[-1, 1]`` as in the paper (for Replay-Attack), call::

  $ ./bin/svmtrain_lbp.py -n --eval -s replay

Do not forget the ``-s`` option if you want the scores for each video saved in a file.

To see all the options for this script, just type ``--help`` at the command
line.

Classification with support vector machine (SVM) on a different database or database subset
===========================================================================================

In the training process, the SVM machine, as well as the normalization and PCA parameters are saved in an .hdf5 file. They can be used later for classification of data from a different database or database subset. This can be done using the script
``./bin/svmtrain_lbp.py``. The default input and output directories are ``./lbp_features``
and ``./res``. To execute the script, call::

  $ ./bin/svmeval_lbp.py replay

Do not forget the ``-s`` option if you want the scores for each video saved in a file. Also, do not forget to specify the right .hdf5 file where the SVM machine and the parameters are saved using the ``-i`` parameter (the default one is ``./res/svm_machine.hdf5`` 

To see all the options for this script, just type ``--help`` at the command
line.


Reproduce paper results
=======================

The exact commands to reproduce the results from the paper are given here. First, feature exatraction should be done as follows::

  $ ./bin/calcframelbp.py -d features/regular replay 
  $ ./bin/calcframelbp.py -d features/transitional replay
  $ ./bin/calcframelbp.py -d features/direction_coded replay
  $ ./bin/calcframelbp.py -d features/modified replay
  $ ./bin/calcframelbp.py -d features/per-block -b 3 replay
  
The results in Table II are obtained with the following commands::

  $ ./bin/mkhistmodel.py -v features/regular -d models/regular replay
  $ ./bin/cmphistmodels.py -v features/regular -m models/regular -d scores/regular -s replay
  
By changing the ``-v`` parameter, you can change the type of features, resulting in the scores for the different columns of the table.

The results in Table III are obtained by the same commands, using the corresponding value for the ``-v`` parameter for the per-block computed feature.

The results in Table IV for LDA and SVM classification are obtained by the following two commands, respectively::

  $ ./bin/ldatrain_lbp.py -v features/regular -d scores/regular -n replay
  $ ./bin/svmtrain_lbp.py -v features/regular -d scores/regular --sn -r replay
     
The results for the CASIA-FASD database can be obtained in the same way, by specifying the ``casia`` parameter at the end of the commands. Note that the results for CASIA-FASD are reported on per-block basis, and using 5-fold cross validation. This means that the results need to be generated 5 times, training with different fold, which can be specified as an argument as well.

Important note: the results in the last column of Table V are not straight-forwardly reproducible at the moment (in particular, the concatenation of histograms is not directly supported using the scripts in this satellite package). Furthermore, at the present state, the scripts do not support the NUAA database. Work to solve this incovenience is in progress :)
  


Problems
--------

In case of problems, please contact any of the authors of the paper.


.. _Bob: http://www.idiap.ch/software/bob