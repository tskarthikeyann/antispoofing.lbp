=============================================================================
 Counter-Measures to Face Spoofing Attacks using Local Binary Patterns (LBP)
=============================================================================

This package implements the LBP counter-measure to spoofing attacks to face
recognition systems as described at the paper `On the Effectiveness of Local
Binary Patterns in Face Anti-spoofing`, by Chingovska, Anjos and Marcel,
presented at the IEEE BioSIG 2012 meeting.

If you use this package and/or its results, please cite the following
publications:

1. The original paper with the counter-measure explained in details::

    @INPROCEEDINGS{Chingovska_BIOSIG_2012,
    author = {Chingovska, Ivana and Anjos, Andr{\'{e}} and Marcel, S{\'{e}}bastien},
    keywords = {Attack, Counter-Measures, Counter-Spoofing, Face Recognition, Liveness Detection, Replay, Spoofing},
    month = sep,
    title = {On the Effectiveness of Local Binary Patterns in Face Anti-spoofing},
    journal = {IEEE BIOSIG 2012},
    year = {2012},
    }
 
2. Bob as the core framework used to run the experiments::

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

Installation
------------

.. note:: 

  If you are reading this page through our GitHub portal and not through PyPI,
  note **the development tip of the package may not be stable** or become
  unstable in a matter of moments.

  Go to `http://pypi.python.org/pypi/antispoofing.lbp
  <http://pypi.python.org/pypi/antispoofing.lbp>`_ to download the latest
  stable version of this package.

There are 2 options you can follow to get this package installed and
operational on your computer: you can use automatic installers like `pip
<http://pypi.python.org/pypi/pip/>`_ (or `easy_install
<http://pypi.python.org/pypi/setuptools>`_) or manually download, unpack and
use `zc.buildout <http://pypi.python.org/pypi/zc.buildout>`_ to create a
virtual work environment just for this package.

Using an automatic installer
============================

Using ``pip`` is the easiest (shell commands are marked with a ``$`` signal)::

  $ pip install antispoofing.lbp

You can also do the same with ``easy_install``::

  $ easy_install antispoofing.lbp

This will download and install this package plus any other required
dependencies. It will also verify if the version of Bob you have installed
is compatible.

This scheme works well with virtual environments by `virtualenv
<http://pypi.python.org/pypi/virtualenv>`_ or if you have root access to your
machine. Otherwise, we recommend you use the next option.

Using ``zc.buildout``
=====================

Download the latest version of this package from `PyPI
<http://pypi.python.org/pypi/antispoofing.lbp>`_ and unpack it in your
working area. The installation of the toolkit itself uses `buildout
<http://www.buildout.org/>`_. You don't need to understand its inner workings
to use this package. Here is a recipe to get you started::
  
  $ python bootstrap.py 
  $ ./bin/buildout

These 2 commands should download and install all non-installed dependencies and
get you a fully operational test and development environment.

.. note::

  The python shell used in the first line of the previous command set
  determines the python interpreter that will be used for all scripts developed
  inside this package. Because this package makes use of `Bob
  <http://idiap.github.com/bob>`_, you must make sure that the ``bootstrap.py``
  script is called with the **same** interpreter used to build Bob, or
  unexpected problems might occur.

  If Bob is installed by the administrator of your system, it is safe to
  consider it uses the default python interpreter. In this case, the above 3
  command lines should work as expected. If you have Bob installed somewhere
  else on a private directory, edit the file ``buildout.cfg`` **before**
  running ``./bin/buildout``. Find the section named ``external`` and edit the
  line ``egg-directories`` to point to the ``lib`` directory of the Bob
  installation you want to use. For example::

    [external]
    egg-directories=/Users/crazyfox/work/bob/build/lib

User Guide
----------

This section explains how to use the package in order to: a) calculate the LBP
features on the REPLAY-ATTACK database; b) perform classification using Chi-2,
Linear Discriminant Analysis (LDA) and Support Vector Machines (SVM).

It is assumed you have followed the installation instructions for the package,
and got the REPLAY-ATTACK database downloaded and uncompressed in a directory.
After running the ``buildout`` command, you should have all required utilities
sitting inside the ``bin`` directory. We expect that the video files downloaded
for the PRINT-ATTACK database are installed in a sub-directory called
``database`` at the root of the package.  You can use a link to the location of
the database files, if you don't want to have the database installed on the
root of this package::

  $ ln -s /path/where/you/installed/the/replay-attack-database database

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

  $ ./bin/calclbp.py --ff 50

In the above command, the face size filter is set to 50 pixels (as in the
paper), and the program will discard all the frames with detected faces smaller
then 50 pixels as invalid.

To see all the options for the scripts ``calclbp.py`` and ``calcframelbp.py``,
just type ``--help`` at the command line. Change the default option in order to
obtain various features, as described in the paper. 

Classification using Chi-2 distance
===================================

The clasification using Chi-2 distance consists of two steps. The first one is
creating the histogram model (average LBP histogram of all the real access
videos in the training set). The second step is comparison of the features of
development and test videos to the model histogram and writing the results.

The script to use for creating the histogram model is
``./bin/mkhistmodel.py``.  It expects that the LBP features of the videos are
stored in a folder ``./bin/lbp_features``. The model histogram will be written
in the default output folder ``./res``. You can change this default features by
setting the input arguments. To execute this script, just run::

  $ ./bin/mkhistmodel.py

The script for performing Chi-2 histogram comparison is
``./bin/cmphistmodels.py``, and it assumes that the model histogram has been
already created. It makes use of the utility script ``spoof/chi2.py`` and
``ml/perf.py`` for writing the results in a file. The default input directory is
``./lbp_features``, while the default input directoru for the histogram model
as well as default output directory is ``./res``. To execute this script, just
run:: 

  $ ./bin/cmphistmodel.py

To see all the options for the scripts ``mkhistmodel.py`` and
``cmphistmodels.py``, just type ``--help`` at the command line.

Classification with linear discriminant analysis (LDA)
======================================================

The classification with LDA is performed using the script
``./bin/ldatrain_lbp.py``. It makes use of the scripts ``ml/lda.py``,
``ml/pca.py`` (if PCA reduction is performed on the data) and ``ml/norm.py``
(if the data need to be normalized). The default input and output directories
are ``./lbp_features`` and ``./res``. To execute the script with prior PCA
dimensionality reduction as is done in the paper, call::

  $ ./bin/ldatrain_lbp.py -r 

To see all the options for this script, just type ``--help`` at the command
line.

Classification with support vector machine (SVM)
================================================

The classification with SVM is performed using the script
``./bin/svmtrain_lbp.py``. It makes use of the scripts ``ml/pca.py`` (if PCA
reduction is performed on the data) and ``ml\norm.py`` (if the data need to be
normalized). The default input and output directories are ``./lbp_features``
and ``./res``. To execute the script with prior normalization of the data in
the range ``[-1, 1]`` as in the paper, the default parameters, call::

  $ ./bin/svmtrain_lbp.py -n

To see all the options for this script, just type ``--help`` at the command
line.

Problems
--------

In case of problems, please contact any of the authors of the paper.
