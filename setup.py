#!/usr/bin/env python
# Ivana Chingovska <ivana.chingovska@idiap.ch>
# Sun Jul  8 20:35:55 CEST 2012
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
from setuptools import setup, find_packages

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='antispoofing.lbp',
    version='1.2.1',
    description='Texture (LBP) based counter-measures for the REPLAY-ATTACK database',
    url='http://pypi.python.org/pypi/antispoofing.lbp',
    license='GPLv3',
    author='Ivana Chingovska',
    author_email='ivana.chingovska@idiap.ch',
    long_description=open('README.rst').read(),
    keywords='antispoofing texture, LBP, face spoofing attacks, face spoofing database, bob, xbob',

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,

    namespace_packages=[
      "antispoofing",
      ],

    install_requires=[
      "setuptools",
      "bob >= 1.2", #1.1.0
      "xbob.db.replay >= 1.0.4", # Replay-Attack database
      "xbob.db.casia_fasd >= 1.1.0", #CASIA database
      "antispoofing.utils >= 1.1.3",  #Utils Package
    ],

    entry_points={
      'console_scripts': [
        'calclbp.py = antispoofing.lbp.script.calclbp:main',
        'calcframelbp.py = antispoofing.lbp.script.calcframelbp:main',
        'mkhistmodel.py = antispoofing.lbp.script.mkhistmodel:main',
        'cmphistmodels.py = antispoofing.lbp.script.cmphistmodels:main',
        'ldatrain_lbp.py = antispoofing.lbp.script.ldatrain_lbp:main',
        'svmtrain_lbp.py = antispoofing.lbp.script.svmtrain_lbp:main',
        ],
      },

)
