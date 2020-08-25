#!/bin/bash

# Git
DIR=/opt/git
mkdir $DIR

cd $DIR
git clone https://github.com/nipy/nipype.git --branch 1.2.3
cd $DIR/nipype
pip3 install .
python3 -c "import nipype; print(nipype.__version__)"

cd $DIR
git clone https://github.com/ostanley/phaseprep.git
cd $DIR/phaseprep

git checkout dev

# Install requirements starting with fmriprep dependencies
pip3 install -r requirements.txt

# add phaseprep to path
python3 setup.py install
