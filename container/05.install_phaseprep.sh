#!/bin/bash

# Git
DIR=/opt/git
mkdir $DIR

cd $DIR
git clone https://github.com/ostanley/nipype.git
cd $DIR/nipype
pip3 install .
python3 -c "import nipype; print(nipype.__version__)"

cd $DIR
git clone https://git.cfmm.robarts.ca/nipype/phaseprep.git
cd $DIR/phaseprep

echo argv[1]
if [argv[1] = 'dev']
do
  git checkout dev
done

# Install requirements starting with fmriprep dependencies
pip3 install -r requirements.txt

# add phaseprep to path
python3 setup.py install
