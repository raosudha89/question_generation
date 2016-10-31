#!/bin/bash

#PBS -S /bin/sh
#PBS -N classifier_3dprinting
#PBS -l pmem=32g
#PBS -m abe
#PBS -l walltime=12:00:00 

source /fs/clip-amr/isi-internship/theano-env/bin/activate
cd /fs/clip-amr/question_generation
mkdir -p datasets/stackexchange/3dprinting.stackexchange.com
python scripts/completeness_classifier_posthistory.py /fs/clip-corpora/stackexchange/3dprinting.stackexchange.com/PostHistory.xml datasets/stackexchange/vectors_200.txt > datasets/stackexchange/3dprinting.stackexchange.com/completeness_classifier_posthistory.out
