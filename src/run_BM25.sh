#!/bin/bash

#PBS -S /bin/sh
#PBS -N BM25_academia
#PBS -l pmem=64g
#PBS -q batch
#PBS -m abe
#PBS -l walltime=24:00:00 

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
SITE_NAME=academia.stackexchange.com
#SITE_NAME=askubuntu.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/isi-internship/theano-env/bin/activate

python $SCRIPTS_DIR/BM25.py --posts_xml $DATA_DIR/$SITE_NAME/Posts.xml --posthistory_xml $DATA_DIR/$SITE_NAME/PostHistory.xml --similar_posts $DATA_DIR/$SITE_NAME/similar_posts.p 
