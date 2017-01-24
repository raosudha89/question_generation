#!/bin/bash

#PBS -S /bin/sh
#PBS -N utility_askubuntu
#PBS -l pmem=32g
#PBS -m abe
#PBS -l walltime=48:00:00 

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
SITE_NAME=askubuntu.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/isi-internship/theano-env/bin/activate

THEANO_FLAGS=floatX=float32 python $SCRIPTS_DIR/calculate_utility.py --utility_post_vectors $DATA_DIR/$SITE_NAME/utility_post_vectors.p --utility_labels $DATA_DIR/$SITE_NAME/utility_labels.p --word_embeddings $DATA_DIR/word_embeddings.p --batch_size 500 --no_of_epochs 5
