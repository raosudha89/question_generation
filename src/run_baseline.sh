#!/bin/bash

#PBS -S /bin/sh
#PBS -N baseline_academia
#PBS -l pmem=16g
#PBS -q batch
#PBS -m abe
#PBS -l walltime=24:00:00 

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
SITE_NAME=academia.stackexchange.com
#SITE_NAME=askubuntu.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/isi-internship/theano-env/bin/activate

THEANO_FLAGS=floatX=float32 python $SCRIPTS_DIR/baseline_post_ques_ranking.py --post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p --ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p --word_embeddings $DATA_DIR/word_embeddings.p --batch_size 100 --no_of_candidates 2 --no_of_epochs 10 