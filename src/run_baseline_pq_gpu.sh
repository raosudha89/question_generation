#!/bin/bash

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
#SITE_NAME=askubuntu.com
#SITE_NAME=english.stackexchange.com
SITE_NAME=askubuntu_unix_superuser
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

#THEANO_FLAGS=floatX=float32,device=gpu0 python $SCRIPTS_DIR/baseline_pq.py \
#              --post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p \
#              --ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p \
#              --word_embeddings $DATA_DIR/word_embeddings.p \
#              --batch_size 100 --no_of_epochs 20 --no_of_candidates 10

THEANO_FLAGS=floatX=float32,device=gpu0 python $SCRIPTS_DIR/baseline_pq.py \
             --post_vectors $DATA_DIR/$SITE_NAME/post_vectors_train.p \
             --ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
             --word_embeddings $DATA_DIR/word_embeddings.p \
             --batch_size 100 --no_of_epochs 20 --no_of_candidates 10