#!/bin/bash

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
SITE_NAME=askubuntu.com
#SITE_NAME=english.stackexchange.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu0 python $SCRIPTS_DIR/baseline_pqa.py \
             --post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p \
             --post_sent_vectors $DATA_DIR/$SITE_NAME/post_sent_vectors.p \
             --ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p \
             --ans_list_vectors $DATA_DIR/$SITE_NAME/ans_list_vectors.p \
             --word_embeddings $DATA_DIR/word_embeddings.p \
             --batch_size 100 --no_of_epochs 20 --no_of_candidates 10
