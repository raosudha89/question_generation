#!/bin/bash

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
SITE_NAME=askubuntu.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu0 python $SCRIPTS_DIR/post_ques_ans_ranking.py --post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p --ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p --ans_list_vectors $DATA_DIR/$SITE_NAME/ans_list_vectors.p --word_embeddings $DATA_DIR/word_embeddings.p --batch_size 500 --no_of_candidates 10 --no_of_epochs 10 --pred_ans_list_vectors $DATA_DIR/$SITE_NAME/pred_ans_list_vectors.p --pred_ans_post_vectors $DATA_DIR/$SITE_NAME/pred_ans_post_vectors.p --pred_ans_post_mask_vectors $DATA_DIR/$SITE_NAME/pred_ans_post_mask_vectors.p 
