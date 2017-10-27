#!/bin/bash

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v8
EMB_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
SITE_NAME=askubuntu_unix_superuser
#SITE_NAME=askubuntu.com
#SITE_NAME=unix.stackexchange.com
#SITE_NAME=superuser.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu1 python $SCRIPTS_DIR/final_evpi_models.py \
                                                --post_ids_train $DATA_DIR/$SITE_NAME/post_ids.p \
                                                --post_vectors_train $DATA_DIR/$SITE_NAME/post_vectors.p \
												--ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors.p \
												--ans_list_vectors_train $DATA_DIR/$SITE_NAME/ans_list_vectors.p \
                                                --word_embeddings $EMB_DIR/word_embeddings.p \
                                                --batch_size 256 --no_of_epochs 40 --no_of_candidates 10 \
												--test_predictions_output $DATA_DIR/$SITE_NAME/test_predictions_baseline_pqa_20.out \
												--model baseline_pqa
