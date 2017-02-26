#!/bin/bash

EMB_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v2
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
#SITE_NAME=askubuntu.com
#SITE_NAME=english.stackexchange.com
SITE_NAME=askubuntu_unix_superuser
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu0 python $SCRIPTS_DIR/baseline_pq_newdata.py \
                                                --post_ids_train $DATA_DIR/$SITE_NAME/post_ids_train.p \
                                                --post_vectors_train $DATA_DIR/$SITE_NAME/post_vectors_train.p \
												--ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
												--post_ids_test $DATA_DIR/$SITE_NAME/post_ids_test.p \
												--post_vectors_test $DATA_DIR/$SITE_NAME/post_vectors_test.p \
												--ques_list_vectors_test $DATA_DIR/$SITE_NAME/ques_list_vectors_test.p \
												--word_embeddings $EMB_DIR/word_embeddings.p \
                                                --batch_size 100 --no_of_epochs 20 --no_of_candidates 10 \
                                                --dev_predictions_output $DATA_DIR/$SITE_NAME/dev_predictions.out \
												--test_predictions_output $DATA_DIR/$SITE_NAME/test_predictions.out
