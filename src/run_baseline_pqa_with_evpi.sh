#!/bin/bash

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
#SITE_NAME=askubuntu.com
SITE_NAME=english.stackexchange.com
#SITE_NAME=askubuntu_unix_superuser
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu1 python $SCRIPTS_DIR/baseline_pqa_with_evpi.py \
                                                --post_ids_train $DATA_DIR/$SITE_NAME/post_ids_train.p \
                                                --post_vectors_train $DATA_DIR/$SITE_NAME/post_vectors_train.p \
												--ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
												--ans_list_vectors_train $DATA_DIR/$SITE_NAME/ans_list_vectors_train.p \
                                                --post_ids_test $DATA_DIR/$SITE_NAME/post_ids_test.p \
												--post_vectors_test $DATA_DIR/$SITE_NAME/post_vectors_test.p \
												--ques_list_vectors_test $DATA_DIR/$SITE_NAME/ques_list_vectors_test.p \
												--ans_list_vectors_test $DATA_DIR/$SITE_NAME/ans_list_vectors_test.p \
                                                --word_embeddings $DATA_DIR/word_embeddings.p \
                                                --batch_size 100 --no_of_epochs 20 --no_of_candidates 10 \
                                                --dev_predictions_output $DATA_DIR/$SITE_NAME/dev_predictions.out \
												--test_predictions_output $DATA_DIR/$SITE_NAME/test_predictions.out
