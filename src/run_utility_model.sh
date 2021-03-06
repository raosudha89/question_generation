#!/bin/bash

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
#SITE_NAME=askubuntu.com
#SITE_NAME=english.stackexchange.com
#SITE_NAME=superuser.com
SITE_NAME=askubuntu_unix_superuser
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu1 python $SCRIPTS_DIR/utility_model.py \
												--post_ids_train $DATA_DIR/$SITE_NAME/post_ids_train.p \
												--post_vectors_train $DATA_DIR/$SITE_NAME/post_vectors_train.p \
												--ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
												--ans_list_vectors_train $DATA_DIR/$SITE_NAME/ans_list_vectors_train.p \
												--utility_post_vectors_train $DATA_DIR/$SITE_NAME/utility_post_vectors_train.p \
												--utility_labels_train $DATA_DIR/$SITE_NAME/utility_labels_train.p \
												--post_ids_test $DATA_DIR/$SITE_NAME/post_ids_test.p \
												--post_vectors_test $DATA_DIR/$SITE_NAME/post_vectors_test.p \
												--ques_list_vectors_test $DATA_DIR/$SITE_NAME/ques_list_vectors_test.p \
												--ans_list_vectors_test $DATA_DIR/$SITE_NAME/ans_list_vectors_test.p \
												--utility_post_vectors_test $DATA_DIR/$SITE_NAME/utility_post_vectors_test.p \
												--utility_labels_test $DATA_DIR/$SITE_NAME/utility_labels_test.p \
												--word_embeddings $DATA_DIR/word_embeddings.p \
												--hidden_dim 100 \
												--batch_size 100 \
												--no_of_epochs 100 \
												--no_of_candidates 10\
												--_lambda 0.5 \
												--dev_predictions_output $DATA_DIR/$SITE_NAME/dev_predictions_orgsetting.out \
												--test_predictions_output $DATA_DIR/$SITE_NAME/test_predictions_orgsetting.out
