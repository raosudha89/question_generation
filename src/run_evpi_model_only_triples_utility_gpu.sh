#!/bin/bash

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
SITE_NAME=askubuntu.com
#SITE_NAME=english.stackexchange.com
#SITE_NAME=superuser.com
#SITE_NAME=askubuntu_unix_superuser
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu0 python $SCRIPTS_DIR/evpi_model_only_triples_utility.py \
												--post_sent_vectors_train $DATA_DIR/$SITE_NAME/post_sent_vectors_train.p \
												--ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
												--ans_list_vectors_train $DATA_DIR/$SITE_NAME/ans_list_vectors_train.p \
												--utility_post_sent_vectors_train $DATA_DIR/$SITE_NAME/utility_post_sent_vectors_train.p \
												--utility_labels_train $DATA_DIR/$SITE_NAME/utility_labels_train.p \
												--post_sent_vectors_test $DATA_DIR/$SITE_NAME/post_sent_vectors_test.p \
												--ques_list_vectors_test $DATA_DIR/$SITE_NAME/ques_list_vectors_test.p \
												--ans_list_vectors_test $DATA_DIR/$SITE_NAME/ans_list_vectors_test.p \
												--utility_post_sent_vectors_test $DATA_DIR/$SITE_NAME/utility_post_sent_vectors_test.p \
												--utility_labels_test $DATA_DIR/$SITE_NAME/utility_labels_test.p \
												--word_embeddings $DATA_DIR/word_embeddings.p \
												--hidden_dim 100 \
												--batch_size 100 \
												--no_of_epochs 20 \
												--no_of_candidates 10\
												--_lambda 0.5 
