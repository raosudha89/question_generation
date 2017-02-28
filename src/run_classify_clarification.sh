#!/bin/bash

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
CROWDFLOWER_DATA_DIR=/fs/clip-amr/question_generation/datasets/crowdflower
SITE_NAME=askubuntu_unix_superuser
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/gpu_virtualenv/bin/activate

THEANO_FLAGS=floatX=float32,device=gpu1 python $SCRIPTS_DIR/classify_clarification.py \
                                                --post_ids_train $DATA_DIR/$SITE_NAME/post_ids_train.p \
                                                --ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
                                                --clarification_labels_file $CROWDFLOWER_DATA_DIR/labelled_data.csv \
												--word_embeddings $DATA_DIR/word_embeddings.p \
                                                --batch_size 50 --no_of_epochs 20