#!/bin/bash

#PBS -S /bin/sh
#PBS -N split_askubuntu_v4
#PBS -l pmem=32g
#PBS -m abe
#PBS -q batch
#PBS -l walltime=12:00:00 

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v4
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
SITE_NAME=askubuntu.com
#SITE_NAME=english.stackexchange.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

python $SCRIPTS_DIR/split_train_test_onlytriples.py --human_evald_post_ids $DATA_DIR/$SITE_NAME/human_evald_post_ids.txt \
										--post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p \
										--post_sent_vectors $DATA_DIR/$SITE_NAME/post_sent_vectors.p \
										--ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p \
										--ans_list_vectors $DATA_DIR/$SITE_NAME/ans_list_vectors.p \
										--post_ids $DATA_DIR/$SITE_NAME/post_ids.p \
										--post_vectors_test $DATA_DIR/$SITE_NAME/post_vectors_test.p \
										--post_sent_vectors_test $DATA_DIR/$SITE_NAME/post_sent_vectors_test.p \
										--ques_list_vectors_test $DATA_DIR/$SITE_NAME/ques_list_vectors_test.p \
										--ans_list_vectors_test $DATA_DIR/$SITE_NAME/ans_list_vectors_test.p \
										--post_ids_test $DATA_DIR/$SITE_NAME/post_ids_test.p \
										--post_vectors_train $DATA_DIR/$SITE_NAME/post_vectors_train.p \
										--post_sent_vectors_train $DATA_DIR/$SITE_NAME/post_sent_vectors_train.p \
										--ques_list_vectors_train $DATA_DIR/$SITE_NAME/ques_list_vectors_train.p \
										--ans_list_vectors_train $DATA_DIR/$SITE_NAME/ans_list_vectors_train.p \
										--post_ids_train $DATA_DIR/$SITE_NAME/post_ids_train.p \
										
