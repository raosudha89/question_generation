#!/bin/bash

#PBS -S /bin/sh
#PBS -N combining
#PBS -l pmem=64g
#PBS -m abe
#PBS -l walltime=12:00:00 

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v4
UBUNTU=askubuntu.com
UNIX=unix.stackexchange.com
SUPERUSER=superuser.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src
SITE_NAME=askubuntu_unix_superuser

mkdir -p $DATA_DIR/$SITE_NAME

cp $DATA_DIR/$UBUNTU/post_sent_vectors_test.p $DATA_DIR/$SITE_NAME/post_sent_vectors_test.p

cp $DATA_DIR/$UBUNTU/post_vectors_test.p $DATA_DIR/$SITE_NAME/post_vectors_test.p

cp $DATA_DIR/$UBUNTU/ques_list_vectors_test.p $DATA_DIR/$SITE_NAME/ques_list_vectors_test.p

cp $DATA_DIR/$UBUNTU/ans_list_vectors_test.p $DATA_DIR/$SITE_NAME/ans_list_vectors_test.p

cp $DATA_DIR/$UBUNTU/post_ids_test.p $DATA_DIR/$SITE_NAME/post_ids_test.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_vectors_train.p \
										$DATA_DIR/$UNIX/post_vectors.p \
										$DATA_DIR/$SUPERUSER/post_vectors.p \
										$DATA_DIR/$SITE_NAME/post_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_sent_vectors_train.p \
										$DATA_DIR/$UNIX/post_sent_vectors.p \
										$DATA_DIR/$SUPERUSER/post_sent_vectors.p \
										$DATA_DIR/$SITE_NAME/post_sent_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ques_list_vectors_train.p \
										$DATA_DIR/$UNIX/ques_list_vectors.p \
										$DATA_DIR/$SUPERUSER/ques_list_vectors.p \
										$DATA_DIR/$SITE_NAME/ques_list_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ans_list_vectors_train.p \
										$DATA_DIR/$UNIX/ans_list_vectors.p \
										$DATA_DIR/$SUPERUSER/ans_list_vectors.p \
										$DATA_DIR/$SITE_NAME/ans_list_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_ids_train.p \
										$DATA_DIR/$UNIX/post_ids.p \
										$DATA_DIR/$SUPERUSER/post_ids.p \
										$DATA_DIR/$SITE_NAME/post_ids_train.p

