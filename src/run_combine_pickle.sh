#!/bin/bash

#PBS -S /bin/sh
#PBS -N combining
#PBS -l pmem=64g
#PBS -m abe
#PBS -l walltime=12:00:00 

DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
UBUNTU=askubuntu.com
UNIX=unix.stackexchange.com
SUPERUSER=superuser.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

cp $DATA_DIR/$UBUNTU/post_sent_vectors_test.p $DATA_DIR/askubuntu_unix_superuser/post_sent_vectors_test.p

cp $DATA_DIR/$UBUNTU/ques_list_vectors_test.p $DATA_DIR/askubuntu_unix_superuser/ques_list_vectors_test.p

cp $DATA_DIR/$UBUNTU/ans_list_vectors_test.p $DATA_DIR/askubuntu_unix_superuser/ans_list_vectors_test.p

cp $DATA_DIR/$UBUNTU/post_ids_test.p $DATA_DIR/askubuntu_unix_superuser/post_ids_test.p

cp $DATA_DIR/$UBUNTU/utility_post_sent_vectors_test.p $DATA_DIR/askubuntu_unix_superuser/utility_post_sent_vectors_test.p

cp $DATA_DIR/$UBUNTU/utility_labels_test.p $DATA_DIR/askubuntu_unix_superuser/utility_labels_test.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_sent_vectors_train.p \
										$DATA_DIR/$UNIX/post_sent_vectors.p \
										$DATA_DIR/$SUPERUSER/post_sent_vectors.p \
										$DATA_DIR/askubuntu_unix_superuser/post_sent_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ques_list_vectors_train.p \
										$DATA_DIR/$UNIX/ques_list_vectors.p \
										$DATA_DIR/$SUPERUSER/ques_list_vectors.p \
										$DATA_DIR/askubuntu_unix_superuser/ques_list_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ans_list_vectors_train.p \
										$DATA_DIR/$UNIX/ans_list_vectors.p \
										$DATA_DIR/$SUPERUSER/ans_list_vectors.p \
										$DATA_DIR/askubuntu_unix_superuser/ans_list_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_ids_train.p \
										$DATA_DIR/$UNIX/post_ids.p \
										$DATA_DIR/$SUPERUSER/post_ids.p \
										$DATA_DIR/askubuntu_unix_superuser/post_ids_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/utility_post_sent_vectors_train.p \
										$DATA_DIR/$UNIX/utility_post_sent_vectors.p \
										$DATA_DIR/$SUPERUSER/utility_post_sent_vectors.p \
										$DATA_DIR/askubuntu_unix_superuser/utility_post_sent_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/utility_labels_train.p \
										$DATA_DIR/$UNIX/utility_labels.p \
										$DATA_DIR/$SUPERUSER/utility_labels.p \
										$DATA_DIR/askubuntu_unix_superuser/utility_labels_train.p

