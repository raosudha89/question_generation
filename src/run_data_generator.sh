#!/bin/bash

#PBS -S /bin/sh
#PBS -N utility_askubuntu
#PBS -l pmem=16g
#PBS -m abe
#PBS -l walltime=24:00:00 

DATADUMP_DIR=/fs/clip-corpora/stackexchange
DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
#SITE_NAME=android.stackexchange.com
SITE_NAME=askubuntu.com
#SITE_NAME=math.stackexchange.com
#SITE_NAME=superuser.com
#SITE_NAME=tex.stackexchange.com
#SITE_NAME=english.stackexchange.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/isi-internship/theano-env/bin/activate
mkdir -p $DATA_DIR/$SITE_NAME/post_docs

python $SCRIPTS_DIR/data_generator.py --posts_xml $DATADUMP_DIR/$SITE_NAME/Posts.xml --comments_xml $DATADUMP_DIR/$SITE_NAME/Comments.xml --posthistory_xml $DATADUMP_DIR/$SITE_NAME/PostHistory.xml --word_embeddings $DATA_DIR/word_embeddings.p --vocab $DATA_DIR/vocab.p --post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p --ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p --ans_list_vectors $DATA_DIR/$SITE_NAME/ans_list_vectors.p --utility_post_vectors $DATA_DIR/$SITE_NAME/utility_post_vectors.p --utility_labels $DATA_DIR/$SITE_NAME/utility_labels.p --lucene_docs_dir $DATA_DIR/$SITE_NAME/post_docs --lucene_similar_posts $DATA_DIR/$SITE_NAME/lucene_similar_posts.txt --site_name $SITE_NAME
