#!/bin/bash

#PBS -S /bin/sh
#PBS -N data_stackoverflow
#PBS -l pmem=62g
#PBS -m abe
#PBS -l walltime=24:00:00 

DATADUMP_DIR=/fs/clip-corpora/stackexchange
DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
SITE_NAME=stackoverflow.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

source /fs/clip-amr/isi-internship/theano-env/bin/activate
rm -r $DATA_DIR/$SITE_NAME/post_docs
rm -r $DATA_DIR/$SITE_NAME/post_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/post_docs

python $SCRIPTS_DIR/data_generator.py   --posts_xml $DATADUMP_DIR/$SITE_NAME-Posts/Posts.xml \
                                        --comments_xml $DATADUMP_DIR/$SITE_NAME-Comments/Comments.xml \
                                        --posthistory_xml $DATADUMP_DIR/$SITE_NAME-PostHistory/PostHistory.xml \
                                        --users_xml $DATADUMP_DIR/$SITE_NAME-Users/Users.xml \
                                        --word_embeddings $DATA_DIR/word_embeddings.p \
                                        --vocab $DATA_DIR/vocab.p \
                                        --post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p \
                                        --post_sent_vectors $DATA_DIR/$SITE_NAME/post_sent_vectors.p \
                                        --ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p \
                                        --ans_list_vectors $DATA_DIR/$SITE_NAME/ans_list_vectors.p \
                                        --utility_post_vectors $DATA_DIR/$SITE_NAME/utility_post_vectors.p \
                                        --utility_labels $DATA_DIR/$SITE_NAME/utility_labels.p \
                                        --lucene_docs_dir $DATA_DIR/$SITE_NAME/post_docs \
                                        --lucene_similar_posts $DATA_DIR/$SITE_NAME/lucene_similar_posts.txt \
                                        --site_name $SITE_NAME \
                                        --utility_post_sent_vectors $DATA_DIR/$SITE_NAME/utility_post_sent_vectors.p \
                                        --utility_ans_list_vectors $DATA_DIR/$SITE_NAME/utility_ans_list_vectors.p \
                                        --post_ids $DATA_DIR/$SITE_NAME/post_ids.p \
                                        --utility_post_ids $DATA_DIR/$SITE_NAME/utility_post_ids.p \
                                        --post_ques_ans_log $DATA_DIR/$SITE_NAME/post_ques_ans.log
