#!/bin/bash

#SBATCH --job-name=data_superuser_v8
#SBATCH --output=data_superuser_v8
#SBATCH --mem=48g
#SBATCH --time=12:00:00

DATADUMP_DIR=/fs/clip-corpora/stackexchange
EMB_DIR=/fs/clip-amr/question_generation/datasets/stackexchange
DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v8
#SITE_NAME=3dprinting.stackexchange.com
#SITE_NAME=academia.stackexchange.com
#SITE_NAME=askubuntu.com
#SITE_NAME=codereview.stackexchange.com
#SITE_NAME=english.stackexchange.com
#SITE_NAME=math.stackexchange.com
#SITE_NAME=physics.stackexchange.com
SITE_NAME=superuser.com
#SITE_NAME=tex.stackexchange.com
#SITE_NAME=unix.stackexchange.com
SCRIPTS_DIR=/fs/clip-amr/question_generation/src

#source /fs/clip-amr/isi-internship/theano-env/bin/activate
source /fs/clip-amr/gpu_virtualenv/bin/activate

mkdir -p $DATA_DIR/$SITE_NAME

rm -r $DATA_DIR/$SITE_NAME/post_docs
rm -r $DATA_DIR/$SITE_NAME/post_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/post_docs

rm -r $DATA_DIR/$SITE_NAME/ques_docs
rm -r $DATA_DIR/$SITE_NAME/ques_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/ques_docs

python $SCRIPTS_DIR/data_generator.py   --posts_xml $DATADUMP_DIR/$SITE_NAME/Posts.xml \
                                        --comments_xml $DATADUMP_DIR/$SITE_NAME/Comments.xml \
                                        --posthistory_xml $DATADUMP_DIR/$SITE_NAME/PostHistory.xml \
                                        --users_xml $DATADUMP_DIR/$SITE_NAME/Users.xml \
                                        --word_embeddings $EMB_DIR/word_embeddings.p \
                                        --vocab $EMB_DIR/vocab.p \
                                        --post_vectors $DATA_DIR/$SITE_NAME/post_vectors.p \
                                        --post_sent_vectors $DATA_DIR/$SITE_NAME/post_sent_vectors.p \
                                        --ques_list_vectors $DATA_DIR/$SITE_NAME/ques_list_vectors.p \
                                        --ans_list_vectors $DATA_DIR/$SITE_NAME/ans_list_vectors.p \
                                        --utility_post_vectors $DATA_DIR/$SITE_NAME/utility_post_vectors.p \
                                        --utility_labels $DATA_DIR/$SITE_NAME/utility_labels.p \
                                        --lucene_docs_dir $DATA_DIR/$SITE_NAME/post_docs \
                                        --lucene_ques_docs_dir $DATA_DIR/$SITE_NAME/ques_docs \
                                        --lucene_similar_posts $DATA_DIR/$SITE_NAME/lucene_similar_posts.txt \
                                        --lucene_similar_questions $DATA_DIR/$SITE_NAME/lucene_similar_questions.txt \
                                        --site_name $SITE_NAME \
                                        --utility_post_sent_vectors $DATA_DIR/$SITE_NAME/utility_post_sent_vectors.p \
                                        --utility_ans_list_vectors $DATA_DIR/$SITE_NAME/utility_ans_list_vectors.p \
                                        --post_ids $DATA_DIR/$SITE_NAME/post_ids.p \
                                        --utility_post_ids $DATA_DIR/$SITE_NAME/utility_post_ids.p \
                                        --post_ques_ans_log $DATA_DIR/$SITE_NAME/post_ques_ans.log
