#!/bin/bash

#PBS -S /bin/sh
#PBS -N baseline_ranking_askubuntu
#PBS -l pmem=16g
#PBS -q batch
#PBS -m abe
#PBS -l walltime=24:00:00 

source /fs/clip-amr/isi-internship/theano-env/bin/activate
cd /fs/clip-amr/question_generation
#mkdir -p datasets/stackexchange/3dprinting.stackexchange.com
#python scripts/completeness_classifier_posthistory.py /fs/clip-corpora/stackexchange/3dprinting.stackexchange.com/PostHistory.xml datasets/stackexchange/vectors_200.txt > datasets/stackexchange/3dprinting.stackexchange.com/completeness_classifier_posthistory.out

#python scripts/post_qa_encoder.py datasets/stackexchange/academia.stackexchange.com/data_posts.p datasets/stackexchange/academia.stackexchange.com/data_questions.p datasets/stackexchange/academia.stackexchange.com/data_answers.p datasets/stackexchange/vectors_200.txt > datasets/stackexchange/academia.stackexchange.com/post_qa_encoder.out

#python scripts/extract_plain_text.py /fs/clip-corpora/stackexchange/stackoverflow.com-Posts/Posts.xml /fs/clip-corpora/stackexchange/stackoverflow.com-Comments/Comments.xml datasets/stackexchange/stackoverflow.com/plain_text.txt

#sh scripts/run_generate_post_qa_data.sh > generate_math_post_qa_data.out

#sh scripts/run_generate_posts_questions.sh > run_generate_posts_questions.out

#sh scripts/run_generate_data_for_lstm.sh > run_generate_data_for_lstm.out

#python scripts/baseline_post_ques.py datasets/stackexchange/all_posts_N2.p datasets/stackexchange/all_questions_N2.p datasets/stackexchange/all_labels_N2.p datasets/stackexchange/word_embeddings.p > baseline_average_N2.out

#python scripts/baseline_post_ques.py datasets/stackexchange/all_posts_N2.p datasets/stackexchange/all_questions_N2.p datasets/stackexchange/all_labels_N2.p datasets/stackexchange/word_embeddings.p

#python scripts/post_qa_encoder.py datasets/stackexchange/all_data_posts.p datasets/stackexchange/all_data_questions.p datasets/stackexchange/all_data_answers.p datasets/stackexchange/vectors_200.txt > datasets/stackexchange/post_qa_encoder.out

#THEANO_FLAGS=floatX=float32 python scripts/baseline_post_ques_ranking.py datasets/stackexchange/all_posts_N2.p datasets/stackexchange/all_questions_N2.p 2 datasets/stackexchange/word_embeddings.p 200

#THEANO_FLAGS=floatX=float32 python scripts/baseline_post_ques_ranking.py datasets/stackexchange/academia.stackexchange.com/main_labelled_posts.p datasets/stackexchange/academia.stackexchange.com/main_labelled_questions.p 2 datasets/stackexchange/word_embeddings.p 50 > datasets/stackexchange/academia.stackexchange.com/baseline_post_ques_ranking.out

THEANO_FLAGS=floatX=float32 python scripts/baseline_post_ques_ranking.py datasets/stackexchange/askubuntu.com/main_labelled_posts.p datasets/stackexchange/askubuntu.com/main_labelled_questions.p 2 datasets/stackexchange/word_embeddings.p 50 > datasets/stackexchange/askubuntu.com/baseline_post_ques_ranking.out

#THEANO_FLAGS=floatX=float32 python scripts/baseline_post_ques_ranking.py datasets/stackexchange/all_main_labelled_posts.p datasets/stackexchange/all_main_labelled_questions.p 2 datasets/stackexchange/word_embeddings.p 200 > datasets/stackexchange/baseline_post_ques_ranking.out

