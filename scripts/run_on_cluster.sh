#!/bin/bash

#PBS -S /bin/sh
#PBS -N extract_plain_text_stackoverflow
#PBS -l pmem=64g
#PBS -m abe
#PBS -l walltime=12:00:00 

source /fs/clip-amr/isi-internship/theano-env/bin/activate
cd /fs/clip-amr/question_generation
#mkdir -p datasets/stackexchange/3dprinting.stackexchange.com
#python scripts/completeness_classifier_posthistory.py /fs/clip-corpora/stackexchange/3dprinting.stackexchange.com/PostHistory.xml datasets/stackexchange/vectors_200.txt > datasets/stackexchange/3dprinting.stackexchange.com/completeness_classifier_posthistory.out

#python scripts/post_qa_encoder.py datasets/stackexchange/academia.stackexchange.com/data_posts.p datasets/stackexchange/academia.stackexchange.com/data_questions.p datasets/stackexchange/academia.stackexchange.com/data_answers.p datasets/stackexchange/vectors_200.txt > datasets/stackexchange/academia.stackexchange.com/post_qa_encoder.out

python scripts/extract_plain_text.py /fs/clip-corpora/stackexchange/stackoverflow.com-Posts/Posts.xml /fs/clip-corpora/stackexchange/stackoverflow.com-Comments/Comments.xml datasets/stackexchange/stackoverflow.com/plain_text.txt
