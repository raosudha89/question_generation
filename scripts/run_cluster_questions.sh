#!/bin/bash

#PBS -S /bin/sh
#PBS -N site_kmeans
#PBS -l pmem=16g
#PBS -m abe
#PBS -l walltime=48:00:00 

DATADIR=/fs/clip-amr/question_generation/datasets/stackexchange
SCRIPTDIR=/fs/clip-amr/question_generation/scripts
WORDVECTORS=/fs/clip-amr/question_generation/datasets/stackexchange/vectors_200.txt
CLUSTERALGO=kmeans

source ~/pythonENV/bin/activate
for site in $DATADIR/*.com;
do
	sitename=`basename ${site}`
	echo $sitename
	python $SCRIPTDIR/cluster_questions.py $DATADIR/$sitename/main_post_questions.p $WORDVECTORS $CLUSTERALGO > $DATADIR/$sitename/clusters_questions.${CLUSTERALGO}.out
done
deactivate
