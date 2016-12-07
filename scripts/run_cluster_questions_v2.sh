#!/bin/bash

#PBS -S /bin/sh
#PBS -N site_kmeans
#PBS -l pmem=16g
#PBS -m abe
#PBS -l walltime=48:00:00 

DATADIR=/fs/clip-corpora/stackexchange
SCRIPTDIR=/fs/clip-amr/question_generation/scripts
WORDVECTORS=/fs/clip-amr/question_generation/datasets/stackexchange/vectors_200.txt
CLUSTERSDIR=/fs/clip-amr/question_generation/datasets/stackexchange/clusters_kmeans
CLUSTERALGO=kmeans

source ~/pythonENV/bin/activate
for site in $DATADIR/*.7z;
do
	sitename=`basename ${site%.7z}`
	echo $sitename
	python $SCRIPTDIR/cluster_questions_v2.py $DATADIR/$sitename/Posts.xml $DATADIR/$sitename/Comments.xml $WORDVECTORS $CLUSTERALGO > $CLUSTERSDIR/$sitename
done
deactivate
