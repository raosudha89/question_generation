#!/bin/bash

#PBS -S /bin/sh
#PBS -N glove_stackexchange_100_et_0.03_it_1
#PBS -l pmem=64g
#PBS -m abe
#PBS -l walltime=24:00:00 

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

#DATADIR=/fs/clip-amr/question_generation/datasets/stackoverflow/academia.stackexchange.com
#CORPUS=$DATADIR/plain_text.txt
DATADIR=/fs/clip-scratch/raosudha/datasets/stackoverflow
CORPUS=$DATADIR/stackexchange_datadump.txt
VOCAB_FILE=$DATADIR/vocab.txt
COOCCURRENCE_FILE=$DATADIR/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=$DATADIR/cooccurrence.shuf.bin
BUILDDIR=/fs/clip-sw/user-supported/GloVe-1.2/build
#BUILDDIR=/fs/clip-xling/CLTE/vector_models/glove
SAVE_FILE=$DATADIR/vectors_100_et_0.03_it_1
VERBOSE=2
MEMORY=64.0
VOCAB_MIN_COUNT=100
MAX_VOCAB=100000
VECTOR_SIZE=100
MAX_ITER=1
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=4
X_MAX=10

#$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$BUILDDIR/vocab_count -max-vocab $MAX_VOCAB -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
$BUILDDIR/glove -save-file $SAVE_FILE -eta 0.03 -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
