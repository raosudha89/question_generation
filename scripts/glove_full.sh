#!/bin/bash

#PBS -S /bin/sh
#PBS -N glove_en_wiki_giga_500
#PBS -l pmem=120g
#PBS -m abe
#PBS -q shallow
#PBS -l walltime=96:00:00 

# Parameters that you should change
EXP_FOLDER=/fs/clip-scratch/yogarshi/datadumps/glove_models/giga+wiki_en
WINDOW=8
CODE_FOLDER=/fs/clip-xling/CLTE/vector_models/glove/
CORPUS="${EXP_FOLDER}"/en_wiki_giga.txt
iterations=40



#Do glove stuff
# 1. Collect voab
"${CODE_FOLDER}"/vocab_count -verbose 2 -max-vocab 2000000000 -min-count 0 < "${CORPUS}" > "${EXP_FOLDER}/full_vocab.txt"
# 2. Get coocurance counts
"${CODE_FOLDER}"/cooccur -window-size "${WINDOW}" -vocab-file "${EXP_FOLDER}"/vocab.txt -memory 27.0 < "${CORPUS}" > "${EXP_FOLDER}"/cooccurance.bin 
# 3. Shuffle
"${CODE_FOLDER}"/shuffle -memory 27.0 < "${EXP_FOLDER}"/cooccurance.bin  > "${EXP_FOLDER}"/cooccurance.shuf.bin 
# 4. Run GloVe
"${CODE_FOLDER}"/glove -input-file "${EXP_FOLDER}"/cooccurance.shuf.bin -vocab-file "${EXP_FOLDER}"/vocab.txt -vector-size 500 -model 1 -save-file "${EXP_FOLDER}"/vectors_500 -threads 12 -iter "${iterations}" -binary 2 -eta 0.03

