DATADIR=/fs/clip-scratch/raosudha/datasets/stackoverflow
SCRIPTDIR=/fs/clip-amr/question_generation/scripts
for site in $DATADIR/*;
do
	echo $site
	cat $site/plain_text.txt >> $DATADIR/stackexchange_datadump.txt
done
