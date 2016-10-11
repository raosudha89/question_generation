DATADIR=/fs/clip-corpora/stackoverflow
DATAOUTDIR=/fs/clip-scratch/raosudha/datasets/stackoverflow
SCRIPTDIR=/fs/clip-amr/question_generation/scripts
for site in $DATADIR/*.7z;
do
	sitename=`basename ${site%.7z}`
	echo $sitename
	mkdir -p $DATAOUTDIR/$sitename
	python $SCRIPTDIR/extract_plain_text.py $DATADIR/$sitename/Posts.xml $DATADIR/$sitename/Comments.xml $DATAOUTDIR/$sitename/plain_text.txt
done
