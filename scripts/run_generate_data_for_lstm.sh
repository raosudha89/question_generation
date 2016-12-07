DATADIR=/fs/clip-amr/question_generation/datasets/stackexchange
SCRIPTDIR=/fs/clip-amr/question_generation/scripts
for site in $DATADIR/*.com;
do
	sitename=`basename $site`
	if [ -f "$DATADIR/$sitename/main_posts.p" ];
	then 
		echo $sitename
		python $SCRIPTDIR/generate_data_for_lstm.py $DATADIR/$sitename/main_posts.p $DATADIR/$sitename/main_post_questions.p $DATADIR/vocab.p $DATADIR/$sitename/main_labelled_posts.p $DATADIR/$sitename/main_labelled_questions.p $DATADIR/$sitename/main_labels.p	
	fi
done
