DATADIR=/fs/clip-corpora/stackexchange
DATAOUTDIR=/fs/clip-amr/question_generation/datasets/stackexchange
SCRIPTDIR=/fs/clip-amr/question_generation/scripts
for site in $DATADIR/*.com;
do
	sitename=`basename $site`
	if [ -f "$DATADIR/$sitename/Posts.xml" ];
	then 
		echo $sitename
		python $SCRIPTDIR/generate_posts_questions.py $DATADIR/$sitename/Posts.xml $DATADIR/$sitename/Comments.xml $DATAOUTDIR/$sitename/main_posts.p $DATAOUTDIR/$sitename/main_post_questions.p	
	fi
done
