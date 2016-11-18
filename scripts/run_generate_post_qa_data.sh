DATADIR=/fs/clip-corpora/stackexchange
DATAOUTDIR=/fs/clip-amr/question_generation/datasets/stackexchange
SCRIPTDIR=/fs/clip-amr/question_generation/scripts
WORDVECTORS=/fs/clip-amr/question_generation/datasets/stackexchange/vectors_200.txt
source /fs/clip-amr/isi-internship/theano-env/bin/activate
for site in $DATADIR/math.*.7z;
do
	sitename=`basename ${site%.7z}`
	echo $sitename
	mkdir -p $DATAOUTDIR/$sitename
	python $SCRIPTDIR/generate_post_qa_data.py $DATADIR/$sitename/Posts.xml $DATADIR/$sitename/PostHistory.xml $DATADIR/$sitename/Comments.xml $WORDVECTORS $DATAOUTDIR/$sitename/data_posts.p $DATAOUTDIR/$sitename/data_questions.p $DATAOUTDIR/$sitename/data_answers.p > $DATAOUTDIR/$sitename/generate_post_qa_data.out	
done
