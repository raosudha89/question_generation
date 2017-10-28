import sys
import random

if __name__ == "__main__":
	post_data_tsv = open(sys.argv[1], 'r')
	qa_data_tsv = open(sys.argv[2], 'r')
	train_post_ids_tsv = open(sys.argv[3], 'r')
	clarification_data_tsv = open(sys.argv[4], 'w')
	sitename = sys.argv[5]
	train_post_ids = train_post_ids_tsv.readlines()[0].strip('\n').split('\t')
	random.shuffle(train_post_ids)
	#candidate_post_ids = train_post_ids[:int(len(train_post_ids)*0.1)] # 10% of train
	candidate_post_ids = train_post_ids[:int(len(train_post_ids)*0.5)] # 50% of test
	titles = {}
	posts = {}
	questions = {}
	i = 0
	for line in post_data_tsv.readlines():
		if i == 0:
			i += 1
			continue
		post_id, title, post = line.strip('\n').split('\t')
		titles[post_id] = title
		posts[post_id] = post
	i = 0
	for line in qa_data_tsv.readlines():
		if i == 0:
			i += 1
			continue
		splits = line.split('\t')
		post_id = splits[0]
		question = splits[1]
		questions[post_id] = question
	clarification_data_tsv.write('%s\t %s\t %s\t %s\n' % ('post_id', 'title', 'post', 'question'))
	for post_id in candidate_post_ids:
		clarification_data_tsv.write('%s\t %s\t %s\t %s\n' % (sitename+'_'+post_id, titles[post_id], posts[post_id], questions[post_id]))
