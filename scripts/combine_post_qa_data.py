import sys, os
import cPickle as p

if __name__ == "__main__":
	if len(sys.argv) < 5:
		print "usage: python combine_post_qa_data.py <output_all_data_posts.p> <output_all_data_questions.p> <output_all_data_answers.p> <list_of_folders>"
		sys.exit(0)
	folders = sys.argv[4:]
	all_data_posts = []
	all_data_questions = []
	all_data_answers = []
	for folder in folders:
		print folder
		if not os.path.isfile(os.path.join(folder, "data_posts.p")):
			continue
		all_data_posts += p.load(open(os.path.join(folder, 'data_posts.p'), 'rb'))
		all_data_questions += p.load(open(os.path.join(folder, 'data_questions.p'), 'rb'))
		all_data_answers += p.load(open(os.path.join(folder, 'data_answers.p'), 'rb'))
	p.dump(all_data_posts, open(sys.argv[1], 'wb'))
	p.dump(all_data_questions, open(sys.argv[2], 'wb'))
	p.dump(all_data_answers, open(sys.argv[3], 'wb'))
