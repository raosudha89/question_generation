import sys, os
import cPickle as p

if __name__ == "__main__":
	if len(sys.argv) < 5:
		print "usage: python combine_pickled_data.py <output_all_posts.p> <output_all_questions.p> <output_all_labels.p> <list_of_folders>"
		sys.exit(0)
	folders = sys.argv[4:]
	all_posts = []
	all_questions = []
	all_labels = []
	for folder in folders:
		print folder
		if not os.path.isfile(os.path.join(folder, "main_labelled_posts.p")):
			continue
		all_posts += p.load(open(os.path.join(folder, 'main_labelled_posts.p'), 'rb'))
		all_questions += p.load(open(os.path.join(folder, 'main_labelled_questions.p'), 'rb'))
		all_labels += p.load(open(os.path.join(folder, 'main_labels.p'), 'rb'))
	p.dump(all_posts, open(sys.argv[1], 'wb'))
	p.dump(all_questions, open(sys.argv[2], 'wb'))
	p.dump(all_labels, open(sys.argv[3], 'wb'))
