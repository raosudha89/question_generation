import sys, pdb
import csv

category_to_label = {'clarification_question':0, \
					 'providing_an_answer_or_a_suggestion_even_if_phrased_as_a_rhetorical_question':1, \
					 'neither':2}

if __name__ == "__main__":
	crowdflower_results_csv = sys.argv[1]
	askubuntu_labels_file = open(sys.argv[2], 'w')
	unix_labels_file = open(sys.argv[3], 'w')
	superuser_labels_file = open(sys.argv[4], 'w')
	askubuntu_labels = {}
	unix_labels = {}
	superuser_labels = {}
	i = 0
	with open(crowdflower_results_csv) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			category = row['choose_one_of_the_following_categories']
			label = category_to_label[category]
			sitename, post_id = row['post_id'].split('_')
			if sitename == 'askubuntu':
				askubuntu_labels[post_id] = label
			elif sitename == 'unix':
				unix_labels[post_id] = label
			elif sitename == 'superuser':
				superuser_labels[post_id] = label
			else:
				pdb.set_trace()
	for post_id, label in askubuntu_labels.iteritems():
		askubuntu_labels_file.write('%s\t%s\n' % (post_id, label))
	for post_id, label in unix_labels.iteritems():
		unix_labels_file.write('%s\t%s\n' % (post_id, label))
	for post_id, label in superuser_labels.iteritems():
		superuser_labels_file.write('%s\t%s\n' % (post_id, label))
