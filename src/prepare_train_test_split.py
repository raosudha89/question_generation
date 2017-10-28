import sys
import random

if __name__ == "__main__":
	posts_tsv_file = open(sys.argv[1], 'r')
	train_post_ids_tsv_file = open(sys.argv[2], 'w')
	test_post_ids_tsv_file = open(sys.argv[3], 'w')
	post_ids = []
	i = 0
	for line in posts_tsv_file.readlines():
		if i == 0:
			i += 1
			continue
		post_ids.append(line.split('\t')[0])
	random.shuffle(post_ids)
	size = len(post_ids)
	train_post_ids, test_post_ids = post_ids[:int(size*0.9)], post_ids[int(size*0.9):]
	for post_id in train_post_ids:
		train_post_ids_tsv_file.write('%s\t' % post_id)
	for post_id in test_post_ids:
		test_post_ids_tsv_file.write('%s\t' % post_id)
