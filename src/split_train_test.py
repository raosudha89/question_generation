import os, sys
import argparse
import numpy as np
import cPickle as p

def main(args):
	post_vectors = p.load(open(args.post_vectors, 'rb'))
	post_sent_vectors = p.load(open(args.post_sent_vectors, 'rb'))
	ques_list_vectors = p.load(open(args.ques_list_vectors, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors, 'rb'))
	post_ids = p.load(open(args.post_ids, 'rb'))
	utility_post_vectors = p.load(open(args.utility_post_vectors, 'rb'))
	utility_post_sent_vectors = p.load(open(args.utility_post_sent_vectors, 'rb'))
	utility_labels = p.load(open(args.utility_labels, 'rb'))
	utility_post_ids = p.load(open(args.utility_post_ids, 'rb'))

	human_evald_post_ids_file = open(args.human_evald_post_ids, 'r')
	line = human_evald_post_ids_file.readlines()
	human_evald_post_ids = line[0].split()
	print human_evald_post_ids

	post_vectors_train = []
	post_sent_vectors_train = []
	ques_list_vectors_train = []
	ans_list_vectors_train = []
	post_ids_train = []
	utility_post_vectors_train = []
	utility_post_sent_vectors_train = []
	utility_labels_train = []
	utility_post_ids_train = []

	post_vectors_test = []
	post_sent_vectors_test = []
	ques_list_vectors_test = []
	ans_list_vectors_test = []
	post_ids_test = []
	utility_post_vectors_test = []
	utility_post_sent_vectors_test = []
	utility_labels_test = []
	utility_post_ids_test = []

	for i, post_id in enumerate(post_ids):
		if post_id in human_evald_post_ids:
			post_vectors_test.append(post_vectors[i])	
			post_sent_vectors_test.append(post_sent_vectors[i])	
			ques_list_vectors_test.append(ques_list_vectors[i])	
			ans_list_vectors_test.append(ans_list_vectors[i])	
			post_ids_test.append(post_ids[i])	
			utility_post_vectors_test.append(utility_post_vectors[i])	
			utility_post_sent_vectors_test.append(utility_post_sent_vectors[i])	
			utility_labels_test.append(utility_labels[i])	
			utility_post_ids_test.append(utility_post_ids[i])	
		else:
			post_vectors_train.append(post_vectors[i])	
			post_sent_vectors_train.append(post_sent_vectors[i])	
			ques_list_vectors_train.append(ques_list_vectors[i])	
			ans_list_vectors_train.append(ans_list_vectors[i])	
			post_ids_train.append(post_ids[i])	
			utility_post_vectors_train.append(utility_post_vectors[i])	
			utility_post_sent_vectors_train.append(utility_post_sent_vectors[i])	
			utility_labels_train.append(utility_labels[i])	
			utility_post_ids_train.append(utility_post_ids[i])	

	p.dump(post_vectors_train, open(args.post_vectors_train, 'wb'))
	p.dump(post_sent_vectors_train, open(args.post_sent_vectors_train, 'wb'))
	p.dump(ques_list_vectors_train, open(args.ques_list_vectors_train, 'wb'))
	p.dump(ans_list_vectors_train, open(args.ans_list_vectors_train, 'wb'))
	p.dump(post_ids_train, open(args.post_ids_train, 'wb'))
	p.dump(utility_post_vectors_train, open(args.utility_post_vectors_train, 'wb'))
	p.dump(utility_post_sent_vectors_train, open(args.utility_post_sent_vectors_train, 'wb'))
	p.dump(utility_labels_train, open(args.utility_labels_train, 'wb'))
	p.dump(utility_post_ids_train, open(args.utility_post_ids_train, 'wb'))

	p.dump(post_vectors_test, open(args.post_vectors_test, 'wb'))
	p.dump(post_sent_vectors_test, open(args.post_sent_vectors_test, 'wb'))
	p.dump(ques_list_vectors_test, open(args.ques_list_vectors_test, 'wb'))
	p.dump(ans_list_vectors_test, open(args.ans_list_vectors_test, 'wb'))
	p.dump(post_ids_test, open(args.post_ids_test, 'wb'))
	p.dump(utility_post_vectors_test, open(args.utility_post_vectors_test, 'wb'))
	p.dump(utility_post_sent_vectors_test, open(args.utility_post_sent_vectors_test, 'wb'))
	p.dump(utility_labels_test, open(args.utility_labels_test, 'wb'))
	p.dump(utility_post_ids_test, open(args.utility_post_ids_test, 'wb'))

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--post_sent_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--ans_list_vectors", type = str)
	argparser.add_argument("--post_ids", type = str)
	argparser.add_argument("--utility_post_vectors", type = str)
	argparser.add_argument("--utility_post_sent_vectors", type = str)
	argparser.add_argument("--utility_labels", type = str)
	argparser.add_argument("--utility_post_ids", type = str)
	argparser.add_argument("--utility_ans_list_vectors", type = str)
	argparser.add_argument("--human_evald_post_ids", type = str)

	argparser.add_argument("--post_vectors_train", type = str)
	argparser.add_argument("--post_sent_vectors_train", type = str)
	argparser.add_argument("--ques_list_vectors_train", type = str)
	argparser.add_argument("--ans_list_vectors_train", type = str)
	argparser.add_argument("--post_ids_train", type = str)
	argparser.add_argument("--utility_post_vectors_train", type = str)
	argparser.add_argument("--utility_post_sent_vectors_train", type = str)
	argparser.add_argument("--utility_labels_train", type = str)
	argparser.add_argument("--utility_post_ids_train", type = str)

	argparser.add_argument("--post_vectors_test", type = str)
	argparser.add_argument("--post_sent_vectors_test", type = str)
	argparser.add_argument("--ques_list_vectors_test", type = str)
	argparser.add_argument("--ans_list_vectors_test", type = str)
	argparser.add_argument("--post_ids_test", type = str)
	argparser.add_argument("--utility_post_vectors_test", type = str)
	argparser.add_argument("--utility_post_sent_vectors_test", type = str)
	argparser.add_argument("--utility_labels_test", type = str)
	argparser.add_argument("--utility_post_ids_test", type = str)

	args = argparser.parse_args()
	print args
	print ""
	main(args)
