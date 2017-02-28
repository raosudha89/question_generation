import sys
import argparse
import theano, lasagne
import numpy as np
import cPickle as p
import theano.tensor as T
from collections import Counter
import pdb
import time
import random, math

DEPTH = 5
def get_data_masks(content, max_len):
	if len(content) > max_len:
		data = content[:max_len]
		data_mask = np.ones(max_len)
	else:
		data = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
		data_mask = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_mask

def generate_data(questions, args):
	data_size = len(questions)
	data_questions = np.zeros((data_size, args.ques_max_len), dtype=np.int32)
	data_question_masks = np.zeros((data_size, args.ques_max_len), dtype=np.float32)
	
	for i in range(data_size):
		data_questions[i], data_question_masks[i] = get_data_masks(questions[i], args.ques_max_len)
		
	return data_questions, data_question_masks

def build_classifier(word_embeddings, len_voc, word_emb_dim, args):

	# input theano vars
	questions = T.imatrix()
	question_masks = T.fmatrix()
	labels = T.ivector()

	l_in = lasagne.layers.InputLayer(shape=(None, args.ques_max_len), input_var=questions)
	l_mask = lasagne.layers.InputLayer(shape=(None, args.ques_max_len), input_var=question_masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	out = lasagne.layers.get_output(l_emb)
	out = T.mean(out * question_masks[:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	l_emb_in = lasagne.layers.InputLayer(shape=(None, args.hidden_dim), input_var=out)
	l_dense = lasagne.layers.DenseLayer(l_emb_in, num_units=args.hidden_dim,\
										nonlinearity=lasagne.nonlinearities.rectify)
	for i in range(DEPTH):
		l_dense = lasagne.layers.DenseLayer(l_emb_in, num_units=args.hidden_dim,\
										nonlinearity=lasagne.nonlinearities.rectify)
	l_dense = lasagne.layers.DenseLayer(l_dense, num_units=1,\
										nonlinearity=lasagne.nonlinearities.sigmoid)
	preds = lasagne.layers.get_output(l_dense)
	loss = T.sum(lasagne.objectives.binary_crossentropy(preds, labels))
	params = lasagne.layers.get_all_params(l_dense, trainable=True)
	print 'Params in model: ', lasagne.layers.count_params(l_dense)
	
	loss += args.rho * sum(T.sum(l ** 2) for l in params)

	updates = lasagne.updates.adam(loss, params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([questions, question_masks, labels], \
									[preds, loss], updates=updates)
	test_fn = theano.function([questions, question_masks, labels], \
									[preds, loss],)
	return train_fn, test_fn

def validate(val_fn, fold_name, epoch, fold, args, out_file=None):
	start = time.time()
	corr = 0
	total = 0
	questions, question_masks, labels = fold
	preds, cost = val_fn(questions, question_masks, labels)
	normalized_preds = [1]*len(preds)
	for i in range(len(preds)):
		if preds[i] < 0.5:
			normalized_preds[i] = 0
	for i in range(len(preds)):
		if normalized_preds[i] == labels[i]:
			corr += 1
		total += 1
	#pdb.set_trace()
	print 'Predictions: %d %d ' % (normalized_preds.count(0), normalized_preds.count(1))
		
	lstring = '%s: epoch:%d, cost:%f, acc:%f,time:%d' % \
				(fold_name, epoch, cost, \
					corr*1.0/total, time.time()-start)
	print lstring
		
def get_labelled_ids(labelled_file):
	i = 0
	labelled_ids = []
	labels = []
	for line in labelled_file.readlines():
		if i == 0:
			i += 1
			continue
		splits = line.split(',')
		labelled_ids.append(splits[3])
		if splits[1] == 'clarification_question':
			labels.append(1)
		else:
			labels.append(0)
	return labelled_ids, labels

def shuffle_data(q, qm, l):
	sq = [None]*len(q)
	sqm = [None]*len(qm)
	sl = [None]*len(l)
	indexes = range(len(q))
	random.shuffle(indexes)
	for i, index in enumerate(indexes):
		sq[i] = q[index]
		sqm[i] = qm[index]
		sl[i] = l[index]
		
	return np.array(sq), np.array(sqm), np.array(sl)

def main(args):
	post_ids = p.load(open(args.post_ids_train, 'rb'))
	post_ids = np.array(post_ids)
	ques_list_vectors = p.load(open(args.ques_list_vectors_train, 'rb'))
	if len(ques_list_vectors) != len(post_ids): #for ubuntu,unix,superuser combined data we don't have all train post_ids
		post_ids = np.zeros(len(ques_list_vectors))
		
	labelled_file = open(args.clarification_labels_file, 'r')
	clarification_labelled_ids, clarification_labels = get_labelled_ids(labelled_file)
	ques_vectors = []
	ques_labels = []
	for i, post_id in enumerate(post_ids):
		if post_id in clarification_labelled_ids:
			ques_vectors.append(ques_list_vectors[i][0])
			ques_labels.append(clarification_labels[clarification_labelled_ids.index(post_id)])
	
	print 'Positives: %d Negatives: %d' % (ques_labels.count(1), ques_labels.count(0))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])

	start = time.time()
	print 'generating data'
	ques_vectors, ques_mask_vectors = generate_data(ques_vectors, args)
	ques_labels = np.array(ques_labels, dtype=np.int32)
	ques_vectors, ques_mask_vectors, ques_labels = shuffle_data(ques_vectors, ques_mask_vectors, ques_labels)
	print 'done! Time taken: ', time.time() - start

	t_size = int(len(ques_vectors)*0.8)
	
	train = [ques_vectors[:t_size], ques_mask_vectors[:t_size], ques_labels[:t_size]]
	test = [ques_vectors[t_size:], ques_mask_vectors[t_size:], ques_labels[t_size:]]

	print 'Size of training data: ', t_size
	print 'Size of test data: ', len(ques_vectors)-t_size

	start = time.time()
	print 'compiling graph...'
	train_fn, test_fn, = build_classifier(word_embeddings, vocab_size, word_emb_dim, args)
	print 'done! Time taken: ', time.time()-start

	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'TRAIN', epoch, train, args)
		validate(test_fn, '\t TEST', epoch, test, args)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_ids_train", type = str)
	argparser.add_argument("--ques_list_vectors_train", type = str)
	argparser.add_argument("--clarification_labels_file", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 50)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--ques_max_len", type = int, default = 20)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
