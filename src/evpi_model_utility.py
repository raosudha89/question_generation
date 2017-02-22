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
import gc

def get_data_masks(content, max_len):
	if len(content) > max_len:
		data = content[:max_len]
		data_mask = np.ones(max_len)
	else:
		data = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
		data_mask = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_mask

def generate_utility_data(posts, labels, args):
	data_size = len(posts)
	data_posts = np.zeros((data_size, args.post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((data_size, args.post_max_len), dtype=np.float32)
	
	data_labels = np.array(labels, dtype=np.int32)
	for i in range(data_size):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], args.post_max_len)
			
	return data_posts, data_post_masks, data_labels

def build_utility_lstm(utility_posts, utility_post_masks, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):
	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=utility_posts[0])
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=utility_post_masks[0])
	#l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
	#l_drop_in = lasagne.layers.DropoutLayer(l_emb, p=0.2)
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	utility_post_out[0] = lasagne.layers.get_output(l_lstm)
	utility_post_out[0] = T.mean(utility_post_out[0] * utility_post_masks[0][:,:,None], axis=1)
	#l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	
	return utility_post_out, params

def build_evpi_model(word_embeddings, len_voc, word_emb_dim, args, freeze=False):

	# input theano vars
	
	utility_posts = T.itensor3()
	utility_post_masks = T.ftensor3()
	utility_labels = T.ivector()

	utility_post_out, utility_post_lstm_params = build_utility_lstm(utility_posts, utility_post_masks, \
																	args.post_max_len, \
																	word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)

	utility_post_concatenate = T.concatenate(utility_post_out, axis=1)
	l_utility_post_out = lasagne.layers.InputLayer(shape=(args.batch_size, args.post_max_sents*args.hidden_dim), input_var=utility_post_concatenate)
	l_utility_post_dense = lasagne.layers.DenseLayer(l_utility_post_drop, num_units=args.hidden_dim,\
													nonlinearity=lasagne.nonlinearities.rectify)
	l_utility_post_dense2 = lasagne.layers.DenseLayer(l_utility_post_dense, num_units=1,\
													nonlinearity=lasagne.nonlinearities.sigmoid)
	utility_preds = lasagne.layers.get_output(l_utility_post_dense2)
	utility_loss = T.sum(lasagne.objectives.binary_crossentropy(utility_preds, utility_labels))

	utility_dense2_params = lasagne.layers.get_all_params(l_utility_post_dense2, trainable=True)
	utility_all_params = utility_post_lstm_params + utility_dense2_params

	utility_loss += args.rho * sum(T.sum(l ** 2) for l in utility_all_params)
	
	utility_updates = lasagne.updates.adam(utility_loss, utility_all_params, learning_rate=args.learning_rate)
	
	utility_train_fn = theano.function([utility_posts, utility_post_masks, utility_labels], \
									[utility_preds, utility_loss], updates=utility_updates)
	utility_dev_fn = theano.function([utility_posts, utility_post_masks, utility_labels], \
									[utility_preds, utility_loss],)

	return utility_train_fn, utility_dev_fn

def utility_iterate_minibatches(posts, post_masks, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], post_masks[excerpt], labels[excerpt]

def validate(utility_val_fn, fold_name, epoch, utility_fold, args):
	start = time.time()
	utility_num_batches = 0
	utility_corr = 0
	utility_total = 0
	utility_cost = 0
	batch_size = args.batch_size
	
	utility_posts, utility_post_masks, utility_labels = utility_fold
	
	for up, upm, ul in utility_iterate_minibatches(utility_posts, utility_post_masks, utility_labels, batch_size, shuffle=True):
		up = np.transpose(up, (1, 0, 2))	
		upm = np.transpose(upm, (1, 0, 2))	
		utility_preds, utility_loss = utility_val_fn(up, upm, ul)
		for j in range(len(utility_preds)):
			if (utility_preds[j] >= 0.5 and ul[j] == 1) or (utility_preds[j] < 0.5 and ul[j] == 0):
				utility_corr += 1
			utility_total += 1
		utility_cost += utility_loss
		utility_num_batches += 1
		
	lstring = '%s: epoch:%d, utility_cost:%f, utility_acc:%f time:%d' % \
				(fold_name, epoch, utility_cost*1.0/utility_num_batches, utility_corr*1.0/utility_total, time.time()-start)
	print lstring

def main(args):
	post_vectors = p.load(open(args.post_vectors_train, 'rb'))
	ques_list_vectors = p.load(open(args.ques_list_vectors_train, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors_train, 'rb'))
	utility_post_vectors = p.load(open(args.utility_post_vectors_train, 'rb'))
	utility_labels = p.load(open(args.utility_labels_train, 'rb'))

	post_vectors_test = p.load(open(args.post_vectors_test, 'rb'))
	ques_list_vectors_test = p.load(open(args.ques_list_vectors_test, 'rb'))
	ans_list_vectors_test = p.load(open(args.ans_list_vectors_test, 'rb'))
	utility_post_vectors_test = p.load(open(args.utility_post_vectors_test, 'rb'))
	utility_labels_test = p.load(open(args.utility_labels_test, 'rb'))

	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	
	print 'vocab_size ', vocab_size, 

	start = time.time()
	print 'generating utility data'
	data_utility_posts, data_utility_post_masks, data_utility_labels = \
					generate_utility_data(utility_post_vectors, utility_labels, args)
	data_utility_posts_test, data_utility_post_masks_test, data_utility_labels_test = \
					generate_utility_data(utility_post_vectors_test, utility_labels_test, args)
	print 'done! Time taken: ', time.time() - start

	utility_train_size = int(len(data_utility_posts)*0.8)
	utility_dev_size = int(len(data_utility_posts)*0.2)/2

	utility_train = [data_utility_posts[:utility_train_size], \
					data_utility_post_masks[:utility_train_size], \
					data_utility_labels[:utility_train_size]]

	utility_dev = [data_utility_posts[utility_train_size: utility_train_size+utility_dev_size], \
					data_utility_post_masks[utility_train_size: utility_train_size+utility_dev_size], \
					data_utility_labels[utility_train_size: utility_train_size+utility_dev_size]]

	utility_test = [np.concatenate((data_utility_posts_test, \
									data_utility_posts[utility_train_size+utility_dev_size:])), \
					np.concatenate((data_utility_post_masks_test, \
									data_utility_post_masks[utility_train_size+utility_dev_size:])), \
					np.concatenate((data_utility_labels_test, \
									data_utility_labels[utility_train_size+utility_dev_size:]))] 

	print 'Size of utility training data: ', utility_train_size
	print 'Size of utility dev data: ', utility_dev_size

	start = time.time()
	print 'compiling graph...'
	utility_train_fn, utility_dev_fn = build_evpi_model(word_embeddings, vocab_size, word_emb_dim, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start

	gc.collect()
	#train_fn, dev_fn, utility_train_fn, utility_dev_fn = None, None, None, None
	# train network
	for epoch in range(args.no_of_epochs):
		validate(utility_train_fn, 'Train', epoch, utility_train, args)
		validate(utility_dev_fn, '\t DEV', epoch, utility_dev, args)
		#validate(utility_dev_fn, '\t TEST', epoch, utility_test, args)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_vectors_train", type = str)
	argparser.add_argument("--ques_list_vectors_train", type = str)
	argparser.add_argument("--ans_list_vectors_train", type = str)
	argparser.add_argument("--utility_post_vectors_train", type = str)
	argparser.add_argument("--utility_labels_train", type = str)
	argparser.add_argument("--post_vectors_test", type = str)
	argparser.add_argument("--ques_list_vectors_test", type = str)
	argparser.add_argument("--ans_list_vectors_test", type = str)
	argparser.add_argument("--utility_post_vectors_test", type = str)
	argparser.add_argument("--utility_labels_test", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 100)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_sents", type = int, default = 10)
	argparser.add_argument("--post_max_len", type = int, default = 10)
	argparser.add_argument("--ques_max_len", type = int, default = 10)
	argparser.add_argument("--ans_max_len", type = int, default = 10)
	argparser.add_argument("--_lambda", type = float, default = 0.5)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
