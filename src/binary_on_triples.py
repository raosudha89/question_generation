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

def get_data_masks(content, max_len):
	if len(content) > max_len:
		data = content[:max_len]
		data_mask = np.ones(max_len)
	else:
		data = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
		data_mask = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_mask

def generate_data(post_sents, ans_list, args):
	data_size = 2*len(post_sents)
	data_post_sents = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.int32)
	data_post_sent_masks = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.float32)
	data_labels = np.zeros(data_size, dtype=np.int32)

	for i in range(data_size/2):
		for j in range(min(args.post_max_sents, len(post_sents[i]))):
			data_post_sents[i][j], data_post_sent_masks[i][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
		data_labels[i] = 0
	for i in range(data_size/2):
		for j in range(min(args.post_max_sents, len(post_sents[i])-1)):
			data_post_sents[data_size/2+i][j], data_post_sent_masks[data_size/2+i][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
		data_post_sents[data_size/2+i][-1], data_post_sent_masks[data_size/2+i][-1] = get_data_masks(ans_list[i][0], args.post_max_sent_len)
		data_labels[data_size/2+i] = 1

	all_data = zip(data_post_sents, data_post_sent_masks, data_labels)
	random.shuffle(all_data)
	data_post_sents, data_post_sent_masks, data_labels = zip(*all_data)
	return np.array(data_post_sents), np.array(data_post_sent_masks), np.array(data_labels)

def build_lstm(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	post_sents = T.itensor3()
	post_sent_masks = T.ftensor3()
	post_labels = T.ivector()
	post_ans_labels = T.ivector()

	l_post_in = [None]*args.post_max_sents
	l_post_mask = [None]*args.post_max_sents
	l_post_emb = [None]*args.post_max_sents
	l_post_lstm = [None]*args.post_max_sents
	post_out = [None]*args.post_max_sents

	l_post_in[0] = lasagne.layers.InputLayer(shape=(None, args.post_max_sent_len), input_var=post_sents[0])
	l_post_mask[0] = lasagne.layers.InputLayer(shape=(None, args.post_max_sent_len), input_var=post_sent_masks[0])
	l_post_emb[0] = lasagne.layers.EmbeddingLayer(l_post_in[0], len_voc, word_emb_dim, W=word_embeddings)
	l_post_lstm[0] = lasagne.layers.LSTMLayer(l_post_emb[0], args.hidden_dim, mask_input=l_post_mask[0], )

	for i in range(1, args.post_max_sents):
		l_post_in[i] = lasagne.layers.InputLayer(shape=(None, args.post_max_sent_len), input_var=post_sents[i])
		l_post_mask[i] = lasagne.layers.InputLayer(shape=(None, args.post_max_sent_len), input_var=post_sent_masks[i])
		l_post_emb[i] = lasagne.layers.EmbeddingLayer(l_post_in[i], len_voc, word_emb_dim, W=word_embeddings)

		l_post_lstm[i] = lasagne.layers.LSTMLayer(l_post_emb[i], args.hidden_dim, mask_input=l_post_mask[i],\
									ingate=lasagne.layers.Gate(W_in=l_post_lstm[0].W_in_to_ingate,\
																W_hid=l_post_lstm[0].W_hid_to_ingate,\
																b=l_post_lstm[0].b_ingate,\
																nonlinearity=l_post_lstm[0].nonlinearity_ingate),\
									outgate=lasagne.layers.Gate(W_in=l_post_lstm[0].W_in_to_outgate,\
																W_hid=l_post_lstm[0].W_hid_to_outgate,\
																b=l_post_lstm[0].b_outgate,\
																nonlinearity=l_post_lstm[0].nonlinearity_outgate),\
									forgetgate=lasagne.layers.Gate(W_in=l_post_lstm[0].W_in_to_forgetgate,\
																W_hid=l_post_lstm[0].W_hid_to_forgetgate,\
																b=l_post_lstm[0].b_forgetgate,\
																nonlinearity=l_post_lstm[0].nonlinearity_forgetgate),\
									cell=lasagne.layers.Gate(W_in=l_post_lstm[0].W_in_to_cell,\
																W_hid=l_post_lstm[0].W_hid_to_cell,\
																b=l_post_lstm[0].b_cell,\
																nonlinearity=l_post_lstm[0].nonlinearity_cell),\
									peepholes=False,\
									)

		# freeze embeddings
		if freeze:
			l_post_emb[i].params[l_post_emb[i].W].remove('trainable')

	for i in range(args.post_max_sents):
		post_out[i] = lasagne.layers.get_output(l_post_lstm[i])
		post_out[i] = T.mean(post_out[i] * post_sent_masks[i][:,:,None], axis=1)
	
	post_out = T.mean(post_out, axis=0)
	
	l_post_out = lasagne.layers.InputLayer(shape=(None, args.hidden_dim), input_var=post_out)
	l_post_dense = lasagne.layers.DenseLayer(l_post_out, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
	post_preds = lasagne.layers.get_output(l_post_dense)
	loss = T.sum(lasagne.objectives.binary_crossentropy(post_preds, post_labels))

	post_params = lasagne.layers.get_all_params(l_post_lstm[0], trainable=True)
	post_emb_params = lasagne.layers.get_all_params(l_post_emb[0], trainbale=True)
	dense_params = lasagne.layers.get_all_params(l_post_dense, trainable=True)
	all_params = post_params + post_emb_params + dense_params
	
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)

	train_fn = theano.function([post_sents, post_sent_masks, post_labels], \
									[loss, post_preds], updates=updates)
	dev_fn = theano.function([post_sents, post_sent_masks, post_labels], \
									[loss, post_preds])
	
	return train_fn, dev_fn

def iterate_minibatches(post_sents, post_sent_masks, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(post_sents.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, post_sents.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield post_sents[excerpt], post_sent_masks[excerpt], labels[excerpt]

def validate(val_fn, fold_name, epoch, fold, args):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	total = 0
	ones = 0
	post_sents, post_sent_masks, labels = fold
	for p, pm, l in iterate_minibatches(post_sents, post_sent_masks, labels, args.batch_size, shuffle=True):
		p = np.transpose(p, (1, 0, 2))
		pm = np.transpose(pm, (1, 0, 2))
		loss, post_preds = val_fn(p, pm, l)
		cost += loss
		for i, pred in enumerate(post_preds):
			if (pred < 0.5 and l[i] == 0) or (pred >= 0.5 and l[i] == 1):
				corr += 1 
			if pred >= 0.5:
				ones += 1
			total += 1
		num_batches += 1

	lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, \
					corr*1.0/total, time.time()-start)
	print lstring
	print ones*1.0/total

def main(args):
	post_sent_vectors = p.load(open(args.post_sent_vectors, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors, 'rb'))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	N = args.no_of_candidates
	
	start = time.time()
	print 'generating data'
	post_sents, post_sent_masks, labels = \
					generate_data(post_sent_vectors, ans_list_vectors, args)
	print 'done! Time taken: ', time.time() - start

	t_size = int(len(post_sents)*0.8)
	train = [post_sents[:t_size], post_sent_masks[:t_size], labels[:t_size]]
	dev = [post_sents[t_size:], post_sent_masks[t_size:], labels[t_size:]]

	start = time.time()
	print 'compiling graph...'
	train_fn, dev_fn  = build_lstm(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start

	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'Train', epoch, train, args)
		validate(dev_fn, '\t DEV', epoch, dev, args)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--post_sent_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--ans_list_vectors", type = str)
	argparser.add_argument("--utility_post_vectors", type = str)
	argparser.add_argument("--utility_post_sent_vectors", type = str)
	argparser.add_argument("--utility_labels", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 200)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 100)
	argparser.add_argument("--post_max_sents", type = int, default = 10)
	argparser.add_argument("--post_max_sent_len", type = int, default = 10)
	argparser.add_argument("--ques_max_len", type = int, default = 10)
	argparser.add_argument("--ans_max_len", type = int, default = 10)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
