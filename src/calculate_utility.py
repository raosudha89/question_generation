import sys
import argparse
import theano, lasagne
import numpy as np
import cPickle as p
import theano.tensor as T
from collections import Counter
import pdb
import time

def iterate_minibatches(posts, masks, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], masks[excerpt], labels[excerpt]

def build_lstm(word_embeddings, len_voc, word_emb_dim, hidden_dim, max_len, batch_size, learning_rate, rho, freeze=False):

	# input theano vars
	posts = T.imatrix(name='post')
	masks = T.matrix(name='mask')
	labels = T.ivector('label')

	# define network
	l_in = lasagne.layers.InputLayer(shape=(word_emb_dim, max_len), input_var=posts)
	l_mask = lasagne.layers.InputLayer(shape=(word_emb_dim, max_len), input_var=masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)

	# now feed sequences of spans into VAN
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )

	# freeze embeddings
	if freeze:
		l_emb.params[l_emb.W].remove('trainable')

	# now predict
	l_forward_slice = lasagne.layers.SliceLayer(l_lstm, -1, 1)
	l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=2,\
									nonlinearity=lasagne.nonlinearities.softmax)

	# objective computation
	preds = lasagne.layers.get_output(l_out)
	loss = T.sum(lasagne.objectives.categorical_crossentropy(preds, labels))
	loss += rho * sum(T.sum(l ** 2) for l in lasagne.layers.get_all_params(l_out))
	all_params = lasagne.layers.get_all_params(l_out, trainable=True)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=learning_rate)

	train_fn = theano.function([posts, masks, labels], [preds, loss], updates=updates)
	val_fn = theano.function([posts, masks, labels], [preds, loss])
	return train_fn, val_fn

def validate(val_fn, fold_name, epoch, fold, batch_size):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	total = 0
	posts, post_masks, labels = fold
	for p, pm, l in iterate_minibatches(posts, post_masks, labels, batch_size, shuffle=True):
		preds, loss = val_fn(p, pm, l)
		cost += loss
		preds = np.argmax(preds, axis=1)
		for i, pred in enumerate(preds):
			if pred == labels[i]:
				corr += 1
			total += 1
		num_batches += 1

	lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
				(fold_name, epoch, cost*1.0 / num_batches, corr*1.0 / total, time.time()-start)
	print lstring

def get_data_masks(content, max_len):
	if len(content) > max_len:
		data = content[:max_len]
		data_mask = np.ones(max_len)
	else:
		data = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
		data_mask = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_mask

def generate_data(posts, post_max_len, labels):
	data_posts = np.zeros((len(posts), post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((len(posts), post_max_len), dtype=np.float32)
	labels = np.array(labels, dtype=np.int32)

	for i in range(len(posts)):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], post_max_len)

	return data_posts, data_post_masks, labels

def main(args):
	post_vectors = p.load(open(args.utility_post_vectors, 'rb'))
	labels = p.load(open(args.utility_labels, 'rb'))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	
	print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len

	start = time.time()
	print 'generating data'
	posts, post_masks, labels = generate_data(post_vectors, args.post_max_len, labels)
	print 'done! Time taken: ', time.time() - start

	t_size = int(len(posts)*0.8)
	train = [posts[:t_size], post_masks[:t_size], labels[:t_size]]
	dev = [posts[t_size:], post_masks[t_size:], labels[t_size:]]

	print 'Size of training data: ', t_size
	print 'Size of dev data: ', len(posts)-t_size

	start = time.time()
	print 'compiling graph...'
	train_fn, val_fn = build_lstm(word_embeddings, vocab_size, word_emb_dim, args.hidden_dim, args.post_max_len, args.batch_size, args.learning_rate, args.rho, freeze=freeze)
	print 'done! Time taken: ', time.time()-start
	
	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'Train', epoch, train, args.batch_size)
		validate(val_fn, '\t DEV', epoch, dev, args.batch_size)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--utility_post_vectors", type = str)
	argparser.add_argument("--utility_labels", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 200)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 100)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
