import sys
import argparse
import theano, lasagne
import numpy as np
import cPickle as p
import theano.tensor as T
from collections import Counter
import pdb
import time
from random import randint

def iterate_minibatches(posts, post_masks, ans_list, ans_masks_list, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], post_masks[excerpt], ans_list[excerpt], ans_masks_list[excerpt], labels[excerpt]

def build_lstm(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	posts = T.imatrix()
	post_masks = T.matrix()
	ans_list = T.itensor3()
	ans_masks_list = T.ftensor3()
	labels = T.imatrix()
	pred_ans_list = T.ftensor3()

	l_post_in = lasagne.layers.InputLayer(shape=(None, args.post_max_len), input_var=posts)
	l_post_mask = lasagne.layers.InputLayer(shape=(None, args.post_max_len), input_var=post_masks)
	l_post_emb = lasagne.layers.EmbeddingLayer(l_post_in, len_voc, word_emb_dim, W=word_embeddings)

	l_post_lstm = lasagne.layers.LSTMLayer(l_post_emb, args.hidden_dim, mask_input=l_post_mask)

	# freeze embeddings
	if freeze:
		l_post_emb.params[l_post_emb.W].remove('trainable')

	post_out = lasagne.layers.get_output(l_post_lstm)
	post_out = T.mean(post_out * post_masks[:,:,None], axis=1)
	
	l_ans_in = [None]*N
	l_ans_mask = [None]*N
	l_ans_emb = [None]*N
	l_ans_lstm = [None]*N
	ans_out = [None]*N

	for i in range(N):
		l_ans_in[i] = lasagne.layers.InputLayer(shape=(None, args.ans_max_len), input_var=ans_list[i])
		l_ans_mask[i] = lasagne.layers.InputLayer(shape=(None, args.ans_max_len), input_var=ans_masks_list[i])
		l_ans_emb[i] = lasagne.layers.EmbeddingLayer(l_ans_in[i], len_voc, word_emb_dim, W=word_embeddings)
				
	l_ans_lstm[0] = lasagne.layers.LSTMLayer(l_ans_emb[0], args.hidden_dim, mask_input=l_ans_mask[0], )
	for i in range(1, N):
		l_ans_lstm[i] = lasagne.layers.LSTMLayer(l_ans_emb[i], args.hidden_dim, mask_input=l_ans_mask[i],\
									ingate=lasagne.layers.Gate(W_in=l_ans_lstm[0].W_in_to_ingate,\
																W_hid=l_ans_lstm[0].W_hid_to_ingate,\
																b=l_ans_lstm[0].b_ingate,\
																nonlinearity=l_ans_lstm[0].nonlinearity_ingate),\
									outgate=lasagne.layers.Gate(W_in=l_ans_lstm[0].W_in_to_outgate,\
																W_hid=l_ans_lstm[0].W_hid_to_outgate,\
																b=l_ans_lstm[0].b_outgate,\
																nonlinearity=l_ans_lstm[0].nonlinearity_outgate),\
									forgetgate=lasagne.layers.Gate(W_in=l_ans_lstm[0].W_in_to_forgetgate,\
																W_hid=l_ans_lstm[0].W_hid_to_forgetgate,\
																b=l_ans_lstm[0].b_forgetgate,\
																nonlinearity=l_ans_lstm[0].nonlinearity_forgetgate),\
									cell=lasagne.layers.Gate(W_in=l_ans_lstm[0].W_in_to_cell,\
																W_hid=l_ans_lstm[0].W_hid_to_cell,\
																b=l_ans_lstm[0].b_cell,\
																nonlinearity=l_ans_lstm[0].nonlinearity_cell),\
									peepholes=False,\
									)

	for i in range(N):
		ans_out[i] = lasagne.layers.get_output(l_ans_lstm[i])
		ans_out[i] = T.mean(ans_out[i] * ans_masks_list[i][:,:,None], axis=1)

	M = theano.shared(np.eye(args.hidden_dim, dtype=np.float32))
	post_ans_out = [None]*N
	post_pred_ans_out = [None]*N
	for i in range(N):
		post_ans_out[i] = T.sum(T.dot(post_out,M)*ans_out[i], axis=1)	
		post_pred_ans_out[i] = T.sum(T.dot(post_out,M)*pred_ans_list[i], axis=1) 

	ans_probs = lasagne.nonlinearities.softmax(T.transpose(T.stack(post_ans_out)))
	pred_ans_probs = lasagne.nonlinearities.softmax(T.transpose(T.stack(post_pred_ans_out)))

	post_params = lasagne.layers.get_all_params(l_post_lstm, trainable=True)
	post_emb_params = lasagne.layers.get_all_params(l_post_emb, trainbale=True)
	ans_params = lasagne.layers.get_all_params(l_ans_lstm[0], trainable=True)
	ans_emb_params = lasagne.layers.get_all_params(l_ans_emb[0], trainbale=True)
	all_params = post_params + post_emb_params + ans_params + ans_emb_params + [M]
	
	# objective computation
	loss = T.sum(lasagne.objectives.categorical_crossentropy(ans_probs, labels))
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)

	train_fn = theano.function([posts, post_masks, ans_list, ans_masks_list, labels], \
									[ans_probs, loss], updates=updates)
	dev_fn = theano.function([posts, post_masks, ans_list, ans_masks_list, labels], \
									[ans_probs, loss],)
	test_fn = theano.function([posts, post_masks, pred_ans_list], pred_ans_probs)
	return train_fn, dev_fn, test_fn

def swap(a, b):
	return b, a

def shuffle(a, am, l, N):
	for i in range(len(a)):
		r = randint(0,N-1)
		a[i][0], a[i][r] = swap(a[i][0], a[i][r])
		am[i][0], am[i][r] = swap(am[i][0], am[i][r])
		l[i][0], l[i][r] = swap(l[i][0], l[i][r])
	return a, am, l

def validate(val_fn, fold_name, epoch, fold, batch_size, args):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	total = 0
	ones = 0
	N = args.no_of_candidates
	posts, post_masks, ans_list, ans_masks_list, labels = fold
	for p, pm, a, am, l in iterate_minibatches(posts, post_masks, ans_list, ans_masks_list, labels, batch_size, shuffle=True):
		a, am, l = shuffle(a, am, l, N)
		a_list = [None]*N
		am_list = [None]*N
		for i in range(N):
			a_list[i] = a[:,i,:]
			am_list[i] = am[:,i,:]
		preds, loss = val_fn(p, pm, a_list, am_list, l)
		cost += loss
		for pred in preds:
			if np.argmax(pred) == 0:
				corr += 1
			total += 1

		num_batches += 1

	lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
				(fold_name, epoch, cost*1.0 / num_batches, corr*1.0 / total, time.time()-start)
	print lstring

def validate_pred_answers(val_fn, fold_name, epoch, fold):
	start = time.time()
	corr = 0
	total = 0
	posts, post_masks, pred_answers_list = fold
	preds = val_fn(posts, post_masks, pred_answers_list)
	for pred in preds:
		if np.argmax(pred) == 0:
			corr += 1
		total += 1

	lstring = '%s: epoch:%d, acc:%f, time:%d' % \
				(fold_name, epoch, corr*1.0 / total, time.time()-start)
	print lstring

def get_data_masks(content, max_len):
	if len(content) > max_len:
		data = content[:max_len]
		data_mask = np.ones(max_len)
	else:
		data = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
		data_mask = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_mask

def generate_data(posts, ans_list, args):
	data_size = len(posts)
	data_posts = np.zeros((data_size, args.post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((data_size, args.post_max_len), dtype=np.float32)
	data_ans_list = np.zeros((data_size, args.no_of_candidates, args.ans_max_len), dtype=np.int32) 
	data_ans_masks_list = np.zeros((data_size, args.no_of_candidates, args.ans_max_len), dtype=np.float32) 
	labels = np.zeros((data_size, args.no_of_candidates), dtype=np.int32)

	for i in range(data_size):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], args.post_max_len)
		for j in range(args.no_of_candidates):
			data_ans_list[i][j], data_ans_masks_list[i][j] = get_data_masks(ans_list[i][j], args.ans_max_len)
		labels[i][0] = 1

	return data_posts, data_post_masks, data_ans_list, data_ans_masks_list, labels

def main(args):
	post_vectors = p.load(open(args.utility_post_vectors, 'rb'))
	ans_list_vectors = p.load(open(args.utility_ans_list_vectors, 'rb'))
	test_pred_ans_list_vectors = p.load(open(args.pred_ans_list_vectors, 'rb'))
	test_post_vectors = p.load(open(args.pred_ans_post_vectors, 'rb'))
	test_post_mask_vectors = p.load(open(args.pred_ans_post_mask_vectors, 'rb'))
	test_post_mask_vectors = np.asarray(test_post_mask_vectors, dtype=np.float32)
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	N = args.no_of_candidates
	
	print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len

	start = time.time()
	print 'generating data'
	posts, post_masks, ans_list, ans_masks_list, labels = \
					generate_data(post_vectors, ans_list_vectors, args)
	print 'done! Time taken: ', time.time() - start

	t_size = int(len(posts)*0.8)
	train = [posts[:t_size], post_masks[:t_size], ans_list[:t_size], ans_masks_list[:t_size], labels[:t_size]]
	dev = [posts[t_size:], post_masks[t_size:], ans_list[t_size:], ans_masks_list[t_size:], labels[:t_size]]
	test = test_post_vectors, test_post_mask_vectors, test_pred_ans_list_vectors

	print 'Size of training data: ', t_size
	print 'Size of dev data: ', len(posts)-t_size

	start = time.time()
	print 'compiling graph...'
	train_fn, dev_fn, test_fn = build_lstm(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start
	
	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'Train', epoch, train, args.batch_size, args)
		validate(dev_fn, '\t DEV', epoch, dev, args.batch_size, args)
		validate_pred_answers(test_fn, '\t TEST', epoch, test)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--utility_post_vectors", type = str)
	argparser.add_argument("--utility_ans_list_vectors", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 200)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 100)
	argparser.add_argument("--post_max_sents", type = int, default = 10)
	argparser.add_argument("--ans_max_len", type = int, default = 10)
	argparser.add_argument("--pred_ans_list_vectors", type = str)
	argparser.add_argument("--pred_ans_post_vectors", type = str)
	argparser.add_argument("--pred_ans_post_mask_vectors", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
