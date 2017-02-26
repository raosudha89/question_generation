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

def generate_data(posts,  ques_list, args):
	data_size = len(posts)
	data_posts = np.zeros((data_size, args.post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((data_size, args.post_max_len), dtype=np.float32)

	N = args.no_of_candidates	
	data_ques_list = np.zeros((data_size, N, args.ques_max_len), dtype=np.int32)
	data_ques_masks_list = np.zeros((data_size, N, args.ques_max_len), dtype=np.float32)

	for i in range(data_size):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], args.post_max_len)
		for j in range(N):
			data_ques_list[i][j], data_ques_masks_list[i][j] = get_data_masks(ques_list[i][j], args.ques_max_len)

	return data_posts, data_post_masks, data_ques_list, data_ques_masks_list

def build_lstm(content_list, content_masks_list, N, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):
	out = [None]*N
	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[0])
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out[0] = lasagne.layers.get_output(l_lstm)
	out[0] = T.mean(out[0] * content_masks_list[0][:,:,None], axis=1)
	for i in range(1, N):
		l_in_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[i])
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=l_emb.W)
		l_lstm_ = lasagne.layers.LSTMLayer(l_emb_, hidden_dim, mask_input=l_mask_,\
									ingate=lasagne.layers.Gate(W_in=l_lstm.W_in_to_ingate,\
																W_hid=l_lstm.W_hid_to_ingate,\
																b=l_lstm.b_ingate,\
																nonlinearity=l_lstm.nonlinearity_ingate),\
									outgate=lasagne.layers.Gate(W_in=l_lstm.W_in_to_outgate,\
																W_hid=l_lstm.W_hid_to_outgate,\
																b=l_lstm.b_outgate,\
																nonlinearity=l_lstm.nonlinearity_outgate),\
									forgetgate=lasagne.layers.Gate(W_in=l_lstm.W_in_to_forgetgate,\
																W_hid=l_lstm.W_hid_to_forgetgate,\
																b=l_lstm.b_forgetgate,\
																nonlinearity=l_lstm.nonlinearity_forgetgate),\
									cell=lasagne.layers.Gate(W_in=l_lstm.W_in_to_cell,\
																W_hid=l_lstm.W_hid_to_cell,\
																b=l_lstm.b_cell,\
																nonlinearity=l_lstm.nonlinearity_cell),\
									peepholes=False,\
									)
		out[i] = lasagne.layers.get_output(l_lstm_)
		out[i] = T.mean(out[i] * content_masks_list[i][:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	return out, params

def build_lstm_posts(posts, post_masks, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):

	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=posts)
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=post_masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out = lasagne.layers.get_output(l_lstm)
	out = T.mean(out * post_masks[:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in post_lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

def build_baseline(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ques_list = T.itensor3()
	ques_masks_list = T.ftensor3()
	labels = T.imatrix()

	post_out, post_lstm_params = build_lstm_posts(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	ques_out, ques_lstm_params = build_lstm(ques_list, ques_masks_list, N, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	
	pq_preds = [None]*N
	post_ques = T.concatenate([post_out, ques_out[0]], axis=1)
	l_post_ques_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
	l_post_ques_dense = lasagne.layers.DenseLayer(l_post_ques_in, num_units=args.hidden_dim,\
													  nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ques_dense2 = lasagne.layers.DenseLayer(l_post_ques_dense, num_units=1,\
													   nonlinearity=lasagne.nonlinearities.sigmoid)
	pq_preds[0] = lasagne.layers.get_output(l_post_ques_dense2)
	loss = T.sum(lasagne.objectives.binary_crossentropy(pq_preds[0], labels[:,0]))
	for i in range(1, N):
			post_ques = T.concatenate([post_out, ques_out[i]], axis=1)
			l_post_ques_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
			l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_in_, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify,\
															W=l_post_ques_dense.W,\
															b=l_post_ques_dense.b)
			l_post_ques_dense2_ = lasagne.layers.DenseLayer(l_post_ques_dense_, num_units=1,\
															nonlinearity=lasagne.nonlinearities.sigmoid,\
															W=l_post_ques_dense2.W,\
															b=l_post_ques_dense2.b)
			pq_preds[i] = lasagne.layers.get_output(l_post_ques_dense2_)
			loss += T.sum(lasagne.objectives.binary_crossentropy(pq_preds[i], labels[:,i]))
	
	post_ques_dense_params2 = lasagne.layers.get_all_params(l_post_ques_dense2, trainable=True)

	all_params = post_lstm_params + ques_lstm_params + post_ques_dense_params2
	
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)

	train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, labels], \
									[loss] + pq_preds, updates=updates)
	dev_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, labels], \
									[loss] + pq_preds,)

	return train_fn, dev_fn

def shuffle(q, qm, l):
	shuffled_q = np.zeros((len(q), len(q[0]), len(q[0][0])), dtype=np.int32)
	shuffled_qm = np.zeros((len(qm), len(qm[0]), len(qm[0][0])), dtype=np.float32)
	shuffled_l = np.zeros((len(l), len(l[0])), dtype=np.int32)
	for i in range(len(q)):
		indexes = range(len(q[i]))
		random.shuffle(indexes)
		for j, index in enumerate(indexes):
			shuffled_q[i][j] = q[i][index]
			shuffled_qm[i][j] = qm[i][index]
			shuffled_l[i][j] = l[i][index]
	return shuffled_q, shuffled_qm, shuffled_l

def iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	data = []
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		data.append([posts[excerpt], post_masks[excerpt], ques_list[excerpt], ques_masks_list[excerpt]])
	return data

def get_rank(preds, labels):
	preds = np.array(preds)
	correct = np.where(labels==1)[0][0]
	sort_index_preds = np.argsort(preds)
	desc_sort_index_preds = sort_index_preds[::-1] #since ascending sort and we want descending
	rank = np.where(desc_sort_index_preds==correct)[0][0]
	return rank+1

def validate(val_fn, fold_name, epoch, fold, args):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	mrr = 0
	total = 0
	_lambda = 0.5
	N = args.no_of_candidates
	posts, post_masks, ques_list, ques_masks_list = fold
	for p, pm, q, qm in iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, \
														 args.batch_size, shuffle=True):
		l = np.zeros((args.batch_size, N), dtype=np.int32)
		l[:,0] = 1
		q, qm, l = shuffle(q, qm, l)
		q = np.transpose(q, (1, 0, 2))
		qm = np.transpose(qm, (1, 0, 2))
		out = val_fn(p, pm, q, qm, l)
		loss = out[0]
		preds = out[1:]
		preds = np.transpose(preds, (1, 0, 2))
		preds = preds[:,:,0]
		for j in range(len(preds)):
			pdb.set_trace()
			rank = get_rank(preds[j], l[j])
			if rank == 1:
				corr += 1
			mrr += 1.0/rank
			total += 1
		cost += loss
		num_batches += 1
		
	lstring = '%s: epoch:%d, cost:%f, acc:%f, mrr:%f,time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, \
					corr*1.0/total, mrr*1.0/total, time.time()-start)
	print lstring

def main(args):
	post_vectors = p.load(open(args.post_vectors, 'rb'))
	ques_list_vectors = p.load(open(args.ques_list_vectors, 'rb'))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	N = args.no_of_candidates
	
	print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len

	start = time.time()
	print 'generating data'
	
	posts, post_masks, ques_list, ques_masks_list = \
					generate_data(post_vectors, ques_list_vectors, args)
	print 'done! Time taken: ', time.time() - start
	
	t_size = int(len(posts)*0.8)
	train = [posts[:t_size], post_masks[:t_size], ques_list[:t_size], ques_masks_list[:t_size]]
	dev = [posts[t_size:], post_masks[t_size:], ques_list[t_size:], ques_masks_list[t_size:]]

	print 'Size of training data: ', t_size
	print 'Size of dev data: ', len(posts)-t_size

	start = time.time()
	print 'compiling graph...'
	train_fn, dev_fn, = build_baseline(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start

	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'Train', epoch, train, args)
		validate(dev_fn, '\t DEV', epoch, dev, args)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 200)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 100)
	argparser.add_argument("--ques_max_len", type = int, default = 10)
	args = argparser.parse_args()
	print args
	print ""
	main(args)