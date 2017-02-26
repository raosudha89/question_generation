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

def generate_data(posts, ques_list, ans_list, args):
	data_size = len(posts)
	data_posts = np.zeros((data_size, args.post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((data_size, args.post_max_len), dtype=np.float32)
	
	N = args.no_of_candidates	
	data_ques_list = np.zeros((data_size, N, args.ques_max_len), dtype=np.int32)
	data_ques_masks_list = np.zeros((data_size, N, args.ques_max_len), dtype=np.float32)

	data_ans_list = np.zeros((data_size, N, args.ans_max_len), dtype=np.int32)
	data_ans_masks_list = np.zeros((data_size, N, args.ans_max_len), dtype=np.float32)

	for i in range(data_size):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], args.post_max_len)
		for j in range(N):
			data_ques_list[i][j], data_ques_masks_list[i][j] = get_data_masks(ques_list[i][j], args.ques_max_len)
			data_ans_list[i][j], data_ans_masks_list[i][j] = get_data_masks(ans_list[i][j], args.ans_max_len)
		
		#Random candidates
		# data_ques_list[i][0], data_ques_masks_list[i][0] = get_data_masks(ques_list[i][0], args.ques_max_len)
		# data_ans_list[i][0], data_ans_masks_list[i][0] = get_data_masks(ans_list[i][0], args.ans_max_len)
		# for j in range(1, N):
		# 	rand_index = random.randint(0, data_size-1)
		# 	data_ques_list[i][j], data_ques_masks_list[i][j] = get_data_masks(ques_list[rand_index][j], args.ques_max_len)
		# 	data_ans_list[i][j], data_ans_masks_list[i][j] = get_data_masks(ans_list[rand_index][j], args.ans_max_len)

	return data_posts, data_post_masks, data_ques_list, data_ques_masks_list, data_ans_list, data_ans_masks_list

def build_gru(content_list, content_masks_list, N, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):
	out = [None]*N
	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[0])
	#l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
	

def build_lstm(content_list, content_masks_list, N, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):
	out = [None]*N
	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[0])
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	#l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out[0] = lasagne.layers.get_output(l_lstm)
	# out[0] = lasagne.layers.get_output(l_emb)
	out[0] = T.mean(out[0] * content_masks_list[0][:,:,None], axis=1)
	for i in range(1, N):
		l_in_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[i])
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=l_emb.W)
		#l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, hidden_dim, W=l_emb.W)
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
		# out[i] = lasagne.layers.get_output(l_emb_)
		out[i] = T.mean(out[i] * content_masks_list[i][:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	# params = lasagne.layers.get_all_params(l_emb, trainable=True)
	# print 'Params in lstm: ', lasagne.layers.count_params(l_emb)
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

def build_lstm_posts(posts, post_masks, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):

	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=posts)
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=post_masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	#l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out = lasagne.layers.get_output(l_lstm)
	# out = lasagne.layers.get_output(l_emb)
	out = T.mean(out * post_masks[:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	# params = lasagne.layers.get_all_params(l_emb, trainable=True)
	# print 'Params in post_lstm: ', lasagne.layers.count_params(l_emb)
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in post_lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

def build_baseline(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ques_list = T.itensor3()
	ques_masks_list = T.ftensor3()
	ans_list = T.itensor3()
	ans_masks_list = T.ftensor3()
	labels = T.imatrix()

	post_out, post_lstm_params = build_lstm_posts(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ques_out, ques_lstm_params = build_lstm(ques_list, ques_masks_list, N, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	ans_out, ans_lstm_params = build_lstm(ans_list, ans_masks_list, N, args.ans_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	
	pqa_preds = [None]*N
	post_ques_ans = T.concatenate([post_out, ques_out[0], ans_out[0]], axis=1)
	l_post_ques_ans_in = lasagne.layers.InputLayer(shape=(args.batch_size, 3*args.hidden_dim), input_var=post_ques_ans)
	l_post_ques_ans_denses = [None]*DEPTH
	for k in range(DEPTH):
		if k == 0:
			l_post_ques_ans_denses[k] = lasagne.layers.DenseLayer(l_post_ques_ans_in, num_units=args.hidden_dim,\
																	nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ques_ans_denses[k] = lasagne.layers.DenseLayer(l_post_ques_ans_denses[k-1], num_units=args.hidden_dim,\
																	nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ques_ans_dense = lasagne.layers.DenseLayer(l_post_ques_ans_denses[-1], num_units=1,\
													   nonlinearity=lasagne.nonlinearities.sigmoid)
	pqa_preds[0] = lasagne.layers.get_output(l_post_ques_ans_dense)
	loss = T.sum(lasagne.objectives.binary_crossentropy(pqa_preds[0], labels[:,0]))
	for i in range(1, N):
		post_ques_ans = T.concatenate([post_out, ques_out[i], ans_out[i]], axis=1)
		l_post_ques_ans_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 3*args.hidden_dim), input_var=post_ques_ans)
		for k in range(DEPTH):
			if k == 0:
				l_post_ques_ans_dense_ = lasagne.layers.DenseLayer(l_post_ques_ans_in_, num_units=args.hidden_dim,\
																	nonlinearity=lasagne.nonlinearities.rectify,\
																	W=l_post_ques_ans_denses[k].W,\
																	b=l_post_ques_ans_denses[k].b)
			else:
				l_post_ques_ans_dense_ = lasagne.layers.DenseLayer(l_post_ques_ans_dense_, num_units=args.hidden_dim,\
																	nonlinearity=lasagne.nonlinearities.rectify,\
																	W=l_post_ques_ans_denses[k].W,\
																	b=l_post_ques_ans_denses[k].b)
		l_post_ques_ans_dense_ = lasagne.layers.DenseLayer(l_post_ques_ans_dense_, num_units=1,\
													   nonlinearity=lasagne.nonlinearities.sigmoid)
		pqa_preds[i] = lasagne.layers.get_output(l_post_ques_ans_dense_)
		loss += T.sum(lasagne.objectives.binary_crossentropy(pqa_preds[i], labels[:,i]))
	
	post_ques_ans_dense_params = lasagne.layers.get_all_params(l_post_ques_ans_dense, trainable=True)

	all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + post_ques_ans_dense_params
	print 'Params in concat ', lasagne.layers.count_params(l_post_ques_ans_dense)
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss] + pqa_preds, updates=updates)
	dev_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss] + pqa_preds,)
	return train_fn, dev_fn

def iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	data = []
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		data.append([posts[excerpt], post_masks[excerpt], ques_list[excerpt], ques_masks_list[excerpt], ans_list[excerpt], ans_masks_list[excerpt]])
	return data

def get_rank(preds, labels):
	preds = np.array(preds)
	correct = np.where(labels==1)[0][0]
	sort_index_preds = np.argsort(preds)
	desc_sort_index_preds = sort_index_preds[::-1] #since ascending sort and we want descending
	rank = np.where(desc_sort_index_preds==correct)[0][0]
	return rank+1

def shuffle(q, qm, a, am, l):
	shuffled_q = np.zeros((len(q), len(q[0]), len(q[0][0])), dtype=np.int32)
	shuffled_qm = np.zeros((len(qm), len(qm[0]), len(qm[0][0])), dtype=np.float32)
	shuffled_a = np.zeros((len(a), len(a[0]), len(a[0][0])), dtype=np.int32)
	shuffled_am = np.zeros((len(am), len(am[0]), len(am[0][0])), dtype=np.float32)
	shuffled_l = np.zeros((len(l), len(l[0])), dtype=np.int32)
	
	for i in range(len(q)):
		indexes = range(len(q[i]))
		random.shuffle(indexes)
		for j, index in enumerate(indexes):
			shuffled_q[i][j] = q[i][index]
			shuffled_qm[i][j] = qm[i][index]
			shuffled_a[i][j] = a[i][index]
			shuffled_am[i][j] = am[i][index]
			shuffled_l[i][j] = l[i][index]
			
	return shuffled_q, shuffled_qm, shuffled_a, shuffled_am, shuffled_l

def validate(val_fn, fold_name, epoch, fold, args):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	mrr = 0
	total = 0
	_lambda = 0.5
	N = args.no_of_candidates
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list = fold
	for p, pm, q, qm, a, am in iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list,\
														 args.batch_size, shuffle=True):
		l = np.zeros((args.batch_size, N), dtype=np.int32)
		l[:,0] = 1
		q, qm, a, am, l = shuffle(q, qm, a, am, l)
		#p = np.transpose(p, (1, 0, 2))
		#pm = np.transpose(pm, (1, 0, 2))
		q = np.transpose(q, (1, 0, 2))
		qm = np.transpose(qm, (1, 0, 2))
		a = np.transpose(a, (1, 0, 2))
		am = np.transpose(am, (1, 0, 2))
		out = val_fn(p, pm, q, qm, a, am, l)
		loss = out[0]
		preds = out[1:]
		preds = np.transpose(preds, (1, 0, 2))
		preds = preds[:,:,0]
		for j in range(args.batch_size):
			rank = get_rank(preds[j], l[j])
			# if 'DEV' in fold_name and epoch == 4:
			# 	pdb.set_trace()
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

def shuffle_data(p, pm, q, qm, a, am):
	sp = [None]*len(p)
	spm = [None]*len(pm)
	sq = [None]*len(q)
	sqm = [None]*len(qm)
	sa = [None]*len(a)
	sam = [None]*len(am)
	indexes = range(len(p))
	random.shuffle(indexes)
	for i, index in enumerate(indexes):
		sp[i] = p[index]
		spm[i] = pm[index]
		sq[i] = q[index]
		sqm[i] = qm[index]
		sa[i] = a[index]
		sam[i] = am[index]
	return np.array(sp), np.array(spm), np.array(sq), np.array(sqm), np.array(sa), np.array(sam)
		
def main(args):
	post_vectors = p.load(open(args.post_vectors, 'rb'))
	ques_list_vectors = p.load(open(args.ques_list_vectors, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors, 'rb'))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	
	post_vectors_test = p.load(open(args.post_vectors_test, 'rb'))
	ques_list_vectors_test = p.load(open(args.ques_list_vectors_test, 'rb'))
	ans_list_vectors_test = p.load(open(args.ans_list_vectors_test, 'rb'))
	
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	N = args.no_of_candidates
	
	print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len

	start = time.time()
	print 'generating data'
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list = \
					generate_data(post_vectors, ques_list_vectors, ans_list_vectors, args)
	print 'done! Time taken: ', time.time() - start

	# posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list = \
	# 				shuffle_data(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list)
	t_size = int(len(posts)*0.8)
	
	train = [posts[:t_size], post_masks[:t_size], ques_list[:t_size], ques_masks_list[:t_size], ans_list[:t_size], ans_masks_list[:t_size]]
	# dev = [posts[t_size:], post_masks[t_size:], ques_list[t_size:], ques_masks_list[t_size:], ans_list[t_size:], ans_masks_list[t_size:]]

	dev = [np.concatenate((posts_test, posts[t_size:])), \
			np.concatenate((post_masks_test, post_masks[t_size:])), \
			np.concatenate((ques_list_test, ques_list[t_size:])), \
			np.concatenate((ques_masks_list_test, ques_masks_list[t_size:])), \
			np.concatenate((ans_list_test, ans_list[t_size:])), \
			np.concatenate((ans_masks_list_test, ans_masks_list[t_size:])), \
			np.concatenate((post_ids_test, post_ids[t_size:]))]

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
	argparser.add_argument("--post_sent_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--ans_list_vectors", type = str)
	argparser.add_argument("--post_vectors_test", type = str)
	argparser.add_argument("--ques_list_vectors_test", type = str)
	argparser.add_argument("--ans_list_vectors_test", type = str)
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
	
