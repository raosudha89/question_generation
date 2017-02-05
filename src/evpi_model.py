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

def generate_data(post_sents,  ques_list, ans_list, args):
	data_size = len(post_sents)
	data_post_sents = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.int32)
	data_post_sent_masks = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.float32)

	N = args.no_of_candidates	
	data_ques_list = np.zeros((data_size, N, args.ques_max_len), dtype=np.int32)
	data_ques_masks_list = np.zeros((data_size, N, args.ques_max_len), dtype=np.float32)

	data_ans_list = np.zeros((data_size, N, args.ans_max_len), dtype=np.int32)
	data_ans_masks_list = np.zeros((data_size, N, args.ans_max_len), dtype=np.float32)

	data_labels = np.zeros((data_size, N))

	for i in range(data_size):
		for j in range(min(args.post_max_sents, len(post_sents[i]))):
			data_post_sents[i][j], data_post_sent_masks[i][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
		for j in range(N):
			data_ques_list[i][j], data_ques_masks_list[i][j] = get_data_masks(ques_list[i][j], args.ques_max_len)
			data_ans_list[i][j], data_ans_masks_list[i][j] = get_data_masks(ans_list[i][j], args.ans_max_len)
		data_labels[i][0] = 1

	return data_post_sents, data_post_sent_masks, data_ques_list, data_ques_masks_list, data_ans_list, data_ans_masks_list, data_labels

def generate_utility_data(post_sents, labels, args):
	data_size = len(post_sents)
	data_post_sents = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.int32)
	data_post_sent_masks = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.float32)
	
	data_labels = np.array(labels, dtype=np.int32)
	for i in range(data_size):
		for j in range(min(args.post_max_sents, len(post_sents[i]))):
			data_post_sents[i][j], data_post_sent_masks[i][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
			
	return data_post_sents, data_post_sent_masks, data_labels

def build_lstm(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	post_sents = T.itensor3()
	post_sent_masks = T.ftensor3()
	ques_list = T.itensor3()
	ques_masks_list = T.ftensor3()
	ans_list = T.itensor3()
	ans_masks_list = T.ftensor3()
	labels = T.imatrix()

	l_post_in = [None]*args.post_max_sents
	l_post_mask = [None]*args.post_max_sents
	l_post_emb = [None]*args.post_max_sents
	l_post_lstm = [None]*args.post_max_sents
	post_out = [None]*args.post_max_sents
	for i in range(args.post_max_sents):
		l_post_in[i] = lasagne.layers.InputLayer(shape=(None, args.post_max_sent_len), input_var=post_sents[i])
		l_post_mask[i] = lasagne.layers.InputLayer(shape=(None, args.post_max_sent_len), input_var=post_sent_masks[i])
		l_post_emb[i] = lasagne.layers.EmbeddingLayer(l_post_in[i], len_voc, word_emb_dim, W=word_embeddings)

		l_post_lstm[i] = lasagne.layers.LSTMLayer(l_post_emb[i], args.hidden_dim, mask_input=l_post_mask[i], )

		# freeze embeddings
		if freeze:
			l_post_emb[i].params[l_post_emb[i].W].remove('trainable')

		post_out[i] = lasagne.layers.get_output(l_post_lstm[i])
		post_out[i] = T.mean(post_out[i] * post_sent_masks[i][:,:,None], axis=1)
	
	l_ques_in = [None]*N
	l_ques_mask = [None]*N
	l_ques_emb = [None]*N
	l_ques_lstm = [None]*N
	ques_out = [None]*N

	for i in range(N):
		l_ques_in[i] = lasagne.layers.InputLayer(shape=(None, args.ques_max_len), input_var=ques_list[i])
		l_ques_mask[i] = lasagne.layers.InputLayer(shape=(None, args.ques_max_len), input_var=ques_masks_list[i])
		l_ques_emb[i] = lasagne.layers.EmbeddingLayer(l_ques_in[i], len_voc, word_emb_dim, W=word_embeddings)
				
	l_ques_lstm[0] = lasagne.layers.LSTMLayer(l_ques_emb[0], args.hidden_dim, mask_input=l_ques_mask[0], )
	for i in range(1, N):
		l_ques_lstm[i] = lasagne.layers.LSTMLayer(l_ques_emb[i], args.hidden_dim, mask_input=l_ques_mask[i],\
									ingate=lasagne.layers.Gate(W_in=l_ques_lstm[0].W_in_to_ingate,\
																W_hid=l_ques_lstm[0].W_hid_to_ingate,\
																b=l_ques_lstm[0].b_ingate,\
																nonlinearity=l_ques_lstm[0].nonlinearity_ingate),\
									outgate=lasagne.layers.Gate(W_in=l_ques_lstm[0].W_in_to_outgate,\
																W_hid=l_ques_lstm[0].W_hid_to_outgate,\
																b=l_ques_lstm[0].b_outgate,\
																nonlinearity=l_ques_lstm[0].nonlinearity_outgate),\
									forgetgate=lasagne.layers.Gate(W_in=l_ques_lstm[0].W_in_to_forgetgate,\
																W_hid=l_ques_lstm[0].W_hid_to_forgetgate,\
																b=l_ques_lstm[0].b_forgetgate,\
																nonlinearity=l_ques_lstm[0].nonlinearity_forgetgate),\
									cell=lasagne.layers.Gate(W_in=l_ques_lstm[0].W_in_to_cell,\
																W_hid=l_ques_lstm[0].W_hid_to_cell,\
																b=l_ques_lstm[0].b_cell,\
																nonlinearity=l_ques_lstm[0].nonlinearity_cell),\
									peepholes=False,\
									)

	for i in range(N):
		ques_out[i] = lasagne.layers.get_output(l_ques_lstm[i])
		ques_out[i] = T.mean(ques_out[i] * ques_masks_list[i][:,:,None], axis=1)

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

	post_out = T.mean(post_out, axis=0)
	
	pred_ans_out = [None]*N
	pqa_out = [None]*(N*N)
	for i in range(N):
		for j in range(N):
			if i == j:
				pred_ans_out[i] = T.sum(T.stack([post_out, ques_out[i]], axis=2), axis=2)
			pqa_out[i*10+j] = T.sum(T.stack([post_out, ques_out[i], ans_out[j]], axis=2), axis=2)
	
	errors = [None]*N
	for i in range(N):
		errors[i] = lasagne.objectives.squared_error(pred_ans_out[i], ans_out[i])
	#for i in range(1, N):
	#	loss += T.sum(abs(lasagne.objectives.squared_error(pred_answer_out[0], answer_out[i]) - lasagne.objectives.squared_error(question_out[0], question_out[i])))
	loss = T.sum(errors * T.transpose(labels)[:,:,None])

	all_sq_errors = [None]*(N*N)
	for i in range(N):
		for j in range(N):
			all_sq_errors[i*10+j] = T.sum(lasagne.objectives.squared_error(pred_ans_out[i], ans_out[j]), axis=1)

	utility_post_sents = T.itensor3()
	utility_post_sent_masks = T.ftensor3()
	utility_labels = T.ivector()
	
	l_utility_post_in = [None]*args.post_max_sents
	l_utility_post_mask = [None]*args.post_max_sents
	l_utility_post_emb = [None]*args.post_max_sents
	l_utility_post_lstm = [None]*args.post_max_sents
	utility_post_out = [None]*args.post_max_sents
	for i in range(args.post_max_sents):
		l_utility_post_in[i] = lasagne.layers.InputLayer(shape=(None, args.post_max_sent_len), input_var=utility_post_sents[i])
		l_utility_post_mask[i] = lasagne.layers.InputLayer(shape=(None, args.post_max_sent_len), input_var=utility_post_sent_masks[i])
		l_utility_post_emb[i] = lasagne.layers.EmbeddingLayer(l_utility_post_in[i], len_voc, word_emb_dim, W=word_embeddings)

		l_utility_post_lstm[i] = lasagne.layers.LSTMLayer(l_utility_post_emb[i], args.hidden_dim, mask_input=l_utility_post_mask[i], \
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
			l_utility_post_emb[i].params[l_utility_post_emb[i].W].remove('trainable')

		utility_post_out[i] = lasagne.layers.get_output(l_utility_post_lstm[i])
		utility_post_out[i] = T.mean(utility_post_out[i] * utility_post_sent_masks[i][:,:,None], axis=1)
	
	utility_post_out = T.mean(utility_post_out, axis=0)
	
	l_utility_post_out = lasagne.layers.InputLayer(shape=(None, args.hidden_dim), input_var=utility_post_out)
	l_utility_post_dense = lasagne.layers.DenseLayer(l_utility_post_out, num_units=1,\
													nonlinearity=lasagne.nonlinearities.sigmoid)
	utility_preds = lasagne.layers.get_output(l_utility_post_dense)
	utility_loss = T.sum(lasagne.objectives.binary_crossentropy(utility_preds, utility_labels))
	
	l_pqa_out = [None]*(N*N)
	l_pqa_dense = [None]*(N*N)
	utility_pqa = [None]*(N*N)
	for i in range(N):
		for j in range(N):
			l_pqa_out[i*10+j] = lasagne.layers.InputLayer(shape=(None, args.hidden_dim), input_var=pqa_out[i*10+j])
			l_pqa_dense[i*10+j] = lasagne.layers.DenseLayer(l_pqa_out[i*10+j], num_units=1,\
                                                    	nonlinearity=lasagne.nonlinearities.sigmoid,\
														W=l_utility_post_dense.W,\
														b=l_utility_post_dense.b)
			utility_pqa[i*10+j] = lasagne.layers.get_output(l_pqa_dense[i*10+j])

	post_params = lasagne.layers.get_all_params(l_post_lstm[0], trainable=True)
	post_emb_params = lasagne.layers.get_all_params(l_post_emb[0], trainbale=True)
	ques_params = lasagne.layers.get_all_params(l_ques_lstm[0], trainable=True)
	ques_emb_params = lasagne.layers.get_all_params(l_ques_emb[0], trainbale=True)
	ans_params = lasagne.layers.get_all_params(l_ans_lstm[0], trainable=True)
	ans_emb_params = lasagne.layers.get_all_params(l_ans_emb[0], trainable=True)
	utility_post_params = lasagne.layers.get_all_params(l_utility_post_lstm[0], trainable=True)
	utility_post_emb_params = lasagne.layers.get_all_params(l_utility_post_emb[0], trainbale=True)
	utility_dense_params = lasagne.layers.get_all_params(l_utility_post_dense, trainable=True)
	all_params = post_params + post_emb_params + ques_params + ques_emb_params + ans_params + ans_emb_params
	utility_all_params = utility_post_params + utility_post_emb_params + utility_dense_params
	
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)
	utility_loss += args.rho * sum(T.sum(l ** 2) for l in utility_all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	utility_updates = lasagne.updates.adam(utility_loss, utility_all_params, learning_rate=args.learning_rate)

	train_fn = theano.function([post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss] + utility_pqa + all_sq_errors, updates=updates)
	dev_fn = theano.function([post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss] + utility_pqa + all_sq_errors,)
	utility_train_fn = theano.function([utility_post_sents, utility_post_sent_masks, utility_labels], \
									[utility_preds, utility_loss], updates=utility_updates)
	utility_dev_fn = theano.function([utility_post_sents, utility_post_sent_masks, utility_labels], \
									[utility_preds, utility_loss],)

	return train_fn, dev_fn, utility_train_fn, utility_dev_fn

def shuffle(p, pm, q, qm, a, am, l):
	for i in range(len(p)):
		all_data = zip(p[i], pm[i], q[i], qm[i], a[i], am[i], l[i])
		random.shuffle(all_data)
		p[i], pm[i], q[i], qm[i], a[i], am[i], l[i] = zip(*all_data)
	return p, pm, q, qm, a, am, l

def iterate_minibatches(post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(post_sents.shape[0])
		np.random.shuffle(indices)
	data = []
	for start_idx in range(0, post_sents.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		data.append([post_sents[excerpt], post_sent_masks[excerpt], ques_list[excerpt], ques_masks_list[excerpt], ans_list[excerpt], ans_masks_list[excerpt]])
	return data

def utility_iterate_minibatches(post_sents, post_sent_masks, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(post_sents.shape[0])
		np.random.shuffle(indices)
	data = []
	for start_idx in range(0, post_sents.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		data.append([post_sents[excerpt], post_sent_masks[excerpt], labels[excerpt]])
	return data

def get_rank(utilities, labels):
	pdb.set_trace()
	correct = np.where(labels==1)[0][0]
	sort_index_utilities = np.argsort(utilities)
	rank = np.where(sort_index_utilities==correct)[0][0]
	return len(utilities) - rank #since ascending sort and we want descending

def validate(val_fn, utility_val_fn, fold_name, epoch, fold, utility_fold, args):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	mrr = 0
	total = 0
	utility_corr = 0
	utility_total = 0
	utility_cost = 0
	_lambda = 0.5
	N = args.no_of_candidates
	N_post_sents = args.post_max_sents
	post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list = fold
	utility_post_sents, utility_post_sent_masks, utility_labels = utility_fold
	minibatches = iterate_minibatches(post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, args.batch_size, shuffle=True)
	num_batches = len(minibatches)
	utility_batch_size = len(utility_post_sents)/num_batches
	utility_minibatches = utility_iterate_minibatches(utility_post_sents, utility_post_sent_masks, utility_labels, utility_batch_size, shuffle=True)
	for i in range(num_batches):
		p, pm, q, qm, a, am = minibatches[i]
		l = np.zeros((args.batch_size, N), dtype=np.int32)
		l[:,0] = 1
		p, pm, q, qm, a, am, l = shuffle(p, pm, q, qm, a, am, l)
		p = np.transpose(p, (1, 0, 2))
		pm = np.transpose(pm, (1, 0, 2))
		q = np.transpose(q, (1, 0, 2))
		qm = np.transpose(qm, (1, 0, 2))
		a = np.transpose(a, (1, 0, 2))
		am = np.transpose(am, (1, 0, 2))
		out = val_fn(p, pm, q, qm, a, am, l)
		loss = out[0]
		preds = out[1:N*N+1]
		preds = np.transpose(preds, (1, 0, 2))
		errors = out[N*N+1:]
		errors = np.transpose(errors, (1, 0))
		#for j in range(len(preds)):
		#	if np.argmax(preds[j*10+j]) == np.argmax(l[j]):
		#		corr += 1
		#	total += 1
		for j in range(len(preds)):
			utilities = [0.0]*N
			for k in range(N):
				for m in range(N):
					utilities[k] += math.exp(-_lambda * errors[j][k*10+m])*preds[j][k*10+m]
			if np.argmax(utilities) == np.argmax(l[j]):
				corr += 1
			mrr += 1.0/(1+get_rank(utilities, l[j]))
			total += 1
		
		cost += loss
		up, upm, ul = utility_minibatches[i]
		up = np.transpose(up, (1, 0, 2))	
		upm = np.transpose(upm, (1, 0, 2))	
		utility_preds, utility_loss = utility_val_fn(up, upm, ul)
		for j in range(len(utility_preds)):
			if (utility_preds[j] >= 0.5 and ul[j] == 1) or (utility_preds[j] < 0.5 and ul[j] == 0):
				utility_corr += 1
			utility_total += 1
		utility_cost += utility_loss

	lstring = '%s: epoch:%d, cost:%f, utility_cost:%f, acc:%f, mrr:%f, utility_acc:%f time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, utility_cost*1.0/num_batches, \
					corr*1.0/total, mrr*1.0/total, utility_corr*1.0/utility_total, time.time()-start)
	print lstring

def main(args):
	post_sent_vectors = p.load(open(args.post_sent_vectors, 'rb'))
	ques_list_vectors = p.load(open(args.ques_list_vectors, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors, 'rb'))
	utility_post_sent_vectors = p.load(open(args.utility_post_sent_vectors, 'rb'))
	utility_labels = p.load(open(args.utility_labels, 'rb'))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	N = args.no_of_candidates
	
	print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len

	start = time.time()
	print 'generating data'
	post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels = \
					generate_data(post_sent_vectors, ques_list_vectors, ans_list_vectors, args)
	print 'done! Time taken: ', time.time() - start

	start = time.time()
	print 'generating utility data'
	utility_post_sents, utility_post_sent_masks, utility_labels = \
					generate_utility_data(utility_post_sent_vectors, utility_labels, args)
	print 'done! Time taken: ', time.time() - start

	t_size = int(len(post_sents)*0.8)
	train = [post_sents[:t_size], post_sent_masks[:t_size], ques_list[:t_size], ques_masks_list[:t_size], ans_list[:t_size], ans_masks_list[:t_size]]
	dev = [post_sents[t_size:], post_sent_masks[t_size:], ques_list[t_size:], ques_masks_list[t_size:], ans_list[t_size:], ans_masks_list[t_size:]]

	print 'Size of training data: ', t_size
	print 'Size of dev data: ', len(post_sents)-t_size

	utility_t_size = int(len(utility_post_sents)*0.8)
	utility_train = [utility_post_sents[:t_size], utility_post_sent_masks[:t_size], utility_labels[:t_size]]
	utility_dev = [utility_post_sents[t_size:], utility_post_sent_masks[t_size:], utility_labels[t_size:]]

	print 'Size of utility training data: ', utility_t_size
	print 'Size of utility dev data: ', len(utility_post_sents)-utility_t_size

	start = time.time()
	print 'compiling graph...'
	train_fn, dev_fn, utility_train_fn, utility_dev_fn = build_lstm(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start
	
	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, utility_train_fn, 'Train', epoch, train, utility_train, args)
		validate(dev_fn, utility_dev_fn, '\t DEV', epoch, dev, utility_dev, args)
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
	argparser.add_argument("--post_max_sent_len", type = int, default = 20)
	argparser.add_argument("--ques_max_len", type = int, default = 10)
	argparser.add_argument("--ans_max_len", type = int, default = 10)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	