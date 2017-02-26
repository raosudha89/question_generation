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
DEPTH=5
#METHOD = 'BASELINE'
#METHOD = 'EVPI'
#METHOD = 'EVPI_2'
METHOD = 'EVPI_DISJOINT'

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
	data_ques_list = np.zeros((data_size, N, N, args.ques_max_len), dtype=np.int32)
	data_ques_masks_list = np.zeros((data_size, N, N, args.ques_max_len), dtype=np.float32)

	data_ans_list = np.zeros((data_size, N, N, args.ans_max_len), dtype=np.int32)
	data_ans_masks_list = np.zeros((data_size, N, N, args.ans_max_len), dtype=np.float32)
	all_zero_answers = 0
	for i in range(data_size):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], args.post_max_len)
		if np.all(np.array(ans_list[i][0][0])==0):
			all_zero_answers += 1
		for j in range(N):
			# zero_indices = []
			# nonzero_index = None
			# for k in range(N):
			# 	if np.all(np.array(ans_list[i][j][k])==0):
			# 		zero_indices.append(k)
			# 	else:
			# 		for m in zero_indices:
			# 			ans_list[i][j][m] = ans_list[i][j][k]
			# 		zero_indices = []
			# 		nonzero_index = k
			# if zero_indices and nonzero_index:
			# 	for m in zero_indices:
			# 		ans_list[i][j][m] = ans_list[i][j][nonzero_index]
			# if zero_indices and not nonzero_index:
			# 	all_zero_answers += 1
			for k in range(N):
				data_ques_list[i][j][k], data_ques_masks_list[i][j][k] = get_data_masks(ques_list[i][j][k], args.ques_max_len)
				data_ans_list[i][j][k], data_ans_masks_list[i][j][k] = get_data_masks(ans_list[i][j][k], args.ans_max_len)
			
		#Random candidates
		# data_ques_list[i][0], data_ques_masks_list[i][0] = get_data_masks(ques_list[i][0], args.ques_max_len)
		# data_ans_list[i][0], data_ans_masks_list[i][0] = get_data_masks(ans_list[i][0], args.ans_max_len)
		# for j in range(1, N):
		# 	rand_index = random.randint(0, data_size-1)
		# 	data_ques_list[i][j], data_ques_masks_list[i][j] = get_data_masks(ques_list[rand_index][j], args.ques_max_len)
		# 	data_ans_list[i][j], data_ans_masks_list[i][j] = get_data_masks(ans_list[rand_index][j], args.ans_max_len)
	print 'No. of all zero answers: ', all_zero_answers
	return data_posts, data_post_masks, data_ques_list, data_ques_masks_list, data_ans_list, data_ans_masks_list

def build_lstm(content_list, content_masks_list, N, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):
	out = [None]*N
	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[:,0,0,:])
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[:,0,0,:])
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out[0] = [None]*N
	out[0][0] = lasagne.layers.get_output(l_lstm)
	out[0][0] = T.mean(out[0][0] * content_masks_list[:,0,0,:][:,:,None], axis=1)
	for i in range(N):
		if i != 0:
			out[i] = [None]*N
		for j in range(N):
			if i == 0 and j == 0:
				continue
			l_in_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[:,i,j,:])
			l_mask_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[:,i,j,:])
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
			out[i][j] = lasagne.layers.get_output(l_lstm_)
			out[i][j] = T.mean(out[i][j] * content_masks_list[:,i,j,:][:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in lstm: ', lasagne.layers.count_params(l_lstm)
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

def get_pqa_preds(post_out, ques_out, ans_out, N, args):
	pqa_preds = [None]*(N*N)
	post_ques_ans = T.concatenate([post_out, ques_out[0][0], ans_out[0][0]], axis=1)
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
	
	for i in range(N):
		for j in range(N):
			if i == 0 and j == 0:
				continue
			post_ques_ans = T.concatenate([post_out, ques_out[i][0], ans_out[i][j]], axis=1)
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
			pqa_preds[i*N+j] = lasagne.layers.get_output(l_post_ques_ans_dense_)
	
	post_ques_ans_dense_params = lasagne.layers.get_all_params(l_post_ques_ans_dense, trainable=True)
	print 'Params in pqa concat ', lasagne.layers.count_params(l_post_ques_ans_dense)
	return pqa_preds, post_ques_ans_dense_params
	
def get_pa_preds(post_out, ans_out, N, args):
	pa_preds = [None]*(N*N)
	post_ans = T.concatenate([post_out, ans_out[0][0]], axis=1)
	l_post_ans_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ans)
	l_post_ans_denses = [None]*DEPTH
	for k in range(DEPTH):
		if k == 0:
			l_post_ans_denses[k] = lasagne.layers.DenseLayer(l_post_ans_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ans_denses[k] = lasagne.layers.DenseLayer(l_post_ans_denses[k-1], num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_denses[-1], num_units=1,\
												nonlinearity=lasagne.nonlinearities.sigmoid)
	pa_preds[0] = lasagne.layers.get_output(l_post_ans_dense)
	
	for i in range(N):
		for j in range(N):
			if i == 0 and j == 0:
				continue
			post_ans = T.concatenate([post_out, ans_out[i][j]], axis=1)
			l_post_ans_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ans)
			for k in range(DEPTH):
				if k == 0:
					l_post_ans_dense_ = lasagne.layers.DenseLayer(l_post_ans_in_, num_units=args.hidden_dim,\
																	nonlinearity=lasagne.nonlinearities.rectify,\
																	W=l_post_ans_denses[k].W,\
																	b=l_post_ans_denses[k].b)
				else:
					l_post_ans_dense_ = lasagne.layers.DenseLayer(l_post_ans_dense_, num_units=args.hidden_dim,\
																	nonlinearity=lasagne.nonlinearities.rectify,\
																	W=l_post_ans_denses[k].W,\
																	b=l_post_ans_denses[k].b)
			l_post_ans_dense_ = lasagne.layers.DenseLayer(l_post_ans_dense_, num_units=1,\
														   nonlinearity=lasagne.nonlinearities.sigmoid)
			pa_preds[i*N+j] = lasagne.layers.get_output(l_post_ans_dense_)
	
	post_ans_dense_params = lasagne.layers.get_all_params(l_post_ans_dense, trainable=True)
	print 'Params in pa concat ', lasagne.layers.count_params(l_post_ans_dense)
	return pa_preds, post_ans_dense_params

def get_pq_out(post_out, ques_out, N, args):
	pq_out = [None]*(N*N)
	post_ques = T.concatenate([post_out, ques_out[0][0]], axis=1)
	l_post_ques_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
	l_post_ques_denses = [None]*DEPTH
	for k in range(DEPTH):
		if k == 0:
			l_post_ques_denses[k] = lasagne.layers.DenseLayer(l_post_ques_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ques_denses[k] = lasagne.layers.DenseLayer(l_post_ques_denses[k-1], num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	pq_out[0] = lasagne.layers.get_output(l_post_ques_denses[-1])
	
	for i in range(N):
		for j in range(N):
			if i == 0 and j == 0:
				continue
			post_ques = T.concatenate([post_out, ques_out[i][j]], axis=1)
			l_post_ques_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
			for k in range(DEPTH):
				if k == 0:
					l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_in, num_units=args.hidden_dim,\
																	nonlinearity=lasagne.nonlinearities.rectify,\
																	W=l_post_ques_denses[k].W,\
																	b=l_post_ques_denses[k].b)
				else:
					l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_dense_, num_units=args.hidden_dim,\
																	nonlinearity=lasagne.nonlinearities.rectify,\
																	W=l_post_ques_denses[k].W,\
																	b=l_post_ques_denses[k].b)
			pq_out[i*N+j] = lasagne.layers.get_output(l_post_ques_dense_)
	
	post_ques_dense_params = lasagne.layers.get_all_params(l_post_ques_denses[-1], trainable=True)
	print 'Params in pq concat ', lasagne.layers.count_params(l_post_ques_denses[-1])
	return pq_out, post_ques_dense_params

def build_baseline(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ques_list = T.itensor4()
	ques_masks_list = T.ftensor4()
	ans_list = T.itensor4()
	ans_masks_list = T.ftensor4()
	labels = T.imatrix()

	post_out, post_lstm_params = build_lstm_posts(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ques_out, ques_lstm_params = build_lstm(ques_list, ques_masks_list, N, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	ans_out, ans_lstm_params = build_lstm(ans_list, ans_masks_list, N, args.ans_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	
	if METHOD == 'BASELINE':
		pqa_preds, post_ques_ans_dense_params = get_pqa_preds(post_out, ques_out, ans_out, N, args)
		loss = 0.0
		for i in range(N):
			loss += T.sum(lasagne.objectives.binary_crossentropy(pqa_preds[i*N+i], labels[:,i]))
		
		squared_errors = [None]*(N*N)
		for i in range(N):
			for j in range(N):
				squared_errors[i*N+j] = lasagne.objectives.squared_error(ans_out[i][0], ans_out[i][j])			
		
		all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + post_ques_ans_dense_params
	
		loss += args.rho * sum(T.sum(l ** 2) for l in all_params)
	
		updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
		
		train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
										[loss] + pqa_preds + squared_errors, updates=updates)
		dev_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
										[loss] + pqa_preds + squared_errors,)
		
	elif METHOD == 'EVPI':	
		pa_preds, post_ans_dense_params = get_pa_preds(post_out, ans_out, N, args)
		pq_out, post_ques_dense_params = get_pq_out(post_out, ques_out, N, args)
		loss = 0.0
		for i in range(N):
			loss += T.sum(lasagne.objectives.binary_crossentropy(pa_preds[i*N+i], labels[:,i]))

		squared_errors = [None]*(N*N)
		for i in range(N):
			for j in range(N):
				squared_errors[i*N+j] = lasagne.objectives.squared_error(pq_out[i][0], ans_out[i][j])
			loss += T.sum(squared_errors[i*N+i])

		all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + \
						post_ans_dense_params + post_ques_dense_params
	
		loss += args.rho * sum(T.sum(l ** 2) for l in all_params)
	
		updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
		
		train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
										[loss] + pa_preds + squared_errors, updates=updates)
		dev_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
										[loss] + pa_preds + squared_errors,)
	elif METHOD == 'EVPI_2':
		pqa_preds, post_ques_ans_dense_params = get_pqa_preds(post_out, ques_out, ans_out, N, args)
		loss = 0.0
		for i in range(N):
			loss += T.sum(lasagne.objectives.binary_crossentropy(pqa_preds[i*N+i], labels[:,i]))
		pa_preds, post_ans_dense_params = get_pa_preds(post_out, ans_out, N, args)
		for i in range(N):
			loss += T.sum(lasagne.objectives.binary_crossentropy(pa_preds[i*N+i], labels[:,i]))
		all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + \
						post_ans_dense_params + post_ques_ans_dense_params
		loss += args.rho * sum(T.sum(l ** 2) for l in all_params)
	
		updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
		
		train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
										[loss] + pa_preds + pqa_preds, updates=updates)
		dev_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
										[loss] + pa_preds + pqa_preds,)
	elif METHOD == 'EVPI_DISJOINT':
		pqa_preds, post_ques_ans_dense_params = get_pqa_preds(post_out, ques_out, ans_out, N, args)
		loss = 0.0
		for i in range(N):
			loss += T.sum(lasagne.objectives.binary_crossentropy(pqa_preds[i*N+i], labels[:,i]))
		all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + post_ques_ans_dense_params
		loss += args.rho * sum(T.sum(l ** 2) for l in all_params)
		updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
		
		utility_loss = 0.0
		pa_preds, post_ans_dense_params = get_pa_preds(post_out, ans_out, N, args)
		for i in range(N):
			utility_loss += T.sum(lasagne.objectives.binary_crossentropy(pa_preds[i*N+i], labels[:,i]))
		utility_params = post_lstm_params + ans_lstm_params + post_ans_dense_params
		utility_loss += args.rho * sum(T.sum(l ** 2) for l in utility_params)
		utility_updates = lasagne.updates.adam(utility_loss, utility_params, learning_rate=args.learning_rate)
		
		train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
										[loss] + pqa_preds, updates=updates)
		dev_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
										[loss] + pqa_preds,)
		
		utility_train_fn = theano.function([posts, post_masks, ans_list, ans_masks_list, labels], \
										[utility_loss] + pa_preds, updates=utility_updates)
		utility_dev_fn = theano.function([posts, post_masks, ans_list, ans_masks_list, labels], \
										[utility_loss] + pa_preds,)

	if METHOD == 'EVPI_DISJOINT':
		return train_fn, dev_fn, utility_train_fn, utility_dev_fn
	else:
		return train_fn, dev_fn

def iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], post_masks[excerpt], ques_list[excerpt], ques_masks_list[excerpt], ans_list[excerpt], ans_masks_list[excerpt], post_ids[excerpt]

def get_rank(preds, labels):
	preds = np.array(preds)
	correct = np.where(labels==1)[0][0]
	sort_index_preds = np.argsort(preds)
	desc_sort_index_preds = sort_index_preds[::-1] #since ascending sort and we want descending
	rank = np.where(desc_sort_index_preds==correct)[0][0]
	return rank+1

def shuffle(q, qm, a, am, l, r):
	shuffled_q = np.zeros(q.shape, dtype=np.int32)
	shuffled_qm = np.zeros(qm.shape, dtype=np.float32)
	shuffled_a = np.zeros(a.shape, dtype=np.int32)
	shuffled_am = np.zeros(am.shape, dtype=np.float32)
	shuffled_l = np.zeros(l.shape, dtype=np.int32)
	shuffled_r = np.zeros(r.shape, dtype=np.int32)
	
	for i in range(len(q)):
		indexes = range(len(q[i]))
		random.shuffle(indexes)
		for j, index in enumerate(indexes):
			shuffled_q[i][j] = q[i][index]
			shuffled_qm[i][j] = qm[i][index]
			shuffled_a[i][j] = a[i][index]
			shuffled_am[i][j] = am[i][index]
			shuffled_l[i][j] = l[i][index]
			shuffled_r[i][j] = r[i][index]
			
	return shuffled_q, shuffled_qm, shuffled_a, shuffled_am, shuffled_l, shuffled_r

def write_test_predictions(out_file, postId, utilities, ranks):
	lstring = "[%s]: " % (postId)
	N = len(utilities)
	scores = [0]*N
	for i in range(N):
		scores[ranks[i]] = utilities[i]
	for i in range(N):
		lstring += "%f " % (scores[i])
	out_file_o = open(out_file, 'a')
	out_file_o.write(lstring + '\n')
	out_file_o.close()

def validate(val_fn, fold_name, epoch, fold, args, out_file=None):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	mrr = 0
	total = 0
	_lambda = 0.5
	N = args.no_of_candidates
	recall = [0]*N

	if out_file:
		out_file_o = open(out_file, 'a')
		out_file_o.write("\nEpoch: %d\n" % epoch)
		out_file_o.close()
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids = fold
	for p, pm, q, qm, a, am, ids in iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list,\
														 post_ids, args.batch_size, shuffle=True):
		l = np.zeros((args.batch_size, N), dtype=np.int32)
		r = np.zeros((args.batch_size, N), dtype=np.int32)
		l[:,0] = 1
		for j in range(N):
			r[:,j] = j
		q, qm, a, am, l, r = shuffle(q, qm, a, am, l, r)
		out = val_fn(p, pm, q, qm, a, am, l)
		loss = out[0]
		preds = out[1:1+N*N]
		preds = np.array(preds)[:,:,0]
		preds = np.transpose(preds)
		errors = out[1+N*N:]
		errors = np.array(errors)[:,:,0]
		errors = np.transpose(errors)
		for j in range(args.batch_size):
			if METHOD == 'BASELINE':
				utilities = [0.0]*N
				# sum_utilities = [0.0]*N
				# weighted_sum_utilities = [0.0]*N
				for k in range(N):
					# marginalized_sum = 0.0
					# for m in range(N):
					# 	marginalized_sum += math.exp(-_lambda * errors[j][k*N+m])
					# for m in range(N):
					# 	sum_utilities[k] += preds[j][k*N+m]
					# 	weighted_sum_utilities[k] += (math.exp(-_lambda * errors[j][k*N+m]) / marginalized_sum) * preds[j][k*N+m]
					utilities[k] =  preds[j][k*N+k]
			elif METHOD == 'EVPI' or METHOD == 'EVPI_2':
				utilities = [0.0]*N
				for k in range(N):
					marginalized_sum = 0.0
					for m in range(N):
						marginalized_sum += math.exp(-_lambda * errors[j][k*N+m])
					for m in range(N):
						utilities[k] += (math.exp(-_lambda * errors[j][k*N+m]) / marginalized_sum) * preds[j][k*N+m]
				
			if 'TRAIN' in fold_name and epoch == 5:
				pdb.set_trace()
			if out_file:
				write_test_predictions(out_file, ids[j], utilities, r[j])
			rank = get_rank(utilities, l[j])
			if rank == 1:
				corr += 1
			mrr += 1.0/rank
			for index in range(N):
				if rank <= index+1:
					recall[index] += 1
			total += 1
		cost += loss
		num_batches += 1
		
	recall = [round(curr_r*1.0/total, 3) for curr_r in recall]
	lstring = '%s: epoch:%d, cost:%f, acc:%f, mrr:%f,time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, \
					corr*1.0/total, mrr*1.0/total, time.time()-start)
	print lstring
	print recall

def validate_with_utility(val_fn, utility_val_fn, fold_name, epoch, fold, args, out_file=None):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	mrr = 0
	total = 0
	_lambda = 0.5
	N = args.no_of_candidates
	recall = [0]*N

	if out_file:
		out_file_o = open(out_file, 'a')
		out_file_o.write("\nEpoch: %d\n" % epoch)
		out_file_o.close()
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids = fold
	for p, pm, q, qm, a, am, ids in iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list,\
														 post_ids, args.batch_size, shuffle=True):
		l = np.zeros((args.batch_size, N), dtype=np.int32)
		r = np.zeros((args.batch_size, N), dtype=np.int32)
		l[:,0] = 1
		for j in range(N):
			r[:,j] = j
		q, qm, a, am, l, r = shuffle(q, qm, a, am, l, r)
		out = utility_val_fn(p, pm, a, am, l)
		loss = out[0]
		preds = out[1:]
		preds = np.array(preds)[:,:,0]
		preds = np.transpose(preds)
		
		out = val_fn(p, pm, q, qm, a, am, l)
		utility_loss = out[0]
		errors = out[1:]
		errors = np.array(errors)[:,:,0]
		errors = np.transpose(errors)
		for j in range(args.batch_size):
			for k in range(N):
				marginalized_sum = 0.0
				for m in range(N):
					marginalized_sum += math.exp(-_lambda * errors[j][k*N+m])
				for m in range(N):
					utilities[k] += (math.exp(-_lambda * errors[j][k*N+m]) / marginalized_sum) * preds[j][k*N+m]
			if 'TRAIN' in fold_name and epoch == 5:
				pdb.set_trace()
			if out_file:
				write_test_predictions(out_file, ids[j], utilities, r[j])
			rank = get_rank(utilities, l[j])
			if rank == 1:
				corr += 1
			mrr += 1.0/rank
			for index in range(N):
				if rank <= index+1:
					recall[index] += 1
			total += 1
		cost += loss
		num_batches += 1
		
	recall = [round(curr_r*1.0/total, 3) for curr_r in recall]
	lstring = '%s: epoch:%d, cost:%f, acc:%f, mrr:%f,time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, \
					corr*1.0/total, mrr*1.0/total, time.time()-start)
	print lstring
	print recall

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
	start = time.time()
	print 'reading data...'
	post_ids = p.load(open(args.post_ids_train, 'rb'))
	post_ids = np.array(post_ids)
	post_vectors = p.load(open(args.post_vectors_train, 'rb'))
	if len(post_vectors) != len(post_ids): #for ubuntu,unix,superuser combined data we don't have all train post_ids
		post_ids = np.zeros(len(post_vectors))

	ques_list_vectors = p.load(open(args.ques_list_vectors_train, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors_train, 'rb'))
	
	post_ids_test = p.load(open(args.post_ids_test, 'rb'))
	post_ids_test = np.array(post_ids_test)
	post_vectors_test = p.load(open(args.post_vectors_test, 'rb'))
	ques_list_vectors_test = p.load(open(args.ques_list_vectors_test, 'rb'))
	ans_list_vectors_test = p.load(open(args.ans_list_vectors_test, 'rb'))
	
	out_file = open(args.test_predictions_output, 'w')
	out_file.close()
	out_file = open(args.dev_predictions_output, 'w')
	out_file.close()

	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	N = args.no_of_candidates	
	print 'done! Time taken: ', time.time()-start
	
	print 'vocab_size ', vocab_size, 

	start = time.time()
	print 'generating data'
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list = \
					generate_data(post_vectors, ques_list_vectors, ans_list_vectors, args)
	posts_test, post_masks_test, ques_list_test, ques_masks_list_test, ans_list_test, ans_masks_list_test = \
					generate_data(post_vectors_test, ques_list_vectors_test, ans_list_vectors_test, args)
	print 'done! Time taken: ', time.time() - start

	train_size = int(len(posts)*0.8)
	#dev_size = int(len(posts)*0.2)/2
	dev_size = 0
	train = [posts[:train_size], post_masks[:train_size], ques_list[:train_size], ques_masks_list[:train_size], ans_list[:train_size], ans_masks_list[:train_size], post_ids[:train_size]]

	dev = [posts[train_size: train_size+dev_size], \
			post_masks[train_size: train_size+dev_size], \
			ques_list[train_size: train_size+dev_size], \
			ques_masks_list[train_size: train_size+dev_size], \
			ans_list[train_size: train_size+dev_size], \
			ans_masks_list[train_size: train_size+dev_size], \
			post_ids[train_size: train_size+dev_size]]

	test = [np.concatenate((posts_test, posts[train_size+dev_size:])), \
			np.concatenate((post_masks_test, post_masks[train_size+dev_size:])), \
			np.concatenate((ques_list_test, ques_list[train_size+dev_size:])), \
			np.concatenate((ques_masks_list_test, ques_masks_list[train_size+dev_size:])), \
			np.concatenate((ans_list_test, ans_list[train_size+dev_size:])), \
			np.concatenate((ans_masks_list_test, ans_masks_list[train_size+dev_size:])), \
			np.concatenate((post_ids_test, post_ids[train_size+dev_size:]))]

	print 'Size of training data: ', train_size
	print 'Size of dev data: ', dev_size
	print 'Size of test data: ', len(test[0])

	start = time.time()
	print 'compiling graph...'
	if METHOD == 'EVPI_DISJOINT':
		train_fn, dev_fn, utility_train_fn, utility_dev_fn = build_baseline(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	else:
		train_fn, dev_fn, = build_baseline(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start
	
	if METHOD == 'EVPI_DISJOINT':
		# train network
		for epoch in range(args.no_of_epochs):
			validate_with_utility(train_fn, utility_train_fn, 'TRAIN', epoch, train, args)
			validate_with_utility(dev_fn, utility_dev_fn, '\t TEST', epoch, test, args, args.test_predictions_output)
			print "\n"		
	else:
		# train network
		for epoch in range(args.no_of_epochs):
			validate(train_fn, 'TRAIN', epoch, train, args)
			# validate(dev_fn, '\t DEV', epoch, dev, args)
			# validate(dev_fn, '\t DEV', epoch, dev, args, args.dev_predictions_output)
			validate(dev_fn, '\t TEST', epoch, test, args, args.test_predictions_output)
			print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_ids_train", type = str)
	argparser.add_argument("--post_vectors_train", type = str)
	argparser.add_argument("--ques_list_vectors_train", type = str)
	argparser.add_argument("--ans_list_vectors_train", type = str)
	argparser.add_argument("--post_ids_test", type = str)
	argparser.add_argument("--post_vectors_test", type = str)
	argparser.add_argument("--ques_list_vectors_test", type = str)
	argparser.add_argument("--ans_list_vectors_test", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 100)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 100)
	argparser.add_argument("--ques_max_len", type = int, default = 10)
	argparser.add_argument("--ans_max_len", type = int, default = 10)
	argparser.add_argument("--_lambda", type = float, default = 0.5)
	argparser.add_argument("--test_predictions_output", type = str)
	argparser.add_argument("--dev_predictions_output", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
