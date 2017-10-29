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

def generate_data(posts, ques_list, ans_list, post_ids, args):
	N = args.no_of_candidates	
	data_size = len(posts)
	data_posts = np.zeros((data_size*N, args.post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((data_size*N, args.post_max_len), dtype=np.float32)
	
	data_ques = np.zeros((data_size*N, args.ques_max_len), dtype=np.int32)
	data_ques_masks = np.zeros((data_size*N, args.ques_max_len), dtype=np.float32)

	data_ans = np.zeros((data_size*N, args.ans_max_len), dtype=np.int32)
	data_ans_masks = np.zeros((data_size*N, args.ans_max_len), dtype=np.float32)

	data_labels = np.zeros((data_size*N), dtype=np.int32)
	data_post_ids = np.zeros((data_size*N), dtype=np.int32)

	for i in range(data_size):
		for j in range(N):
			data_posts[i*N+j], data_post_masks[i*N+j] = get_data_masks(posts[i], args.post_max_len)
			data_ques[i*N+j], data_ques_masks[i*N+j] = get_data_masks(ques_list[i][j][0], args.ques_max_len)
			data_ans[i*N+j], data_ans_masks[i*N+j] = get_data_masks(ans_list[i][j][0], args.ans_max_len)
			if j == 0:
				data_labels[i*N+j] = 1
			else:
				data_labels[i*N+j] = 0
			data_post_ids[i*N+j] = post_ids[i]
	return data_posts, data_post_masks, data_ques, data_ques_masks, data_ans, data_ans_masks, data_labels, data_post_ids	

def build_lstm(data, data_masks, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):

	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=data)
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=data_masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	#l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out = lasagne.layers.get_output(l_lstm)
	# out = lasagne.layers.get_output(l_emb)
	out = T.mean(out * data_masks[:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	# params = lasagne.layers.get_all_params(l_emb, trainable=True)
	# print 'Params in post_lstm: ', lasagne.layers.count_params(l_emb)
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

def build_baseline(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ques = T.imatrix()
	ques_masks = T.fmatrix()
	ans = T.imatrix()
	ans_masks = T.fmatrix()
	labels = T.ivector()

	post_out, post_lstm_params = build_lstm(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ques_out, ques_lstm_params = build_lstm(ques, ques_masks, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	ans_out, ans_lstm_params = build_lstm(ans, ans_masks, args.ans_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	
	post_ques_ans = T.concatenate([post_out, ques_out, ans_out], axis=1)
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
	pqa_preds = lasagne.layers.get_output(l_post_ques_ans_dense)
	loss = T.sum(lasagne.objectives.binary_crossentropy(pqa_preds, labels))
		
	post_ques_ans_dense_params = lasagne.layers.get_all_params(l_post_ques_ans_dense, trainable=True)

	all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + post_ques_ans_dense_params
	print 'Params in concat ', lasagne.layers.count_params(l_post_ques_ans_dense)
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([posts, post_masks, ques, ques_masks, ans, ans_masks, labels], \
									[loss, pqa_preds], updates=updates)
	test_fn = theano.function([posts, post_masks, ques, ques_masks, ans, ans_masks, labels], \
									[loss, pqa_preds],)
	return train_fn, test_fn

def build_utility(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ans = T.imatrix()
	ans_masks = T.fmatrix()
	labels = T.ivector()

	post_out, post_lstm_params = build_lstm(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ans_out, ans_lstm_params = build_lstm(ans, ans_masks, args.ans_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	
	post_ans = T.concatenate([post_out, ans_out], axis=1)
	l_post_ans_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ans)
	for k in range(DEPTH):
		if k == 0:
			l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_dense, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_dense, num_units=1,\
												nonlinearity=lasagne.nonlinearities.sigmoid)
	pa_preds = lasagne.layers.get_output(l_post_ans_dense)
	loss = T.sum(lasagne.objectives.binary_crossentropy(pa_preds, labels))

	post_ans_dense_params = lasagne.layers.get_all_params(l_post_ans_dense, trainable=True)

	all_params = post_lstm_params + ans_lstm_params + post_ans_dense_params
	print 'Params in concat ', lasagne.layers.count_params(l_post_ans_dense)
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([posts, post_masks, ans, ans_masks, labels], \
									[loss, pa_preds], updates=updates)
	test_fn = theano.function([posts, post_masks, ans, ans_masks, labels], \
									[loss, pa_preds],)
	return train_fn, test_fn

def build_answer_generator(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):
	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ques = T.imatrix()
	ques_masks = T.fmatrix()
	ans = T.imatrix()
	ans_masks = T.fmatrix()
	labels = T.ivector()

	post_out, post_lstm_params = build_lstm(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ques_out, ques_lstm_params = build_lstm(ques, ques_masks, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	ans_out, ans_lstm_params = build_lstm(ans, ans_masks, args.ans_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	# Prob of a given p,q
	post_ques = T.concatenate([post_out, ques_out], axis=1)
	l_post_ques_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
	for k in range(DEPTH):
		if k == 0:
			l_post_ques_dense = lasagne.layers.DenseLayer(l_post_ques_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ques_dense = lasagne.layers.DenseLayer(l_post_ques_dense, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	
	post_ques_dense_params = lasagne.layers.get_all_params(l_post_ques_dense, trainable=True)		
	print 'Params in post_ques ', lasagne.layers.count_params(l_post_ques_dense)
	
	pq_out = lasagne.layers.get_output(l_post_ques_dense)
	
	ques_squared_errors = [None]*(N*N)
	pq_a_squared_errors = [None]*(N*N)
	for i in range(N):
		for j in range(N):
			ques_squared_errors[i*N+j] = lasagne.objectives.squared_error(ques_out[i], ques_out[j])
			pq_a_squared_errors[i*N+j] = lasagne.objectives.squared_error(pq_out[i], ans_out[j])
	
	pq_a_loss = 0.0	
	for i in range(N):
		pq_a_loss += T.sum(T.dot(labels[:,i], lasagne.objectives.squared_error(pq_out[i], ans_out[i])))
		#pq_a_loss += T.sum(T.dot(T.transpose(lasagne.objectives.squared_error(pq_out[i], ans_out[i])), labels[:,i]) ) # multiple with labels since we want this only for true answer
		#for j in range(N):
			#pq_a_loss += T.sum(pq_a_squared_errors[i*N+j] * (1-lasagne.nonlinearities.tanh(ques_squared_errors[i*N+j])) * labels[:,i])
			#pq_a_loss += T.sum(T.dot(T.transpose(pq_a_squared_errors[i*N+j] * (1-lasagne.nonlinearities.tanh(ques_squared_errors[i*N+j]))), labels[:,i]))
			#pq_a_loss += T.sum(pq_a_squared_errors[i*N+j] * (1-lasagne.nonlinearities.tanh(ques_squared_errors[i*N+j])))
	
	#utility function
	post_ans = T.concatenate([post_out, ans_out], axis=1)
	l_post_ans_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ans)
	for k in range(DEPTH):
		if k == 0:
			l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_dense, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_dense, num_units=1,\
												nonlinearity=lasagne.nonlinearities.sigmoid)
	pa_preds = lasagne.layers.get_output(l_post_ans_dense)
	pa_loss = T.sum(lasagne.objectives.binary_crossentropy(pa_preds, labels))
	post_ans_dense_params = lasagne.layers.get_all_params(l_post_ans_dense, trainable=True)
	print 'Params in post_ans ', lasagne.layers.count_params(l_post_ans_dense)
	all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + post_ques_dense_params + post_ans_dense_params
	
	loss = pq_a_loss + pa_loss	

	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss, pq_a_loss, pa_loss] + pq_out + pq_a_squared_errors + ques_squared_errors + pa_preds, updates=updates)
	test_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss, pq_a_loss, pa_loss] + pq_out + pq_a_squared_errors + ques_squared_errors + pa_preds,)
	return train_fn, test_fn

def iterate_minibatches(posts, post_masks, ques, ques_masks, ans, ans_masks, labels, post_ids, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], post_masks[excerpt], ques[excerpt], ques_masks[excerpt], ans[excerpt], ans_masks[excerpt], labels[excerpt], post_ids[excerpt]


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
	
	indices = range(q.shape[0])
	random.shuffle(indices)
	for j, index in enumerate(indices):
		shuffled_q[j] = q[index]
		shuffled_qm[j] = qm[index]
		shuffled_a[j] = a[index]
		shuffled_am[j] = am[index]
		shuffled_l[j] = l[index]
		shuffled_r[j] = r[index]
			
	return shuffled_q, shuffled_qm, shuffled_a, shuffled_am, shuffled_l, shuffled_r

def shuffle_within_list(q, qm, a, am, l, N):
	r = np.zeros(len(q), dtype=np.int32)
	for i in range(len(q)/N):
		for j in range(N):
			r[i*N+j] = j
		q[i*N:i*N+N], qm[i*N:i*N+N], a[i*N:i*N+N], am[i*N:i*N+N], l[i*N:i*N+N], r[i*N:i*N+N]\
		= shuffle(q[i*N:i*N+N], qm[i*N:i*N+N], a[i*N:i*N+N], am[i*N:i*N+N], l[i*N:i*N+N], r[i*N:i*N+N])
	return q, qm, a, am, l, r 

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

def validate(val_fn, utility_val_fn, pq_val_fn, answer_generator_val_fn, \
								fold_name, epoch, fold, args, out_file=None):
	start = time.time()
	num_batches = 0
	cost = 0
	pq_a_cost = 0
	utility_cost = 0
	pq_cost = 0
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
	posts, post_masks, ques, ques_masks, ans, ans_masks, labels, post_ids = fold
	for p, pm, q, qm, a, am, l, ids in iterate_minibatches(posts, post_masks, ques, ques_masks, ans, ans_masks, labels,\
														 post_ids, args.batch_size, shuffle=False):
		q, qm, a, am, l, r = shuffle_within_list(q, qm, a, am, l, N)
		if args.model == 'baseline_pqa':
			out = val_fn(p, pm, q, qm, a, am, l)
			loss = out[0]
			probs = out[1]
			cost += loss
		elif args.model == 'baseline_pa':
			utility_out = utility_val_fn(p, pm, a, am, l)
			loss = utility_out[0]
			utility_preds = utility_out[1]
			cost += loss
		elif args.model == 'baseline_pq':
			pq_out = pq_val_fn(p, pm, q, qm, l)
			#pq_loss = pq_out[0]
			loss = pq_out[0]
			pq_preds = pq_out[1:]
			pq_preds = np.transpose(pq_preds, (1, 0, 2))
			pq_preds = pq_preds[:,:,0]
			cost += loss
		elif args.model == 'evpi_sum' or args.model == 'evpi_max':
			out = answer_generator_val_fn(p, pm, q, qm, a, am, l)
			loss = out[0]
			pq_a_loss = out[1]
			pa_loss = out[2]
		
			pq_out = out[3:3+N]
			pq_out = np.array(pq_out)[:,:,0]
			pq_out = np.transpose(pq_out)
		
			pq_a_errors = out[3+N:3+N+N*N]
			pq_a_errors = np.array(pq_a_errors)[:,:,0]
			pq_a_errors = np.transpose(pq_a_errors)
		
			q_errors = out[3+N+N*N: 3+N+N*N+N*N]
			q_errors = np.array(q_errors)[:,:,0]
			q_errors = np.transpose(q_errors)
		
			pa_preds = out[3+N+N*N+N*N:]
			pa_preds = np.array(pa_preds)[:,:,0]
			pa_preds = np.transpose(pa_preds)
			
			cost += loss
			pq_a_cost += pq_a_loss
			utility_cost += pa_loss
		
		for j in range(args.batch_size/N):
			preds = [0.0]*N
			for k in range(N):
				if args.model == 'baseline_pqa':
					preds[k] = probs[j*N+k][0]
				if args.model == 'baseline_pa':
					preds[k] = utility_preds[j*N+k][0]
				if args.model == 'baseline_pq':
					preds[k] = pq_preds[j*N+k][0]
				if args.model == 'evpi_max':
					all_preds = [0.0]*N
					for m in range(N):
						all_preds[m] = math.exp(-_lambda*pq_a_errors[j][k*N+m]) * pa_preds[j][k]
					preds[k] = max(all_preds)
				if args.model == 'evpi_sum':
					for m in range(N):
						preds[k] += math.exp(-_lambda*pq_a_errors[j][k*N+m]) * pa_preds[j][k]
					
			rank = get_rank(preds, l[j*N: j*N+N])
			# if 'TRAIN' in fold_name and epoch == 5:
			# 	pdb.set_trace()
			if rank == 1:
				corr += 1
			mrr += 1.0/rank
			for index in range(N):
				if rank <= index+1:
					recall[index] += 1
			total += 1
			#r = [val for val in range(10)]
			if out_file:
				write_test_predictions(out_file, ids[j], preds, r[j*N: j*N+N])
		num_batches += 1
	
	recall = [round(curr_r*1.0/total, 3) for curr_r in recall]	
	lstring = '%s: epoch:%d, cost:%f, pq_a_cost:%f, utility_cost:%f, pq_cost:%f, acc:%f, mrr:%f,time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, pq_a_cost*1.0/num_batches, utility_cost*1.0/num_batches, pq_cost*1.0/num_batches, \
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
	post_ids = p.load(open(args.post_ids_train, 'rb'))
	post_ids = np.array(post_ids)
	post_vectors = p.load(open(args.post_vectors_train, 'rb'))
	if len(post_vectors) != len(post_ids): #for ubuntu,unix,superuser combined data we don't have all train post_ids
		post_ids = np.zeros(len(post_vectors))

	ques_list_vectors = p.load(open(args.ques_list_vectors_train, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors_train, 'rb'))
	
	# post_ids_test = p.load(open(args.post_ids_test, 'rb'))
	# post_ids_test = np.array(post_ids_test)
	# post_vectors_test = p.load(open(args.post_vectors_test, 'rb'))
	# ques_list_vectors_test = p.load(open(args.ques_list_vectors_test, 'rb'))
	# ans_list_vectors_test = p.load(open(args.ans_list_vectors_test, 'rb'))
	
	out_file = open(args.test_predictions_output, 'w')
	out_file.close()
	
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	N = args.no_of_candidates
	
	print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len

	start = time.time()
	print 'generating data'
	posts, post_masks, ques, ques_masks, ans, ans_masks, labels, post_ids = \
					generate_data(post_vectors, ques_list_vectors, ans_list_vectors, post_ids, args)
	# posts_test, post_masks_test, ques_list_test, ques_masks_list_test, ans_list_test, ans_masks_list_test = \
	# 				generate_data(post_vectors_test, ques_list_vectors_test, ans_list_vectors_test, args)
	print 'done! Time taken: ', time.time() - start

	# posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list = \
	# 				shuffle_data(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list)
	t_size = int(len(posts)*0.8)
	
	train = [posts[:t_size], post_masks[:t_size], ques[:t_size], ques_masks[:t_size], ans[:t_size], ans_masks[:t_size], labels[:t_size], post_ids[:t_size]]
	test = [posts[t_size:], post_masks[t_size:], ques[t_size:], ques_masks[t_size:], ans[t_size:], ans_masks[t_size:], labels[t_size:], post_ids[t_size:]]

	# test = [np.concatenate((posts_test, posts[t_size:])), \
	# 		np.concatenate((post_masks_test, post_masks[t_size:])), \
	# 		np.concatenate((ques_list_test, ques_list[t_size:])), \
	# 		np.concatenate((ques_masks_list_test, ques_masks_list[t_size:])), \
	# 		np.concatenate((ans_list_test, ans_list[t_size:])), \
	# 		np.concatenate((ans_masks_list_test, ans_masks_list[t_size:])), \
	# 		np.concatenate((post_ids_test, post_ids[t_size:]))]

	print 'Size of training data: ', t_size
	print 'Size of test data: ', len(posts)-t_size

	train_fn, test_fn = None, None
	utility_train_fn, utility_test_fn = None, None
	pq_train_fn, pq_test_fn = None, None
	answer_generator_train_fn, answer_generator_test_fn = None, None
	
	if args.model == 'baseline_pqa':
		start = time.time()
		print 'compiling graph...'
		train_fn, test_fn, = build_baseline(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
		print 'done! Time taken: ', time.time()-start
	elif args.model == 'baseline_pa':
		start = time.time()
		print 'compiling utility graph...'
		utility_train_fn, utility_test_fn, = build_utility(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
		print 'done! Time taken: ', time.time()-start
	elif args.model == 'baseline_pq':
		start = time.time()
		print 'compiling pq graph...'
		pq_train_fn, pq_test_fn, = build_utility(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
		print 'done! Time taken: ', time.time()-start
	elif args.model == 'evpi_max' or args.model == 'evpi_sum':
		start = time.time()
		print 'compiling answer_generator graph...'
		answer_generator_train_fn, answer_generator_test_fn = \
								build_answer_generator(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
		print 'done! Time taken: ', time.time()-start
	else:
		print 'ERROR: Specify a model'
		sys.exit(0)

	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, utility_train_fn, pq_train_fn, answer_generator_train_fn, 'TRAIN', epoch, train, args)
		validate(test_fn, utility_test_fn, pq_test_fn, answer_generator_test_fn, '\t TEST', epoch, test, args, args.test_predictions_output)
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
	argparser.add_argument("--batch_size", type = int, default = 256)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 100)
	argparser.add_argument("--post_max_sents", type = int, default = 10)
	argparser.add_argument("--post_max_sent_len", type = int, default = 10)
	argparser.add_argument("--ques_max_len", type = int, default = 20)
	argparser.add_argument("--ans_max_len", type = int, default = 40)
	argparser.add_argument("--test_predictions_output", type = str)
	argparser.add_argument("--model", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
