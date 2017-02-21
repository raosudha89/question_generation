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

	for i in range(data_size):
		for j in range(min(args.post_max_sents, len(post_sents[i]))):
			data_post_sents[i][j], data_post_sent_masks[i][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
		for j in range(N):
			data_ques_list[i][j], data_ques_masks_list[i][j] = get_data_masks(ques_list[i][j], args.ques_max_len)
			data_ans_list[i][j], data_ans_masks_list[i][j] = get_data_masks(ans_list[i][j], args.ans_max_len)

	return data_post_sents, data_post_sent_masks, data_ques_list, data_ques_masks_list, data_ans_list, data_ans_masks_list

def generate_utility_data_from_triples(post_sents, ans_list, args):
	data_size = 2*len(post_sents)
	data_post_sents = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.int32)
	data_post_sent_masks = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.float32)
	data_labels = np.zeros(data_size, dtype=np.int32)

	for i in range(data_size/2):
		for j in range(min(args.post_max_sents-1, len(post_sents[i]))):
			data_post_sents[2*i][j], data_post_sent_masks[2*i][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
			data_post_sents[2*i+1][j], data_post_sent_masks[2*i+1][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
		rand_index = random.randint(0,9)
		data_post_sents[2*i][j+1], data_post_sent_masks[2*i][j+1] = get_data_masks(ans_list[i][rand_index], args.post_max_sent_len)
		data_post_sents[2*i+1][j+1], data_post_sent_masks[2*i+1][j+1] = get_data_masks(ans_list[i][0], args.post_max_sent_len)
		data_labels[2*i] = 0
		data_labels[2*i+1] = 1
		
	return np.array(data_post_sents), np.array(data_post_sent_masks), np.array(data_labels)

def generate_utility_data(post_sents, labels, args):
	data_size = len(post_sents)
	data_post_sents = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.int32)
	data_post_sent_masks = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.float32)
	
	data_labels = np.array(labels, dtype=np.int32)
	for i in range(data_size):
		for j in range(min(args.post_max_sents, len(post_sents[i]))):
			data_post_sents[i][j], data_post_sent_masks[i][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
			
	return data_post_sents, data_post_sent_masks, data_labels

def build_lstm(content_list, content_masks_list, N, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc):
	out = [None]*N
	l_in = lasagne.layers.InputLayer(shape=(None, max_len), input_var=content_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(None, max_len), input_var=content_masks_list[0])
	#l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out[0] = lasagne.layers.get_output(l_lstm)
	out[0] = T.mean(out[0] * content_masks_list[0][:,:,None], axis=1)
	for i in range(1, N):
		l_in_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=content_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=content_masks_list[i])
		#l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=word_embeddings)
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
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
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	
	return out, params

def build_utility_lstm(utility_post_list, utility_post_masks_list, post_sents, post_sent_masks, ans_list, ans_masks_list, \
					   N, post_max_sents, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc):
	utility_post_out = [None]*post_max_sents
	l_in = lasagne.layers.InputLayer(shape=(None, max_len), input_var=utility_post_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(None, max_len), input_var=utility_post_masks_list[0])
	#l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	# l_drop = lasagne.layers.DropoutLayer(l_emb, p=0.2)
	# l_lstm = lasagne.layers.LSTMLayer(l_drop, hidden_dim, mask_input=l_mask, )
	utility_post_out[0] = lasagne.layers.get_output(l_lstm)
	utility_post_out[0] = T.mean(utility_post_out[0] * utility_post_masks_list[0][:,:,None], axis=1)
	for i in range(1, post_max_sents):
		l_in_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=utility_post_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=utility_post_masks_list[i])
		#l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=word_embeddings)
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
		l_lstm_ = lasagne.layers.LSTMLayer(l_emb_, hidden_dim, mask_input=l_mask_,\
		# l_drop_ = lasagne.layers.DropoutLayer(l_emb_, p=0.2)
		# l_lstm_ = lasagne.layers.LSTMLayer(l_drop_, hidden_dim, mask_input=l_mask_,\
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
		utility_post_out[i] = lasagne.layers.get_output(l_lstm_)
		utility_post_out[i] = T.mean(utility_post_out[i] * utility_post_masks_list[i][:,:,None], axis=1)

	utility_post_concatenate = T.concatenate(utility_post_out, axis=1)
	
	#l_utility_post_out = lasagne.layers.InputLayer(shape=(args.batch_size, args.hidden_dim), input_var=utility_post_out)
	l_utility_post_out = lasagne.layers.InputLayer(shape=(args.batch_size, (args.post_max_sents)*args.hidden_dim), input_var=utility_post_concatenate)
	l_utility_post_dense = lasagne.layers.DenseLayer(l_utility_post_out, num_units=args.hidden_dim,\
	#l_utility_post_drop = lasagne.layers.DropoutLayer(l_utility_post_out, p=0.2)
	#l_utility_post_dense = lasagne.layers.DenseLayer(l_utility_post_drop, num_units=args.hidden_dim,\
													nonlinearity=lasagne.nonlinearities.rectify)
	l_utility_post_dense2 = lasagne.layers.DenseLayer(l_utility_post_dense, num_units=1,\
													nonlinearity=lasagne.nonlinearities.sigmoid)
	utility_preds = lasagne.layers.get_output(l_utility_post_dense2)
	
	params = lasagne.layers.get_all_params(l_lstm, trainable=True) + \
				lasagne.layers.get_all_params(l_utility_post_dense2, trainable=True)
	
	post_sents_out = [None]*(post_max_sents-1)
	for i in range(post_max_sents-1):
		l_in_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=post_sents[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=post_sent_masks[i])
		#l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=word_embeddings)
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
		l_lstm_ = lasagne.layers.LSTMLayer(l_emb_, hidden_dim, mask_input=l_mask_,\
		#l_drop_ = lasagne.layers.DropoutLayer(l_emb_, p=0.2)
		#l_lstm_ = lasagne.layers.LSTMLayer(l_drop_, hidden_dim, mask_input=l_mask_,\
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
		post_sents_out[i] = lasagne.layers.get_output(l_lstm_)
		post_sents_out[i] = T.mean(post_sents_out[i] * post_sent_masks[i][:,:,None], axis=1)
		
	post_ans_preds = [None]*N	
		
	for i in range(N):
		l_in_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=ans_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=ans_masks_list[i])
		#l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=word_embeddings)
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
		l_lstm_ = lasagne.layers.LSTMLayer(l_emb_, hidden_dim, mask_input=l_mask_,\
		# l_drop_ = lasagne.layers.DropoutLayer(l_emb_, p=0.2)
		# l_lstm_ = lasagne.layers.LSTMLayer(l_drop_, hidden_dim, mask_input=l_mask_,\
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
		ans_out = lasagne.layers.get_output(l_lstm_)
		ans_out = T.mean(ans_out * ans_masks_list[i][:,:,None], axis=1)
		post_ans_concatenate = T.concatenate(post_sents_out + [ans_out], axis=1)
	
		l_post_ans_in = lasagne.layers.InputLayer(shape=(args.batch_size, (args.post_max_sents)*args.hidden_dim), input_var=post_ans_concatenate)
		l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_in, num_units=args.hidden_dim,\
		#l_post_ans_drop = lasagne.layers.DropoutLayer(l_post_ans_in, p=0.2)
		#l_post_ans_dense = lasagne.layers.DenseLayer(l_post_ans_drop, num_units=args.hidden_dim,\
													nonlinearity=lasagne.nonlinearities.rectify,\
													W=l_utility_post_dense.W,\
													b=l_utility_post_dense.b)
		l_post_ans_dense2 = lasagne.layers.DenseLayer(l_post_ans_dense, num_units=1,\
													nonlinearity=lasagne.nonlinearities.sigmoid,\
													W=l_utility_post_dense2.W,\
													b=l_utility_post_dense2.b)
		post_ans_preds[i] = lasagne.layers.get_output(l_post_ans_dense2)
		
	return utility_preds, post_ans_preds, params

def build_evpi_model(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	post_sents = T.itensor3()
	post_sent_masks = T.ftensor3()
	ques_list = T.itensor3()
	ques_masks_list = T.ftensor3()
	ans_list = T.itensor3()
	ans_masks_list = T.ftensor3()
	labels = T.imatrix()
	
	utility_post_sents = T.itensor3()
	utility_post_sent_masks = T.ftensor3()
	utility_labels = T.ivector()

	post_out, post_lstm_params = build_lstm(post_sents, post_sent_masks, args.post_max_sents, args.post_max_sent_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc)	
	ques_out, ques_lstm_params = build_lstm(ques_list, ques_masks_list, N, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc)
	ans_out, ans_lstm_params = build_lstm(ans_list, ans_masks_list, N, args.ans_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc)

	#post_out = T.mean(post_out, axis=0)
	# post_concatenate = T.concatenate(post_out, axis=1)
	# l_post_concatenate = lasagne.layers.InputLayer(shape=(args.batch_size, args.post_max_sents*args.hidden_dim), input_var=post_concatenate)
	# l_post_dense = lasagne.layers.DenseLayer(l_post_concatenate, num_units=args.hidden_dim,\
	# 												nonlinearity=lasagne.nonlinearities.rectify)
	# l_post_dense2 = lasagne.layers.DenseLayer(l_post_dense, num_units=args.hidden_dim,\
	# 												nonlinearity=lasagne.nonlinearities.rectify)
	# post_out = lasagne.layers.get_output(l_post_dense2)
	pq_out = [None]*N
	post_ques = T.concatenate([post_out, ques_out[0]], axis=1)
	l_post_ques_in = lasagne.layers.InputLayer(shape=(None, 2*args.hidden_dim), input_var=post_ques)
	l_post_ques_dense = lasagne.layers.DenseLayer(l_post_ques_in, num_units=args.hidden_dim,\
													nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ques_dense2 = lasagne.layers.DenseLayer(l_post_ques_dense, num_units=1,\
													#nonlinearity=lasagne.nonlinearities.rectify)
													nonlinearity=lasagne.nonlinearities.sigmoid)
	pq_out[0] = lasagne.layers.get_output(l_post_ques_dense2)
	for i in range(1, N):
			post_ques = T.concatenate([post_out, ques_out[i]], axis=1)
			l_post_ques_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
			l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_in_, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify,\
															W=l_post_ques_dense.W,\
															b=l_post_ques_dense.b)
			l_post_ques_dense2_ = lasagne.layers.DenseLayer(l_post_ques_dense_, num_units=1,\
															#nonlinearity=lasagne.nonlinearities.rectify,\
															nonlinearity=lasagne.nonlinearities.sigmoid,\
															W=l_post_ques_dense2.W,\
															b=l_post_ques_dense2.b)
			pq_out[i] = lasagne.layers.get_output(l_post_ques_dense2_)
			
	loss = 0.0
	for i in range(N):
		loss += T.sum(lasagne.objectives.binary_crossentropy(pq_out[i], labels[:,i]))

	# all_sq_errors = [None]*(N*N)
	# loss = 0.0
	# for i in range(N):
	# 	for j in range(N):
	# 		all_sq_errors[i*N+j] = T.sum(lasagne.objectives.squared_error(pq_out[i], ans_out[j]), axis=1)
	# 		loss += T.sum(lasagne.objectives.squared_error(pq_out[i], ans_out[j])*labels[:,i,None])

	utility_preds, utility_post_ans_preds, utility_params = build_utility_lstm(utility_post_sents, utility_post_sent_masks, \
																				post_sents, post_sent_masks, ans_list, ans_masks_list, \
																				N, args.post_max_sents, args.post_max_sent_len, \
																				word_embeddings, word_emb_dim, args.hidden_dim, len_voc)

	utility_loss = T.sum(lasagne.objectives.binary_crossentropy(utility_preds, utility_labels))
	
	for i in range(N):
		loss += T.sum(lasagne.objectives.binary_crossentropy(utility_post_ans_preds[i], labels[:,i]))
		
	post_ques_dense_params2 = lasagne.layers.get_all_params(l_post_ques_dense2, trainable=True)
	
	all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + post_ques_dense_params2
	
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)
	utility_loss += args.rho * sum(T.sum(l ** 2) for l in utility_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	utility_updates = lasagne.updates.adam(utility_loss, utility_params, learning_rate=args.learning_rate)
	#updates = lasagne.updates.adagrad(loss, all_params, learning_rate=args.learning_rate)
	#utility_updates = lasagne.updates.adagrad(utility_loss, utility_all_params, learning_rate=args.learning_rate)

	#train_fn = theano.function([post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
	#								[loss] + utility_post_ans_preds + all_sq_errors, updates=updates)
	#dev_fn = theano.function([post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
	#								[loss] + utility_post_ans_preds + all_sq_errors,)
	train_fn = theano.function([post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss] + pq_out + utility_post_ans_preds, updates=updates)
	dev_fn = theano.function([post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss] + pq_out + utility_post_ans_preds,)
	utility_train_fn = theano.function([utility_post_sents, utility_post_sent_masks, utility_labels], \
									[utility_preds, utility_loss], updates=utility_updates)
	utility_dev_fn = theano.function([utility_post_sents, utility_post_sent_masks, utility_labels], \
									[utility_preds, utility_loss],)

	return train_fn, dev_fn, utility_train_fn, utility_dev_fn

def iterate_minibatches(post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, \
							post_ids, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(post_sents.shape[0])
		np.random.shuffle(indices)
	data = []
	for start_idx in range(0, post_sents.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		data.append([post_sents[excerpt], post_sent_masks[excerpt], ques_list[excerpt], ques_masks_list[excerpt], ans_list[excerpt], ans_masks_list[excerpt], post_ids[excerpt]])
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
	utilities = np.array(utilities)[:,0]
	correct = np.where(labels==1)[0][0]
	sort_index_utilities = np.argsort(utilities)
	desc_sort_index_utilities = sort_index_utilities[::-1] #since ascending sort and we want descending
	rank = np.where(desc_sort_index_utilities==correct)[0][0]
	return rank+1

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

def shuffle(q, qm, a, am, l, r):
	shuffled_q = np.zeros((len(q), len(q[0]), len(q[0][0])), dtype=np.int32)
	shuffled_qm = np.zeros((len(qm), len(qm[0]), len(qm[0][0])), dtype=np.float32)
	shuffled_a = np.zeros((len(a), len(a[0]), len(a[0][0])), dtype=np.int32)
	shuffled_am = np.zeros((len(am), len(am[0]), len(am[0][0])), dtype=np.float32)
	shuffled_l = np.zeros((len(l), len(l[0])), dtype=np.int32)
	shuffled_r = np.zeros((len(r), len(r[0])), dtype=np.int32)
	
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

def validate(val_fn, utility_val_fn, fold_name, epoch, fold, utility_fold, args, out_file=None):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	mrr = 0
	total = 0
	utility_corr = 0
	utility_total = 0
	utility_cost = 0
	utility_ones = 0
	N = args.no_of_candidates
	batch_size = args.batch_size
	recall = [0]*N

	if out_file:
		out_file_o = open(out_file, 'a')
		out_file_o.write("\nEpoch: %d\n" % epoch)
		out_file_o.close()
		
	utility_post_sents, utility_post_sent_masks, utility_labels = utility_fold
	for up, upm, ul in utility_iterate_minibatches(utility_post_sents, utility_post_sent_masks, utility_labels, batch_size, shuffle=True):
		up = np.transpose(up, (1, 0, 2))	
		upm = np.transpose(upm, (1, 0, 2))	
		utility_preds, utility_loss = utility_val_fn(up, upm, ul)
		for j in range(len(utility_preds)):
			if (utility_preds[j] >= 0.5 and ul[j] == 1) or (utility_preds[j] < 0.5 and ul[j] == 0):
				utility_corr += 1
			utility_total += 1
		utility_cost += utility_loss
	
	post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids = fold
	for p, pm, q, qm, a, am, ids in iterate_minibatches(post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, \
											post_ids, batch_size, shuffle=True):
		l = np.zeros((batch_size, N), dtype=np.int32)
		r = np.zeros((batch_size, N), dtype=np.int32)
		l[:,0] = 1
		for j in range(N):
			r[:,j] = j
		q, qm, a, am, l, r = shuffle(q, qm, a, am, l, r)
		p = np.transpose(p, (1, 0, 2))
		pm = np.transpose(pm, (1, 0, 2))
		q = np.transpose(q, (1, 0, 2))
		qm = np.transpose(qm, (1, 0, 2))
		a = np.transpose(a, (1, 0, 2))
		am = np.transpose(am, (1, 0, 2))
		out = val_fn(p, pm, q, qm, a, am, l)
		loss = out[0]
		pq_out = out[1:N+1]
		pq_out = np.transpose(pq_out, (1, 0, 2))
		utility_pa = out[N+1:]
		utility_pa = np.transpose(utility_pa, (1, 0, 2))
		# preds = out[1:N+1]
		# preds = np.transpose(preds, (1, 0, 2))
		# errors = out[N+1:]
		# errors = np.transpose(errors, (1, 0))
		for j in range(batch_size):
			utilities = [0.0]*N
			for k in range(N):
				# marginalized_sum = 0.0
				# for m in range(N):
				# 	marginalized_sum += math.exp(-args._lambda * errors[j][k*N+m]) 
				# for m in range(N):
				# 	utilities[k] += (math.exp(-args._lambda * errors[j][k*N+m]) / marginalized_sum) * preds[j][k]
				#utilities[k] = preds[j][k]
				utilities[k] = pq_out[j][k]*utility_pa[j][k]
			if epoch == 4:
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

	recall = [curr_r*1.0/total for curr_r in recall]
	lstring = '%s: epoch:%d, cost:%f, utility_cost:%f, acc:%f, mrr:%f, utility_acc:%f time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, utility_cost*1.0/num_batches, \
					corr*1.0/total, mrr*1.0/total, utility_corr*1.0/utility_total, time.time()-start)
	print lstring
	print recall

def main(args):
	post_ids = p.load(open(args.post_ids_train, 'rb'))
	post_ids = np.array(post_ids)
	post_sent_vectors = p.load(open(args.post_sent_vectors_train, 'rb'))
	if len(post_sent_vectors) != len(post_ids): #for ubuntu,unix,superuser combined data we don't have all train post_ids
		post_ids = np.zeros(len(post_sent_vectors))

	ques_list_vectors = p.load(open(args.ques_list_vectors_train, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors_train, 'rb'))
	utility_post_sent_vectors = p.load(open(args.utility_post_sent_vectors_train, 'rb'))
	utility_labels = p.load(open(args.utility_labels_train, 'rb'))

	post_ids_test = p.load(open(args.post_ids_test, 'rb'))
	post_ids_test = np.array(post_ids_test)
	post_sent_vectors_test = p.load(open(args.post_sent_vectors_test, 'rb'))
	ques_list_vectors_test = p.load(open(args.ques_list_vectors_test, 'rb'))
	ans_list_vectors_test = p.load(open(args.ans_list_vectors_test, 'rb'))
	utility_post_sent_vectors_test = p.load(open(args.utility_post_sent_vectors_test, 'rb'))
	utility_labels_test = p.load(open(args.utility_labels_test, 'rb'))

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
	
	print 'vocab_size ', vocab_size, 

	start = time.time()
	print 'generating data'
	post_sents, post_sent_masks, ques_list, ques_masks_list, ans_list, ans_masks_list = \
					generate_data(post_sent_vectors, ques_list_vectors, ans_list_vectors, args)
	post_sents_test, post_sent_masks_test, ques_list_test, ques_masks_list_test, ans_list_test, ans_masks_list_test, labels_test= \
					generate_data(post_sent_vectors_test, ques_list_vectors_test, ans_list_vectors_test, args)
	print 'done! Time taken: ', time.time() - start

	train_size = int(len(post_sents)*0.8)
	dev_size = int(len(post_sents)*0.2)/2
	train = [post_sents[:train_size], post_sent_masks[:train_size], ques_list[:train_size], ques_masks_list[:train_size], ans_list[:train_size], ans_masks_list[:train_size], post_ids[:train_size]]

	dev = [post_sents[train_size: train_size+dev_size], \
			post_sent_masks[train_size: train_size+dev_size], \
			ques_list[train_size: train_size+dev_size], \
			ques_masks_list[train_size: train_size+dev_size], \
			ans_list[train_size: train_size+dev_size], \
			ans_masks_list[train_size: train_size+dev_size], \
			post_ids[train_size: train_size+dev_size]]

	test = [np.concatenate((post_sents_test, post_sents[train_size+dev_size:])), \
			np.concatenate((post_sent_masks_test, post_sent_masks[train_size+dev_size:])), \
			np.concatenate((ques_list_test, ques_list[train_size+dev_size:])), \
			np.concatenate((ques_masks_list_test, ques_masks_list[train_size+dev_size:])), \
			np.concatenate((ans_list_test, ans_list[train_size+dev_size:])), \
			np.concatenate((ans_masks_list_test, ans_masks_list[train_size+dev_size:])), \
			np.concatenate((post_ids_test, post_ids[train_size+dev_size:]))]

	print 'Size of training data: ', train_size
	print 'Size of dev data: ', dev_size

	start = time.time()
	print 'generating utility data from triples'
	data_utility_post_sents_triples, data_utility_post_sent_masks_triples, data_utility_labels_triples = \
					generate_utility_data_from_triples(post_sent_vectors, ans_list_vectors, args)
	data_utility_post_sents_triples_test, data_utility_post_sent_masks_triples_test, data_utility_labels_triples_test = \
					generate_utility_data_from_triples(post_sent_vectors_test, ans_list_vectors_test, args)
	print 'done! Time taken: ', time.time() - start

	utility_train_triples_size = int(len(data_utility_post_sents_triples)*0.8)
	utility_dev_triples_size = int(len(data_utility_post_sents_triples)*0.2)/2
	
	start = time.time()
	print 'generating utility data'
	data_utility_post_sents, data_utility_post_sent_masks, data_utility_labels = \
					generate_utility_data(utility_post_sent_vectors, utility_labels, args)
	data_utility_post_sents_test, data_utility_post_sent_masks_test, data_utility_labels_test = \
					generate_utility_data(utility_post_sent_vectors_test, utility_labels_test, args)
	print 'done! Time taken: ', time.time() - start

	utility_train_size = int(len(data_utility_post_sents)*0.8)
	utility_dev_size = int(len(data_utility_post_sents)*0.2)/2

	utility_train = [np.concatenate((data_utility_post_sents_triples[:utility_train_triples_size],
									 data_utility_post_sents[:utility_train_size])), \
					np.concatenate((data_utility_post_sent_masks_triples[:utility_train_triples_size],
									 data_utility_post_sent_masks[:utility_train_size])), \
					np.concatenate((data_utility_labels_triples[:utility_train_triples_size],
									 data_utility_labels[:utility_train_size]))]

	utility_dev = [np.concatenate((data_utility_post_sents_triples[utility_train_triples_size: utility_train_triples_size+utility_dev_triples_size],
								  data_utility_post_sents[utility_train_size: utility_train_size+utility_dev_size])), \
					np.concatenate((data_utility_post_sent_masks_triples[utility_train_triples_size: utility_train_triples_size+utility_dev_triples_size],
								  data_utility_post_sent_masks[utility_train_size: utility_train_size+utility_dev_size])), \
					np.concatenate((data_utility_labels_triples[utility_train_triples_size: utility_train_triples_size+utility_dev_triples_size],
								  data_utility_labels[utility_train_size: utility_train_size+utility_dev_size]))]

	utility_test = [np.concatenate((data_utility_post_sents_triples_test, \
									data_utility_post_sents_triples[utility_train_triples_size+utility_dev_triples_size:],
									data_utility_post_sents_test, \
									data_utility_post_sents[utility_train_size+utility_dev_size:])), \
					np.concatenate((data_utility_post_sent_masks_triples_test, \
									data_utility_post_sent_masks_triples[utility_train_triples_size+utility_dev_triples_size:],\
									data_utility_post_sent_masks_test, \
									data_utility_post_sent_masks[utility_train_size+utility_dev_size:])), \
					np.concatenate((data_utility_labels_triples_test, \
									data_utility_labels_triples[utility_train_triples_size+utility_dev_triples_size:],\
									data_utility_labels_test, \
									data_utility_labels[utility_train_size+utility_dev_size:]))] 

	print 'Size of utility training data: ', utility_train_size
	print 'Size of utility dev data: ', utility_dev_size

	start = time.time()
	print 'compiling graph...'
	train_fn, dev_fn, utility_train_fn, utility_dev_fn = build_evpi_model(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start

	#train_fn, dev_fn, utility_train_fn, utility_dev_fn = None, None, None, None
	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, utility_train_fn, 'Train', epoch, train, utility_train, args)
		validate(dev_fn, utility_dev_fn, '\t DEV', epoch, dev, utility_dev, args, args.dev_predictions_output)
		#validate(dev_fn, utility_dev_fn, '\t TEST', epoch, test, utility_test, args, args.test_predictions_output)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_ids_train", type = str)
	argparser.add_argument("--post_sent_vectors_train", type = str)
	argparser.add_argument("--ques_list_vectors_train", type = str)
	argparser.add_argument("--ans_list_vectors_train", type = str)
	argparser.add_argument("--utility_post_sent_vectors_train", type = str)
	argparser.add_argument("--utility_labels_train", type = str)
	argparser.add_argument("--post_ids_test", type = str)
	argparser.add_argument("--post_sent_vectors_test", type = str)
	argparser.add_argument("--ques_list_vectors_test", type = str)
	argparser.add_argument("--ans_list_vectors_test", type = str)
	argparser.add_argument("--utility_post_sent_vectors_test", type = str)
	argparser.add_argument("--utility_labels_test", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 100)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_sents", type = int, default = 5)
	argparser.add_argument("--post_max_sent_len", type = int, default = 10)
	argparser.add_argument("--ques_max_len", type = int, default = 10)
	argparser.add_argument("--ans_max_len", type = int, default = 10)
	argparser.add_argument("--_lambda", type = float, default = 0.5)
	argparser.add_argument("--test_predictions_output", type = str)
	argparser.add_argument("--dev_predictions_output", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
