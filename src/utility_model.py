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

def generate_data(posts,  ques_list, ans_list, args):
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

	return data_posts, data_post_masks, data_ques_list, data_ques_masks_list, data_ans_list, data_ans_masks_list

def generate_utility_data(posts, labels, args):
	data_size = len(posts)
	data_posts = np.zeros((data_size, args.post_max_len + args.ans_max_len), dtype=np.int32)
	data_post_masks = np.zeros((data_size, args.post_max_len + args.ans_max_len), dtype=np.float32)
	
	data_labels = np.array(labels, dtype=np.int32)
	for i in range(data_size):
		data_posts[i], data_post_masks[i]= get_data_masks(posts[i], args.post_max_len + args.ans_max_len)
			
	return data_posts, data_post_masks, data_labels

def build_lstm(content_list, content_masks_list, N, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc):
	out = [None]*N
	l_in = lasagne.layers.InputLayer(shape=(None, max_len), input_var=content_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(None, max_len), input_var=content_masks_list[0])
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out[0] = lasagne.layers.get_output(l_lstm)
	out[0] = T.mean(out[0] * content_masks_list[0][:,:,None], axis=1)
	for i in range(1, N):
		l_in_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=content_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(None, max_len), input_var=content_masks_list[i])
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
	print 'Params in lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

def build_utility_lstm(utility_posts, utility_post_masks, posts, post_masks, ans_list, ans_masks_list, \
					   N, post_max_len, ans_max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc):
	
	l_in = lasagne.layers.InputLayer(shape=(None, post_max_len + ans_max_len), input_var=utility_posts)
	l_mask = lasagne.layers.InputLayer(shape=(None, post_max_len + ans_max_len), input_var=utility_post_masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	l_lstm_back = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, backwards=True)
	l_out = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])
	out = lasagne.layers.get_output(l_out)
	out = T.mean(out * utility_post_masks[:,:,None], axis=1)
	l_dense_in = lasagne.layers.InputLayer(shape=(None, hidden_dim), input_var=out)
	l_dense = lasagne.layers.DenseLayer(l_dense_in, num_units=hidden_dim,\
	 									nonlinearity=lasagne.nonlinearities.rectify)
	l_dense2 = lasagne.layers.DenseLayer(l_dense, num_units=1,\
										nonlinearity=lasagne.nonlinearities.sigmoid)
	utility_preds = lasagne.layers.get_output(l_dense2)
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_dense2, trainable=True)
	
	post_ans_preds = [None]*N
	for i in range(N):
		ans_posts = T.concatenate([ans_list[i], posts], axis=1)
		ans_post_masks = T.concatenate([ans_masks_list[i], post_masks], axis=1)
		l_in_ = lasagne.layers.InputLayer(shape=(None, post_max_len + ans_max_len), input_var=ans_posts)
		l_mask_ = lasagne.layers.InputLayer(shape=(None, post_max_len + ans_max_len), input_var=ans_post_masks)
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, hidden_dim, W=l_emb.W)
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
									# peepholes=False,\
									)
		l_lstm_back_ = lasagne.layers.LSTMLayer(l_emb_, hidden_dim, mask_input=l_mask_, backwards=True, \
											ingate=lasagne.layers.Gate(W_in=l_lstm_back.W_in_to_ingate,\
																		W_hid=l_lstm_back.W_hid_to_ingate,\
																		b=l_lstm_back.b_ingate,\
																		nonlinearity=l_lstm_back.nonlinearity_ingate),\
											outgate=lasagne.layers.Gate(W_in=l_lstm_back.W_in_to_outgate,\
																		W_hid=l_lstm_back.W_hid_to_outgate,\
																		b=l_lstm_back.b_outgate,\
																		nonlinearity=l_lstm_back.nonlinearity_outgate),\
											forgetgate=lasagne.layers.Gate(W_in=l_lstm_back.W_in_to_forgetgate,\
																		W_hid=l_lstm_back.W_hid_to_forgetgate,\
																		b=l_lstm_back.b_forgetgate,\
																		nonlinearity=l_lstm_back.nonlinearity_forgetgate),\
											cell=lasagne.layers.Gate(W_in=l_lstm_back.W_in_to_cell,\
																		W_hid=l_lstm_back.W_hid_to_cell,\
																		b=l_lstm_back.b_cell,\
																		nonlinearity=l_lstm_back.nonlinearity_cell),\
											# peepholes=False,\
											)
		l_out_ = lasagne.layers.ElemwiseSumLayer([l_lstm_, l_lstm_back_])
		out = lasagne.layers.get_output(l_out_)
		out = T.mean(out * ans_post_masks[:,:,None], axis=1)
		l_dense_in_ = lasagne.layers.InputLayer(shape=(None, hidden_dim), input_var=out)
		l_dense_ = lasagne.layers.DenseLayer(l_dense_in_, num_units=hidden_dim,\
											nonlinearity=lasagne.nonlinearities.rectify)
		l_dense2_ = lasagne.layers.DenseLayer(l_dense_, num_units=1,\
											nonlinearity=lasagne.nonlinearities.sigmoid)
		post_ans_preds[i] = lasagne.layers.get_output(l_dense2_)
	
	return utility_preds, post_ans_preds, params

def build_evpi_model(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ans_list = T.itensor3()
	ans_masks_list = T.ftensor3()
	labels = T.imatrix()
	
	utility_posts = T.imatrix()
	utility_post_masks = T.fmatrix()
	utility_labels = T.ivector()

	utility_preds, utility_post_ans_preds, utility_params = build_utility_lstm(utility_posts, utility_post_masks, \
																				posts, post_masks, ans_list, ans_masks_list, \
																				N, args.post_max_len, args.ans_max_len, \
																				word_embeddings, word_emb_dim, args.hidden_dim, len_voc)

	utility_loss = T.sum(lasagne.objectives.binary_crossentropy(utility_preds, utility_labels))
	utility_loss += T.sum(lasagne.objectives.binary_crossentropy(utility_preds, utility_labels)*2*utility_labels)
	loss = 0.0
	for i in range(N):
		loss += T.sum(lasagne.objectives.binary_crossentropy(utility_post_ans_preds[i], labels[:,i]))

	utility_loss += args.rho * sum(T.sum(l ** 2) for l in utility_params)

	# utility_updates = lasagne.updates.adam(utility_loss+loss, utility_params, learning_rate=args.learning_rate)
	utility_updates = lasagne.updates.adam(utility_loss, utility_params, learning_rate=args.learning_rate)

	utility_train_fn = theano.function([utility_posts, utility_post_masks, utility_labels, posts, post_masks, ans_list, ans_masks_list, labels], \
									 [utility_preds, utility_loss, loss] + utility_post_ans_preds, updates=utility_updates)
	utility_dev_fn = theano.function([utility_posts, utility_post_masks, utility_labels, posts, post_masks, ans_list, ans_masks_list, labels], \
									 [utility_preds, utility_loss, loss] + utility_post_ans_preds,)

	return utility_train_fn, utility_dev_fn

def iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, \
							post_ids, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	data = []
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		data.append([posts[excerpt], post_masks[excerpt], ques_list[excerpt], ques_masks_list[excerpt], ans_list[excerpt], ans_masks_list[excerpt], post_ids[excerpt]])
	return data

def utility_iterate_minibatches(posts, post_masks, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	data = []
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		data.append([posts[excerpt], post_masks[excerpt], labels[excerpt]])
	return data

def get_rank(utilities, labels):
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

def validate(utility_val_fn, fold_name, epoch, fold, utility_fold, args, out_file=None):
	start = time.time()
	cost = 0
	corr = 0
	mrr = 0
	total = 0
	utility_corr = 0
	utility_total = 0
	utility_cost = 0

	N = args.no_of_candidates
	batch_size = args.batch_size
	recall = [0]*N
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids = fold
	utility_posts, utility_post_masks, utility_labels = utility_fold
	
	minibatches = iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, \
										post_ids, batch_size, shuffle=True)
	num_batches = len(minibatches)
	utility_batch_size = len(utility_posts)/num_batches
	print "utility_batch_size", utility_batch_size
	utility_minibatches = utility_iterate_minibatches(utility_posts, utility_post_masks, utility_labels, \
													  utility_batch_size, shuffle=True)

	for i in range(num_batches):
		p, pm, q, qm, a, am, ids = minibatches[i]
		l = np.zeros((batch_size, N), dtype=np.int32)
		r = np.zeros((batch_size, N), dtype=np.int32)
		l[:,0] = 1
		for j in range(N):
			r[:,j] = j
		q, qm, a, am, l, r = shuffle(q, qm, a, am, l, r)
		q = np.transpose(q, (1, 0, 2))
		qm = np.transpose(qm, (1, 0, 2))
		a = np.transpose(a, (1, 0, 2))
		am = np.transpose(am, (1, 0, 2))
		up, upm, ul = utility_minibatches[i]
		out = utility_val_fn(up, upm, ul, p, pm, a, am, l)
		utility_preds = out[0]
		utility_loss = out[1]
		loss = out[2]
		preds = out[3:]
		for j in range(len(utility_preds)):
			if 'Train' in fold_name and epoch == 8:
				pdb.set_trace()	
			if (utility_preds[j] >= 0.5 and ul[j] == 1) or (utility_preds[j] < 0.5 and ul[j] == 0):
				utility_corr += 1
			utility_total += 1
		utility_cost += utility_loss
		preds = np.array(preds)[:,:,0]
		preds = np.transpose(preds)
		for j in range(len(preds)):
			# if 'Train' in fold_name and epoch == 15:
			#	pdb.set_trace()	
			rank = get_rank(preds[j], l[j])
			if rank == 1:
				corr += 1
			mrr += 1.0/rank
			for index in range(N):
				if rank <= index+1:
					recall[index] += 1
			total += 1
		cost += loss

	recall = [round(curr_r*1.0/total, 3) for curr_r in recall]
	lstring = '%s: epoch:%d, cost:%f, utility_cost:%f, acc:%f, mrr:%f, utility_acc:%f time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, utility_cost*1.0/num_batches, \
					corr*1.0/total, mrr*1.0/total, utility_corr*1.0/utility_total, time.time()-start)
	print lstring
	print recall

def main(args):
	post_ids = p.load(open(args.post_ids_train, 'rb'))
	post_ids = np.array(post_ids)
	post_vectors = p.load(open(args.post_vectors_train, 'rb'))
	if len(post_vectors) != len(post_ids): #for ubuntu,unix,superuser combined data we don't have all train post_ids
		post_ids = np.zeros(len(post_vectors))

	ques_list_vectors = p.load(open(args.ques_list_vectors_train, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors_train, 'rb'))
	utility_post_vectors = p.load(open(args.utility_post_vectors_train, 'rb'))
	utility_labels = p.load(open(args.utility_labels_train, 'rb'))

	post_ids_test = p.load(open(args.post_ids_test, 'rb'))
	post_ids_test = np.array(post_ids_test)
	post_vectors_test = p.load(open(args.post_vectors_test, 'rb'))
	ques_list_vectors_test = p.load(open(args.ques_list_vectors_test, 'rb'))
	ans_list_vectors_test = p.load(open(args.ans_list_vectors_test, 'rb'))
	utility_post_vectors_test = p.load(open(args.utility_post_vectors_test, 'rb'))
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
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list = \
					generate_data(post_vectors, ques_list_vectors, ans_list_vectors, args)
	posts_test, post_masks_test, ques_list_test, ques_masks_list_test, ans_list_test, ans_masks_list_test = \
					generate_data(post_vectors_test, ques_list_vectors_test, ans_list_vectors_test, args)
	print 'done! Time taken: ', time.time() - start

	train_size = int(len(posts)*0.8)
	dev_size = int(len(posts)*0.2)/2
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
	utility_train_fn, utility_dev_fn = build_evpi_model(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start

	#train_fn, dev_fn, utility_train_fn, utility_dev_fn = None, None, None, None
	# train network
	for epoch in range(args.no_of_epochs):
		validate(utility_train_fn, 'Train', epoch, train, utility_train, args)
		validate(utility_dev_fn, '\t DEV', epoch, dev, utility_dev, args, args.dev_predictions_output)
		#validate(dev_fn, utility_dev_fn, '\t TEST', epoch, test, utility_test, args, args.test_predictions_output)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_ids_train", type = str)
	argparser.add_argument("--post_vectors_train", type = str)
	argparser.add_argument("--ques_list_vectors_train", type = str)
	argparser.add_argument("--ans_list_vectors_train", type = str)
	argparser.add_argument("--utility_post_vectors_train", type = str)
	argparser.add_argument("--utility_labels_train", type = str)
	argparser.add_argument("--post_ids_test", type = str)
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
	argparser.add_argument("--post_max_len", type = int, default = 200)
	argparser.add_argument("--ques_max_len", type = int, default = 20)
	argparser.add_argument("--ans_max_len", type = int, default = 20)
	argparser.add_argument("--_lambda", type = float, default = 0.5)
	argparser.add_argument("--test_predictions_output", type = str)
	argparser.add_argument("--dev_predictions_output", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
