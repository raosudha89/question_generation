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

def iterate_minibatches(post_sents, post_sent_masks, ans_list, ans_masks_list, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(post_sents.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, post_sents.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield post_sents[excerpt], post_sent_masks[excerpt], ans_list[excerpt], ans_masks_list[excerpt], labels[excerpt]

def iterate_minibatches_v2(post_sents, post_sent_masks, ans_list, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(post_sents.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, post_sents.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield post_sents[excerpt], post_sent_masks[excerpt], ans_list[excerpt]

def build_lstm(word_embeddings, len_voc, word_emb_dim, N, args, freeze=False):

	# input theano vars
	post_sents = T.itensor3()
	post_sent_masks = T.ftensor3()
	ans_list = T.itensor3()
	ans_masks_list = T.ftensor3()
	labels = T.imatrix()
	pred_ans_list = T.ftensor3()

	l_post_in = [None]*args.post_max_sents
	l_post_mask = [None]*args.post_max_sents
	l_post_emb = [None]*args.post_max_sents
	l_post_lstm = [None]*args.post_max_sents
	post_out = [None]*args.post_max_sents
	for i in range(args.post_max_sents):
		l_post_in[i] = lasagne.layers.InputLayer(shape=(None, args.post_max_len), input_var=post_sents[i])
		l_post_mask[i] = lasagne.layers.InputLayer(shape=(None, args.post_max_len), input_var=post_sent_masks[i])
		l_post_emb[i] = lasagne.layers.EmbeddingLayer(l_post_in[i], len_voc, word_emb_dim, W=word_embeddings)

		l_post_lstm[i] = lasagne.layers.LSTMLayer(l_post_emb[i], args.hidden_dim, mask_input=l_post_mask[i], )

		# freeze embeddings
		if freeze:
			l_post_emb[i].params[l_post_emb[i].W].remove('trainable')

		post_out[i] = lasagne.layers.get_output(l_post_lstm[i])
		post_out[i] = T.mean(post_out[i] * post_sent_masks[i][:,:,None], axis=1)
	
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

	M = theano.shared(np.eye(args.hidden_dim, dtype=np.float32))
	post_ans_out = [None]*N
	post_pred_ans_out = [None]*N
	for i in range(N):
		post_ans_out[i] = T.sum(T.dot(post_out,M)*ans_out[i], axis=1)	
		post_pred_ans_out[i] = T.sum(T.dot(post_out,M)*pred_ans_list[i], axis=1) 

	ans_probs = lasagne.nonlinearities.softmax(T.transpose(T.stack(post_ans_out)))
	pred_ans_probs = lasagne.nonlinearities.softmax(T.transpose(T.stack(post_pred_ans_out)))

	post_params = lasagne.layers.get_all_params(l_post_lstm[0], trainable=True)
	post_emb_params = lasagne.layers.get_all_params(l_post_emb[0], trainbale=True)
	ans_params = lasagne.layers.get_all_params(l_ans_lstm[0], trainable=True)
	ans_emb_params = lasagne.layers.get_all_params(l_ans_emb[0], trainbale=True)
	all_params = post_params + post_emb_params + ans_params + ans_emb_params + [M]
	
	# objective computation
	loss = T.sum(lasagne.objectives.categorical_crossentropy(ans_probs, labels))
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)

	train_fn = theano.function([post_sents, post_sent_masks, ans_list, ans_masks_list, labels], \
									[ans_probs, loss], updates=updates)
	dev_fn = theano.function([post_sents, post_sent_masks, ans_list, ans_masks_list, labels], \
									[ans_probs, loss],)
	test_fn = theano.function([post_sents, post_sent_masks, pred_ans_list], pred_ans_probs)
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
	N_post_sents = args.post_max_sents
	post_sents, post_sent_masks, ans_list, ans_masks_list, labels = fold
	for p, pm, a, am, l in iterate_minibatches(post_sents, post_sent_masks, ans_list, ans_masks_list, labels, batch_size, shuffle=True):
		p_sents = [None]*N_post_sents
		pm_sents = [None]*N_post_sents
		for i in range(N_post_sents):
			p_sents[i] = p[:,i,:]
			pm_sents[i] = pm[:,i,:]
		a, am, l = shuffle(a, am, l, N)
		a_list = [None]*N
		am_list = [None]*N
		for i in range(N):
			a_list[i] = a[:,i,:]
			am_list[i] = am[:,i,:]
		preds, loss = val_fn(p_sents, pm_sents, a_list, am_list, l)
		cost += loss
		for i, pred in enumerate(preds):
			if np.argmax(pred) == np.argmax(l[i]):
				corr += 1
			total += 1

		num_batches += 1

	lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
				(fold_name, epoch, cost*1.0 / num_batches, corr*1.0 / total, time.time()-start)
	print lstring

def validate_pred_answers(val_fn, fold_name, epoch, fold, batch_size, args):
	start = time.time()
	corr = 0
	total = 0
	N = args.no_of_candidates
	N_post_sents = args.post_max_sents
	post_sents, post_sent_masks, pred_answers_list = fold
	pred_ans_list = [None]*len(post_sents)
	for i in range(len(post_sents)):
		pred_ans_list[i] = pred_answers_list[:,i,:]
	pred_ans_list = np.array(pred_ans_list)
	for p, pm, a in iterate_minibatches_v2(post_sents, post_sent_masks, pred_ans_list, batch_size, shuffle=True):
		p_sents = [None]*N_post_sents
		pm_sents = [None]*N_post_sents
		for i in range(N_post_sents):
			p_sents[i] = p[:,i,:]
			pm_sents[i] = pm[:,i,:]
		a_list = [None]*N
		for i in range(N):
			a_list[i] = a[:,i,:]
		preds = val_fn(p_sents, pm_sents, a_list)
		for i, pred in enumerate(preds):
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

def generate_data(post_sents, ans_list, args):
	data_size = len(post_sents)
	data_post_sents = np.zeros((data_size, args.post_max_sents, args.post_max_len), dtype=np.int32)
	data_post_sent_masks = np.zeros((data_size, args.post_max_sents, args.post_max_len), dtype=np.float32)
	data_ans_list = np.zeros((data_size, args.no_of_candidates, args.ans_max_len), dtype=np.int32) 
	data_ans_masks_list = np.zeros((data_size, args.no_of_candidates, args.ans_max_len), dtype=np.float32) 
	labels = np.zeros((data_size, args.no_of_candidates), dtype=np.int32)

	for i in range(data_size):
		for j in range(min(args.post_max_sents, len(post_sents[i]))):
			data_post_sents[i][j], data_post_sent_masks[i][j] = get_data_masks(post_sents[i][j], args.post_max_len)
		for j in range(args.no_of_candidates):
			data_ans_list[i][j], data_ans_masks_list[i][j] = get_data_masks(ans_list[i][j], args.ans_max_len)
		labels[i][0] = 1

	return data_post_sents, data_post_sent_masks, data_ans_list, data_ans_masks_list, labels

def generate_test_data(post_sents, args):
	data_size = len(post_sents)
	data_post_sents = np.zeros((data_size, args.post_max_sents, args.post_max_len), dtype=np.int32)
	data_post_sent_masks = np.zeros((data_size, args.post_max_sents, args.post_max_len), dtype=np.float32)

	for i in range(data_size):
		for j in range(min(args.post_max_sents, len(post_sents[i]))):
			data_post_sents[i][j], data_post_sent_masks[i][j] = get_data_masks(post_sents[i][j], args.post_max_len)

	return data_post_sents, data_post_sent_masks

def split_into_sents(post_vectors, post_max_sents):
	post_sent_vectors = [None]*len(post_vectors)
	for i, post_vector in enumerate(post_vectors):
		split_size = len(post_vector)/post_max_sents
		post_sent_vectors[i] = np.split(post_vector[:split_size*post_max_sents], split_size)
	return post_sent_vectors

def main(args):
	post_sent_vectors = p.load(open(args.utility_post_sent_vectors, 'rb'))
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
	post_sents, post_sent_masks, ans_list, ans_masks_list, labels = \
					generate_data(post_sent_vectors, ans_list_vectors, args)
	print 'done! Time taken: ', time.time() - start

	test = [None, None, None]
	test_post_sent_vectors = split_into_sents(test_post_vectors, args.post_max_sents)
	test[0], test[1] = generate_test_data(test_post_sent_vectors, args)
	test[2] = test_pred_ans_list_vectors	

	t_size = int(len(post_sents)*0.8)
	train = [post_sents[:t_size], post_sent_masks[:t_size], ans_list[:t_size], ans_masks_list[:t_size], labels[:t_size]]
	dev = [post_sents[t_size:], post_sent_masks[t_size:], ans_list[t_size:], ans_masks_list[t_size:], labels[:t_size]]

	print 'Size of training data: ', t_size
	print 'Size of dev data: ', len(post_sents)-t_size

	start = time.time()
	print 'compiling graph...'
	train_fn, dev_fn, test_fn = build_lstm(word_embeddings, vocab_size, word_emb_dim, N, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start
	
	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'Train', epoch, train, args.batch_size, args)
		validate(dev_fn, '\t DEV', epoch, dev, args.batch_size, args)
		validate_pred_answers(test_fn, '\t TEST', epoch, test, args.batch_size, args)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--utility_post_sent_vectors", type = str)
	argparser.add_argument("--utility_ans_list_vectors", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 200)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 20)
	argparser.add_argument("--post_max_sents", type = int, default = 10)
	argparser.add_argument("--ans_max_len", type = int, default = 10)
	argparser.add_argument("--pred_ans_list_vectors", type = str)
	argparser.add_argument("--pred_ans_post_vectors", type = str)
	argparser.add_argument("--pred_ans_post_mask_vectors", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
