import sys
import argparse
import theano, lasagne, time
import cPickle as p												
import numpy as np
import theano.tensor as T	 
from collections import OrderedDict, Counter
import math, random
import pdb

DEPTH = 0

def get_data_masks(content, max_len):
	if len(content) > max_len:
		data = content[:max_len]
		data_mask = np.ones(max_len)
	else:
		data = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
		data_mask = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_mask

def generate_data(posts, post_max_len, questions_list, question_max_len):
	data_posts = np.zeros((len(posts), post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((len(posts), post_max_len), dtype=np.int32)

	N = len(questions_list[0])	
	data_questions_list = np.zeros((len(questions_list), N, question_max_len), dtype=np.int32)
	data_question_masks_list = np.zeros((len(questions_list), N, question_max_len), dtype=np.int32)

	for i in range(len(posts)):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], post_max_len)
		for j in range(N):
			try:
				data_questions_list[i][j], data_question_masks_list[i][j] = get_data_masks(questions_list[i][j], question_max_len)
			except:
				pdb.set_trace()

	return data_posts, data_post_masks, data_questions_list, data_question_masks_list

def iterate_minibatches(posts, post_masks, questions, question_masks, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(len(posts))
		np.random.shuffle(indices)
	for start_idx in range(0, len(posts) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], post_masks[excerpt], questions[excerpt], question_masks[excerpt]

def swap(a, b):
	return b, a

def shuffle_questions(question_list, question_mask_list, label_list):
	rand_i = random.randint(0, len(question_list)-1)
	question_list[0], question_list[rand_i] = swap(question_list[0], question_list[rand_i])
	question_mask_list[0], question_mask_list[rand_i] = swap(question_mask_list[0], question_mask_list[rand_i])
	label_list[0], label_list[rand_i] = swap(label_list[0], label_list[rand_i])
	return question_list, question_mask_list, label_list

def validate(val_fn, fold_name, epoch, fold, batch_size):
	start = time.time()
	num_batches = 0.
	cost = 0.
	acc = 0.
	posts, post_masks, questions_list, question_masks_list = fold
	N = len(questions_list[0])
	for p, pm, q, qm in iterate_minibatches(posts, post_masks, questions_list, question_masks_list, \
												batch_size, shuffle=True):
		l = np.zeros([batch_size, N], dtype=np.int32)
		l[:,0] = 1
		for i in range(batch_size):
			q[i], qm[i], l[i] = shuffle_questions(q[i], qm[i], l[i])	
		p = p.astype('int32')
		pm = pm.astype('float32')
		q_list = np.empty([N, q.shape[0], q.shape[2]], dtype=np.int32)
		qm_list = np.empty([N, qm.shape[0], qm.shape[2]], dtype=np.float32)
		for i in range(N):
			q_list[i] = q[:,i].astype('int32')
			qm_list[i] = qm[:,i].astype('float32')
		loss, probs = val_fn(p, pm, q_list, qm_list, l)
		corr = 0
		for i in range(len(probs)):
			if np.argmax(probs[i]) == np.argmax(l[i]):
				corr += 1
		acc += corr*1.0/len(probs)	
		cost += loss*1.0/len(probs)
		num_batches += 1
	lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
				(fold_name, epoch, cost / num_batches, acc / num_batches, time.time()-start)
	print lstring

def build_lstm(word_embeddings, len_voc, word_emb_dim, hidden_dim, N, post_max_len, question_max_len, batch_size, learning_rate, rho, freeze=False):

	# input theano vars
	posts = T.imatrix(name='post')
	post_masks = T.fmatrix(name='post_mask')
	questions_list = T.itensor3(name='question_list')
	question_masks_list = T.ftensor3(name='question_mask_list')
	labels = T.imatrix(name='labels')
 
	# define network
	l_post_in = lasagne.layers.InputLayer(shape=(batch_size, post_max_len), input_var=posts)
	l_post_mask_in = lasagne.layers.InputLayer(shape=(batch_size, post_max_len), input_var=post_masks)
	l_post_emb = lasagne.layers.EmbeddingLayer(l_post_in, len_voc, word_emb_dim, W=word_embeddings) #OR W=lasagne.init.GlorotNormal()
   	l_post_lstm = lasagne.layers.LSTMLayer(l_post_emb, hidden_dim, \
									mask_input=l_post_mask_in, \
									#only_return_final=True, \
									peepholes=False,\
									)

	l_question_lstm = [None]*N
	l_question_in = lasagne.layers.InputLayer(shape=(batch_size, question_max_len), input_var=questions_list[0])
	l_question_mask_in = lasagne.layers.InputLayer(shape=(batch_size, question_max_len), input_var=question_masks_list[0])
	l_question_emb = lasagne.layers.EmbeddingLayer(l_question_in, len_voc, word_emb_dim, W=word_embeddings)
	l_question_lstm[0] = lasagne.layers.LSTMLayer(l_question_emb, hidden_dim, \
									mask_input=l_question_mask_in, \
									#only_return_final=True, \
									peepholes=False,\
									)
	 
	for i in range(1, N):
		l_question_in = lasagne.layers.InputLayer(shape=(batch_size, question_max_len), input_var=questions_list[i])
		l_question_mask_in = lasagne.layers.InputLayer(shape=(batch_size, question_max_len), input_var=question_masks_list[i])
		l_question_emb = lasagne.layers.EmbeddingLayer(l_question_in, len_voc, word_emb_dim, W=word_embeddings)
		l_question_lstm[i] = lasagne.layers.LSTMLayer(l_question_emb, hidden_dim, \
									mask_input=l_question_mask_in, \
									#only_return_final=True, \
									ingate=lasagne.layers.Gate(W_in=l_question_lstm[0].W_in_to_ingate,\
																W_hid=l_question_lstm[0].W_hid_to_ingate,\
																b=l_question_lstm[0].b_ingate,\
																nonlinearity=l_question_lstm[0].nonlinearity_ingate),\
									outgate=lasagne.layers.Gate(W_in=l_question_lstm[0].W_in_to_outgate,\
																W_hid=l_question_lstm[0].W_hid_to_outgate,\
																b=l_question_lstm[0].b_outgate,\
																nonlinearity=l_question_lstm[0].nonlinearity_outgate),\
									forgetgate=lasagne.layers.Gate(W_in=l_question_lstm[0].W_in_to_forgetgate,\
																W_hid=l_question_lstm[0].W_hid_to_forgetgate,\
																b=l_question_lstm[0].b_forgetgate,\
																nonlinearity=l_question_lstm[0].nonlinearity_forgetgate),\
									cell=lasagne.layers.Gate(W_in=l_question_lstm[0].W_in_to_cell,\
																W_hid=l_question_lstm[0].W_hid_to_cell,\
																b=l_question_lstm[0].b_cell,\
																nonlinearity=l_question_lstm[0].nonlinearity_cell),\
									peepholes=False,\
									)

	#for i in range(DEPTH):
	#	l_post_lstm = lasagne.layers.DenseLayer(l_post_lstm, num_units=hidden_dim,\
	#												nonlinearity=lasagne.nonlinearities.rectify)
	#	l_question_lstm[0] = lasagne.layers.DenseLayer(l_question_lstm[0], num_units=hidden_dim,\
	#												nonlinearity=lasagne.nonlinearities.rectify, \
	#												W=l_post_lstm.W)
	#	l_question_2_lstm = lasagne.layers.DenseLayer(l_question_2_lstm, num_units=hidden_dim,\
	#												nonlinearity=lasagne.nonlinearities.rectify,\
	#												W=l_post_lstm.W)

	l_post_lstm = lasagne.layers.DropoutLayer(l_post_lstm)
	for i in range(N):
		l_question_lstm[i] = lasagne.layers.DropoutLayer(l_question_lstm[i])
		
	# residual learning
	#l_post_lstm = lasagne.layers.ElemwiseSumLayer([l_post_lstm, l_post_emb])
	#l_question_lstm[0] = lasagne.layers.ElemwiseSumLayer([l_question_lstm[0], l_question_1_emb])
	#l_question_2_lstm = lasagne.layers.ElemwiseSumLayer([l_question_2_lstm, l_question_2_emb])
	
	# now get aggregate embeddings
	post_out = lasagne.layers.get_output(l_post_lstm)
	question_out = [None]*N
	for i in range(N):
		question_out[i] = lasagne.layers.get_output(l_question_lstm[i])

	post_out = T.mean(post_out * post_masks[:,:,None], axis=1)
	for i in range(N):
   		question_out[i] = T.mean(question_out[i] * question_masks_list[i,:,:,None], axis=1)
	
	# objective computation
	M = theano.shared(np.eye(hidden_dim, dtype=np.float32))

	probs = [None]*N
	for i in range(N):
		probs[i] = T.sum(T.dot(post_out, M)*question_out[i], axis=1)
 
	emb_post_params = lasagne.layers.get_all_params(l_post_emb, trainable=True)
	emb_question_params = lasagne.layers.get_all_params(l_question_emb, trainable=True)

	all_post_params = lasagne.layers.get_all_params(l_post_lstm, trainable=True)
	all_question_params = lasagne.layers.get_all_params(l_question_lstm[0], trainable=True)
	all_params = emb_post_params + emb_question_params + all_post_params + all_question_params

	probs = lasagne.nonlinearities.softmax(T.stack(probs, axis=1))
	loss = T.sum(lasagne.objectives.categorical_crossentropy(probs, labels))
	loss += rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params + [M], learning_rate=learning_rate)
	train_fn = theano.function([posts, post_masks, questions_list, question_masks_list, labels], \
							   [loss, probs], updates=updates)
	val_fn = theano.function([posts, post_masks, questions_list, question_masks_list, labels], \
							   [loss, probs])
	return train_fn, val_fn

def main(args):
	post_vectors = p.load(open(args.post_vectors, 'rb'))
	ques_list_vectors = p.load(open(args.ques_list_vectors, 'rb'))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	
	print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len, ', question_max_len ', args.question_max_len

	start = time.time()
	print 'generating data'
	posts, post_masks, questions_list, question_masks_list = generate_data(post_vectors, args.post_max_len, ques_list_vectors, args.question_max_len)
	print 'done! Time taken: ', time.time()-start
	#N = len(questions_list[0])
	N = args.no_of_candidates
	print 'Candidate question list size: ', N

	t_size = int(len(posts)*0.8)
	train = [posts[:t_size], post_masks[:t_size], questions_list[:t_size,:N], question_masks_list[:t_size,:N]]
	dev = [posts[t_size:], post_masks[t_size:], questions_list[t_size:,:N], question_masks_list[t_size:,:N]]

	print 'Size of training data: ', t_size
	print 'Size of dev data: ', len(posts)-t_size

	start = time.time()
	print 'compiling graph...'
	train_fn, val_fn = build_lstm(word_embeddings, vocab_size, word_emb_dim, args.hidden_dim, N, args.post_max_len, args.question_max_len, args.batch_size, args.learning_rate, args.rho, freeze=freeze)
	print 'done! Time taken: ', time.time()-start
	
	# train network
	for epoch in range(args.no_of_epochs):

		validate(train_fn, 'Train', epoch, train, args.batch_size)
		validate(val_fn, '\t DEV', epoch, dev, args.batch_size)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 200)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 100)
	argparser.add_argument("--question_max_len", type = int, default = 20)
	argparser.add_argument("--no_of_candidates", type = int, default = 5)
	args = argparser.parse_args()
	print args
	print ""
	main(args)

