import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import theano, lasagne,time
import numpy as np
import theano.tensor as T
from collections import Counter
import pdb
import time
import cPickle as p
import random
from sklearn.metrics import mean_squared_error

def generate_data_masks(contents, max_len):
	data = np.zeros((len(contents), max_len), dtype=np.int32)
	data_masks = np.zeros((len(contents), max_len), dtype=np.int32)
	for i, content in enumerate(contents):
		if len(content) > max_len:
			data[i] = content[:max_len]
			data_masks[i] = np.ones(max_len)
		else:
			data[i] = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
			data_masks[i] = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_masks 

def iterate_minibatches(posts, post_masks, questions, question_masks, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], post_masks[excerpt], questions[excerpt], question_masks[excerpt], labels[excerpt]

def build_lstm(post_max_len, question_max_len, len_voc, d_word, We, d_hid, freeze=False, lr=0.1, rho=1e-5):
	# input theano vars
	posts = T.imatrix()
	post_masks = T.matrix()
	questions = T.imatrix()
	question_masks = T.matrix()
	labels = T.ivector()

	# define network
	l_post_in = lasagne.layers.InputLayer(shape=(None, post_max_len), input_var=posts)
	l_post_mask = lasagne.layers.InputLayer(shape=(None, post_max_len), input_var=post_masks)
	l_post_emb = lasagne.layers.EmbeddingLayer(l_post_in, len_voc, d_word, W=We)

	# now feed sequences of spans into VAN
	l_post_lstm = lasagne.layers.LSTMLayer(l_post_emb, d_hid, mask_input=l_post_mask, )

	# define network
	l_question_in = lasagne.layers.InputLayer(shape=(None, question_max_len), input_var=questions)
	l_question_mask = lasagne.layers.InputLayer(shape=(None, question_max_len), input_var=question_masks)
	l_question_emb = lasagne.layers.EmbeddingLayer(l_question_in, len_voc, d_word, W=We)

	# now feed sequences of spans into VAN
	l_question_lstm = lasagne.layers.LSTMLayer(l_question_emb, d_hid, mask_input=l_question_mask, )

	# freeze embeddings
	if freeze:
		l_post_emb.params[l_post_emb.W].remove('trainable')
		l_question_emb.params[l_question_emb.W].remove('trainable')

	post_out = lasagne.layers.get_output(l_post_lstm)
	question_out = lasagne.layers.get_output(l_question_lstm)

	post_out = T.mean(post_out * post_masks[:,:,None], axis=1)
	question_out = T.mean(question_out * question_masks[:,:,None], axis=1)

	l_post_question = lasagne.layers.ConcatLayer([l_post_lstm, l_question_lstm], axis=1)
	#l_post_question = lasagne.layers.InputLayer(shape=(None, d_word), input_var=T.concatenate([post_out, question_out], axis=1))
	l_out = lasagne.layers.DenseLayer(l_post_question, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)	

	pred_labels = lasagne.layers.get_output(l_out)
	# objective computation
	loss = T.sum(lasagne.objectives.categorical_crossentropy(pred_labels, labels))
	
	post_lstm_params = lasagne.layers.get_all_params(l_post_lstm, trainable=True)
	question_lstm_params = lasagne.layers.get_all_params(l_question_lstm, trainable=True)
	all_params = post_lstm_params + question_lstm_params

	loss += rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)

	train_fn = theano.function([posts, post_masks, questions, question_masks, labels], loss, updates=updates)
	val_fn = theano.function([posts, post_masks, questions, question_masks], pred_labels)
	return train_fn, val_fn

def validate(name, val_fn, fold):
	corr = 0
	total = 0
	c = Counter()
	posts, post_masks, questions, question_masks, labels = fold
	pred_labels = val_fn(posts, post_masks, questions, question_masks)
	no_of_classes = len(pred_labels[0])
	pred_labels = np.argmax(pred_labels, axis=1)
	true_classes = [0]*no_of_classes
	pred_classes = [0]*no_of_classes
	for i, pred_label in enumerate(pred_labels):
		true_classes[labels[i]] += 1
		if pred_label == labels[i]:
			corr += 1
			pred_classes[pred_label] += 1
		total += 1
		c[pred_label] += 1

	lstring = 'fold:%s, corr:%d, total:%d, acc:%f' %\
		(name, corr, total, float(corr) / float(total))
	print lstring
	lstring = 'Positives: corr:%d, total:%d, acc:%f' %\
		(pred_classes[1], true_classes[1], float(pred_classes[1]) / float(true_classes[1]))
	print lstring
	lstring = 'Negatives: corr:%d, total:%d, acc:%f' %\
		(pred_classes[0], true_classes[0], float(pred_classes[0]) / float(true_classes[0]))
	print lstring
	
if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "usage: python baseline_post_ques.py posts.p questions.p labels.p word_embeddings.p"
		sys.exit(0)
	posts = p.load(open(sys.argv[1], 'rb'))
	questions = p.load(open(sys.argv[2], 'rb'))
	labels = p.load(open(sys.argv[3], 'rb'))
	word_embeddings = p.load(open(sys.argv[4], 'rb'))
	d_word = 200

	len_voc = len(word_embeddings)
	d_hid = 100
	lr = 0.001
	rho = 1e-5
	freeze = True
	batch_size = 200
	n_epochs = 5
	post_max_len = 100
	question_max_len = 20

	start = time.time()
	print 'generating data'
	posts, post_masks = generate_data_masks(posts, post_max_len)
	questions, question_masks = generate_data_masks(questions, question_max_len)
	labels = np.asarray(labels, dtype=np.int32)
	print 'done! Time taken: ', time.time()-start

	start = time.time()
	print 'compiling graph...'
	train_fn, val_fn = build_lstm(post_max_len, question_max_len, len_voc, d_word, np.asarray(word_embeddings), d_hid, freeze, lr, rho)
	print 'done! Time taken: ', time.time()-start

	t_size = int(posts.shape[0]*0.8)
	train = posts[:t_size], post_masks[:t_size], questions[:t_size], question_masks[:t_size], labels[:t_size]
	dev = posts[t_size:], post_masks[t_size:], questions[t_size:], question_masks[t_size:], labels[t_size:]
	print "Size of train data: ", len(train[0])
	print "Size of dev data: ", len(dev[0])

	print 'Training...'
	# train network
	for epoch in range(n_epochs):
		cost = 0.
		start = time.time()
		posts, post_masks, questions, question_masks, labels = train	
		num_batches = 0.
		for p, pm, q, qm, l in iterate_minibatches(posts, post_masks, questions, question_masks, labels, batch_size, shuffle=True):
			loss = train_fn(p, pm, q, qm, l)
			cost += loss
			num_batches += 1

		lstring = 'epoch:%d, cost:%f, time:%d' % \
			(epoch, cost / num_batches, time.time()-start)
		print lstring
		validate('dev', val_fn, dev)


