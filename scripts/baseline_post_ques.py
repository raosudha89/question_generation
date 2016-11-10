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

def preprocess(text):
	return word_tokenize(text.lower())

def get_word_indices(words, vocab, max_len):
	word_indices = np.zeros(max_len, dtype=np.int32)
	UNK = "<unk>"
	for i, w in enumerate(words):
		if i >= max_len:
			break
		try:
			word_indices[i] = vocab[w]
		except:
			word_indices[i] = vocab[UNK]
	return word_indices

def generate_vectors(contents_str, max_len, vocab):
	contents = []
	masks = []
	for content_str in contents_str:
		words = preprocess(content_str)
		indices = get_word_indices(words, vocab, max_len)
		contents.append(indices)
		masks.append(np.concatenate((np.ones(min(len(words), max_len), dtype=np.int32), np.zeros(max(0, max_len-len(words)), dtype=np.int32)))) 
	return [np.asarray(contents, dtype=np.int32), np.asarray(masks, dtype=np.int32)]

def get_random_n(questions, question_masks, N):
	random_questions = [None]*N
	random_question_masks = [None]*N
	k = 0
	for i in random.sample(range(0, len(questions)), N):
		random_questions[k] = questions[i]
		random_question_masks[k] = question_masks[i]
		k += 1
	return random_questions, random_question_masks

def generate_data(posts, post_masks, questions, question_masks):
	q_len = len(questions[0])
	for i in range(len(posts)):
		if i == 0:
			data_posts = np.tile(posts[i], (N, 1))
			data_post_masks = np.tile(post_masks[i], (N, 1))
			random_questions, random_question_masks = get_random_n(questions, question_masks, N-1)
			data_questions = np.concatenate((questions[i].reshape(1, q_len), random_questions), axis=0)
			data_question_masks = np.concatenate((question_masks[i].reshape(1, q_len), random_question_masks), axis=0)
			data_labels = np.asarray([1] + [0]*(N-1))
		else:
			data_posts = np.concatenate((data_posts, np.tile(posts[i], (N, 1))), axis=0)
			data_post_masks = np.concatenate((data_post_masks, np.tile(post_masks[i], (N, 1))), axis=0)
			random_questions, random_question_masks = get_random_n(questions, question_masks, N-1)
			data_questions = np.concatenate((data_questions, questions[i].reshape(1, q_len), random_questions), axis=0)
			data_question_masks = np.concatenate((data_question_masks, question_masks[i].reshape(1, q_len), random_question_masks), axis=0)
			data_labels = np.concatenate((data_labels, np.asarray([1] + [0]*(N-1)), axis=0)
				
	return data_posts, data_post_masks, data_questions, data_question_masks, data_labels 

def iterate_minibatches(posts, post_masks, questions, question_masks, answers, answer_masks, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], post_masks[excerpt], questions[excerpt], question_masks[excerpt], answers[excerpt], answer_masks[excerpt]

def build_lstm(post_max_len, question_max_len, answer_max_len, len_voc, d_word, We, d_hid, freeze=False, eps=1e-6, lr=0.1, rho=1e-5):
	# input theano vars
	posts = T.imatrix()
	post_masks = T.matrix()
	questions = T.imatrix()
	question_masks = T.matrix()
	answers = T.imatrix()
	answer_masks = T.matrix()

	# define network
	l_post_in = lasagne.layers.InputLayer(shape=(d_word, post_max_len), input_var=posts)
	l_post_mask = lasagne.layers.InputLayer(shape=(d_word, post_max_len), input_var=post_masks)
	l_post_emb = lasagne.layers.EmbeddingLayer(l_post_in, len_voc, d_word, W=We)

	# now feed sequences of spans into VAN
	l_post_lstm = lasagne.layers.LSTMLayer(l_post_emb, d_hid, mask_input=l_post_mask, )

	# define network
	l_question_in = lasagne.layers.InputLayer(shape=(d_word, question_max_len), input_var=questions)
	l_question_mask = lasagne.layers.InputLayer(shape=(d_word, question_max_len), input_var=question_masks)
	l_question_emb = lasagne.layers.EmbeddingLayer(l_question_in, len_voc, d_word, W=We)

	# now feed sequences of spans into VAN
	l_question_lstm = lasagne.layers.LSTMLayer(l_question_emb, d_hid, mask_input=l_question_mask, )

	l_answer_in = lasagne.layers.InputLayer(shape=(d_word, answer_max_len), input_var=answers)
	l_answer_mask = lasagne.layers.InputLayer(shape=(d_word, answer_max_len), input_var=answer_masks)
	l_answer_emb = lasagne.layers.EmbeddingLayer(l_answer_in, len_voc, d_word, W=We)

	# freeze embeddings
	if freeze:
		l_post_emb.params[l_post_emb.W].remove('trainable')
		l_question_emb.params[l_question_emb.W].remove('trainable')

	post_out = lasagne.layers.get_output(l_post_lstm)
	question_out = lasagne.layers.get_output(l_question_lstm)

	post_out = T.mean(post_out * post_masks[:,:,None], axis=1)
	question_out = T.mean(question_out * question_masks[:,:,None], axis=1)

	answer_out = lasagne.layers.get_output(l_answer_emb)
	answer_out = T.mean(answer_out * answer_masks[:,:,None], axis=1)
	
	#pred_answer_out = T.sum(T.stack([post_out, question_out], axis=2), axis=2)
	pred_answer_out = question_out

	#post_lstm_params = lasagne.layers.get_all_params(l_post_lstm, trainable=True)
	question_lstm_params = lasagne.layers.get_all_params(l_question_lstm, trainable=True)
	#all_params = post_lstm_params + question_lstm_params
	all_params = question_lstm_params

	# objective computation
	loss = T.sum(lasagne.objectives.squared_error(pred_answer_out, answer_out))
	loss += rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)

	#train_fn = theano.function([posts, post_masks, questions, question_masks, answers, answer_masks], loss, updates=updates)
	#val_fn = theano.function([posts, post_masks, questions, question_masks, answers, answer_masks], [pred_answer_out, answer_out])
	train_fn = theano.function([questions, question_masks, answers, answer_masks], loss, updates=updates)
	val_fn = theano.function([questions, question_masks, answers, answer_masks], [pred_answer_out, answer_out])
	return train_fn, val_fn

def cluster_questions(question_embeddings, cluster_algo):
	if cluster_algo == "kmeans":
		n_clusters = int(len(question_embeddings)*SENTENCE_REDUCTION_FACTOR)
		kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=0).fit(question_embeddings) #n_jobs=-1 runs #CPUs jobs in parallel
		question_labels = kmeans.labels_
	elif cluster_algo == "dbscan":
		#dbscan = DBSCAN(eps=0.65, min_samples=1, n_jobs=-1).fit(question_embeddings)
		dbscan = DBSCAN(eps=0.65, min_samples=1).fit(question_embeddings)
		question_labels = dbscan.labels_
		n_clusters = len(set(question_labels)) - (1 if -1 in question_labels else 0)
	else:
		print "Unknown cluster algo ", cluster_algo
		sys.exit(0)
	return question_labels, n_clusters

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "usage: python baseline_post_ques.py posts.p questions.p word_vectors.txt"
		sys.exit(0)
	posts_str = p.load(open(sys.argv[1], 'rb'))
	questions_str = p.load(open(sys.argv[2], 'rb'))
	word_vectors_file = open(sys.argv[3], 'r')
	d_word = 200
	word_embeddings = []
	vocab = {}
	i = 0
	for line in word_vectors_file.readlines():
		vals = line.rstrip().split(' ')
		vocab[vals[0]] = i
		word_embeddings.append(map(float, vals[1:]))
		i += 1

	len_voc = len(vocab.keys())
	d_hid = 200
	lr = 0.001
	rho = 1e-5
	freeze = True
	batch_size = 10
	n_epochs = 20
	post_max_len = 100
	question_max_len = 20

	start = time.time()
	print 'compiling graph...'
	train_fn, val_fn = build_lstm(post_max_len, question_max_len, answer_max_len, len_voc, d_word, np.asarray(word_embeddings), d_hid, freeze, lr, rho)
	print 'done! Time taken: ', time.time()-start

	start = time.time()
	print 'generating data...'
	posts, post_masks = generate_vectors(posts_str, post_max_len, vocab)
	questions, question_masks = generate_vectors(questions_str, question_max_len, vocab)
	posts, post_masks, questions, question_masks, labels = generate_data(posts, post_masks, questions, question_masks)
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

		validate('dev', val_fn, dev)
		lstring = 'epoch:%d, cost:%f, time:%d' % \
			(epoch, cost / num_batches, time.time()-start)
		print lstring


