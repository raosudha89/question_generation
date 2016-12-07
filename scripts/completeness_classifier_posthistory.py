import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import theano, lasagne,time
import numpy as np
import theano.tensor as T
from collections import Counter
import pdb
import time

def preprocess(text):
	return word_tokenize(text.lower())

def get_word_indices(words, vocab, max_len):
	word_indices = np.zeros(max_len, dtype=np.int32)
	unk = "<unk>"
	for i, w in enumerate(words):
		if i >= max_len:
			break
		try:
			word_indices[i] = vocab[w]
		except:
			word_indices[i] = vocab[unk]
	return word_indices

def generate_data(posthistory_file, max_len, vocab):
	posthistory_tree = ET.parse(posthistory_file)
	posts = []
	masks = []
	labels = []
	for posthistory in posthistory_tree.getroot()[:25000]:
		posthistory_typeid = posthistory.attrib['PostHistoryTypeId']
		if posthistory_typeid in ['2', '5']:
			text = posthistory.attrib['Text']
			words = preprocess(text)
			indices = get_word_indices(words, vocab, max_len)
			posts.append(indices)
			masks.append(np.concatenate((np.ones(min(len(words), max_len), dtype=np.int32), np.zeros(max(0, max_len-len(words)), dtype=np.int32)))) 
			if posthistory_typeid == '2':
				labels.append(0)
			elif posthistory_typeid == '5':
				labels.append(1)
	return [np.asarray(posts, dtype=np.int32), np.asarray(masks, dtype=np.int32), np.asarray(labels, dtype=np.int32)]

def iterate_minibatches(posts, masks, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], masks[excerpt], labels[excerpt]

def build_lstm(max_len, len_voc, d_word, We, num_labels, d_hid, freeze=False, eps=1e-6, lr=0.1, rho=1e-5):
	# input theano vars
	posts = T.imatrix(name='post')
	masks = T.matrix(name='mask')
	labels = T.ivector('label')

	# define network
	l_in = lasagne.layers.InputLayer(shape=(d_word, max_len), input_var=posts)
	l_mask = lasagne.layers.InputLayer(shape=(d_word, max_len), input_var=masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, d_word, W=We)

	# now feed sequences of spans into VAN
	l_lstm = lasagne.layers.LSTMLayer(l_emb, d_hid, mask_input=l_mask, )

	# freeze embeddings
	if freeze:
		l_emb.params[l_emb.W].remove('trainable')

	# now predict
	l_forward_slice = lasagne.layers.SliceLayer(l_lstm, -1, 1)
	l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=num_labels,\
									nonlinearity=lasagne.nonlinearities.softmax)

	# objective computation
	preds = lasagne.layers.get_output(l_out)
	loss = T.sum(lasagne.objectives.categorical_crossentropy(preds, labels))
	loss += rho * sum(T.sum(l ** 2) for l in lasagne.layers.get_all_params(l_out))
	all_params = lasagne.layers.get_all_params(l_out, trainable=True)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)

	train_fn = theano.function([posts, masks, labels], [preds, loss], updates=updates)
	val_fn = theano.function([posts, masks], preds)
	debug_fn = theano.function([posts, masks], lasagne.layers.get_output(l_lstm))
	return train_fn, val_fn, debug_fn

def validate(name, val_fn, fold):
	corr = 0
	total = 0
	c = Counter()
	posts, masks, labels = fold
	preds = val_fn(posts, masks)
	preds = np.argmax(preds, axis=1)

	# for s, m, l in iterate_minibatches(posts, masks, labels, 100):
	for i, pred in enumerate(preds):
		if pred == labels[i]:
			corr += 1
		total += 1
		c[pred] += 1

	lstring = 'fold:%s, corr:%d, total:%d, acc:%f' %\
		(name, corr, total, float(corr) / float(total))
	print lstring
	print c
	return lstring

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "usage: python completeness_classifier_posthistory.py <posthistory.xml> <word_vectors.txt>"
		sys.exit(0)
	posthistory_file = open(sys.argv[1], 'r')
	word_vectors_file = open(sys.argv[2], 'r')
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
	num_labels = 2
	d_hid = 100
	lr = 0.001
	rho = 1e-5
	freeze = True
	batch_size = 100
	n_epochs = 20
	max_len = 100

	start = time.time()
	print 'compiling graph...'
	train_fn, val_fn, debug_fn = build_lstm(max_len, len_voc, d_word, np.asarray(word_embeddings), num_labels, d_hid, freeze, lr, rho)
	print 'done! Time taken: ', time.time()-start

	start = time.time()
	print 'generating data...'
	data = generate_data(posthistory_file, max_len, vocab)
	print 'done! Time taken: ', time.time()-start
	posts, masks, labels = data
	t_size = int(posts.shape[0]*0.8)
	train = posts[:t_size], masks[:t_size], labels[:t_size]
	dev = posts[t_size:], masks[t_size:], labels[t_size:]

	print 'Training...'
	# train network
	for epoch in range(n_epochs):
		cost = 0.
		start = time.time()
		posts, masks, labels = train	
		num_batches = 0.
		for p, m, l in iterate_minibatches(posts, masks, labels, batch_size, shuffle=True):
			preds, loss = train_fn(p, m, l)
			cost += loss
			num_batches += 1

		lstring = 'epoch:%d, cost:%f, time:%d' % \
			(epoch, cost / num_batches, time.time()-start)
		print lstring

		trperf = validate('train', val_fn, train)
		devperf = validate('dev', val_fn, dev)
		print '\n'	

