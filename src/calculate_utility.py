import sys
import argparse
import theano, lasagne
import numpy as np
import cPickle as p
import theano.tensor as T
from collections import Counter
import pdb
import time

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

def build_lstm(word_embeddings, len_voc, word_emb_dim, hidden_dim, N, post_max_len, answer_max_len, batch_size, learning_rate, rho, freeze=False):

	# input theano vars
	posts = T.imatrix()
	masks = T.matrix()
	labels = T.ivector()
	pred_answers_list = T.ftensor3()

	# define network
	l_post_in = lasagne.layers.InputLayer(shape=(None, post_max_len), input_var=posts)
	l_post_mask = lasagne.layers.InputLayer(shape=(None, post_max_len), input_var=masks)
	l_post_emb = lasagne.layers.EmbeddingLayer(l_post_in, len_voc, word_emb_dim, W=word_embeddings)


	# now feed sequences of spans into VAN
	l_post_lstm = lasagne.layers.LSTMLayer(l_post_emb, hidden_dim, mask_input=l_post_mask, )

	# freeze embeddings
	if freeze:
		l_post_emb.params[l_post_emb.W].remove('trainable')

	post_out = lasagne.layers.get_output(l_post_lstm)
	post_out = T.mean(post_out * masks[:,:,None], axis=1)
	post_answers_list_out = [None]*N
	for i in range(N):
		post_answers_list_out[i] = T.mean(T.stack([post_out, pred_answers_list[i]], axis=2), axis=2)

	l_post_out = lasagne.layers.InputLayer(shape=(None, hidden_dim), input_var=post_out)
	l_post_answers_list_out = [None]*N
	for i in range(N):
		l_post_answers_list_out[i] = lasagne.layers.InputLayer(shape=(None, hidden_dim), input_var=post_answers_list_out[i])
	
	# now predict
	l_post_dense = lasagne.layers.DenseLayer(l_post_out, num_units=1,\
									nonlinearity=lasagne.nonlinearities.sigmoid)

	l_post_answer_list_dense = [None]*N
	for i in range(N):
		l_post_answer_list_dense[i] = lasagne.layers.DenseLayer(l_post_answers_list_out[i], num_units=1,\
										W=l_post_dense.W, b=l_post_dense.b, \
										nonlinearity=lasagne.nonlinearities.sigmoid)

	post_params = lasagne.layers.get_all_params(l_post_lstm, trainable=True)
	emb_params = lasagne.layers.get_all_params(l_post_emb, trainbale=True)
	dense_params = lasagne.layers.get_all_params(l_post_dense, trainable=True)
	all_params = post_params + emb_params + dense_params
	
	# objective computation
	preds = lasagne.layers.get_output(l_post_dense)
	loss = T.sum(lasagne.objectives.binary_crossentropy(preds, labels))
	loss += rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=learning_rate)

	post_answer_preds = [None]*N
	for i in range(N):
		post_answer_preds[i] = lasagne.layers.get_output(l_post_answer_list_dense[i])

	train_fn = theano.function([posts, masks, labels], [preds, loss], updates=updates)
	dev_fn = theano.function([posts, masks, labels], [preds, loss],)
	val_fn = theano.function([posts, masks, pred_answers_list], post_answer_preds)
	return train_fn, dev_fn, val_fn

def validate(val_fn, fold_name, epoch, fold, batch_size):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	total = 0
	ones = 0
	posts, post_masks, labels = fold
	for p, pm, l in iterate_minibatches(posts, post_masks, labels, batch_size, shuffle=True):
		preds, loss = val_fn(p, pm, l)
		cost += loss
		for i, pred in enumerate(preds):
			if (pred >= 0.5 and labels[i] == 1) or (pred < 0.5 and labels[i] == 0):
				corr += 1
			total += 1
			if (pred >= 0.5):
				ones += 1
		num_batches += 1

	lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
				(fold_name, epoch, cost*1.0 / num_batches, corr*1.0 / total, time.time()-start)
	print lstring
	print ones, total

def validate_pred_answers(val_fn, fold_name, epoch, fold):
	start = time.time()
	corr = 0
	total = 0
	posts, post_masks, pred_answers_list = fold
	post_answer_preds = val_fn(posts, post_masks, pred_answers_list)
	preds = np.array(post_answer_preds).transpose() 
	for pred in preds:
		print pred
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

def generate_data(posts, post_max_len, labels):
	data_posts = np.zeros((len(posts), post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((len(posts), post_max_len), dtype=np.float32)
	labels = np.array(labels, dtype=np.int32)

	for i in range(len(posts)):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], post_max_len)

	return data_posts, data_post_masks, labels

def main(args):
	post_vectors = p.load(open(args.utility_post_vectors, 'rb'))
	labels = p.load(open(args.utility_labels, 'rb'))
	pred_ans_list_vectors = p.load(open(args.pred_ans_list_vectors, 'rb'))
	pred_ans_post_vectors = p.load(open(args.pred_ans_post_vectors, 'rb'))
	pred_ans_post_mask_vectors = p.load(open(args.pred_ans_post_mask_vectors, 'rb'))
	pred_ans_post_mask_vectors = np.asarray(pred_ans_post_mask_vectors, dtype=np.float32)
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	N = args.no_of_candidates
	
	print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len

	start = time.time()
	print 'generating data'
	posts, post_masks, labels = generate_data(post_vectors, args.post_max_len, labels)
	print 'done! Time taken: ', time.time() - start

	t_size = int(len(posts)*0.8)
	train = [posts[:t_size], post_masks[:t_size], labels[:t_size]]
	dev = [posts[t_size:], post_masks[t_size:], labels[t_size:]]
	test = pred_ans_post_vectors, pred_ans_post_mask_vectors, pred_ans_list_vectors

	print 'Size of training data: ', t_size
	print 'Size of dev data: ', len(posts)-t_size

	start = time.time()
	print 'compiling graph...'
	train_fn, dev_fn, val_fn = build_lstm(word_embeddings, vocab_size, word_emb_dim, args.hidden_dim, N, args.post_max_len, args.answer_max_len, args.batch_size, args.learning_rate, args.rho, freeze=freeze)
	print 'done! Time taken: ', time.time()-start
	
	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'Train', epoch, train, args.batch_size)
		validate(dev_fn, '\t DEV', epoch, dev, args.batch_size)
		validate_pred_answers(val_fn, '\t TEST', epoch, test)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--utility_post_vectors", type = str)
	argparser.add_argument("--utility_labels", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--batch_size", type = int, default = 200)
	argparser.add_argument("--no_of_epochs", type = int, default = 20)
	argparser.add_argument("--hidden_dim", type = int, default = 100)
	argparser.add_argument("--no_of_candidates", type = int, default = 5)
	argparser.add_argument("--learning_rate", type = float, default = 0.001)
	argparser.add_argument("--rho", type = float, default = 1e-5)
	argparser.add_argument("--post_max_len", type = int, default = 100)
	argparser.add_argument("--answer_max_len", type = int, default = 100)
	argparser.add_argument("--pred_ans_list_vectors", type = str)
	argparser.add_argument("--pred_ans_post_vectors", type = str)
	argparser.add_argument("--pred_ans_post_mask_vectors", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
