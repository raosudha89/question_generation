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
import gc

def get_data_masks(content, max_len):
	if len(content) > max_len:
		data = content[:max_len]
		data_mask = np.ones(max_len)
	else:
		data = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
		data_mask = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_mask

def generate_utility_data_from_triples(post_sents, ans_list, args):
	data_size = 2*len(post_sents)
	data_post_sents = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.int32)
	data_post_sent_masks = np.zeros((data_size, args.post_max_sents, args.post_max_sent_len), dtype=np.float32)
	data_labels = np.zeros(data_size, dtype=np.int32)

	for i in range(data_size/2):
		for j in range(min(args.post_max_sents, len(post_sents[i]))):
			data_post_sents[2*i][j], data_post_sent_masks[2*i][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
			data_post_sents[2*i+1][j], data_post_sent_masks[2*i+1][j] = get_data_masks(post_sents[i][j], args.post_max_sent_len)
		rand_index = random.randint(0,9)
		data_post_sents[2*i][-1], data_post_sent_masks[2*i][-1] = get_data_masks(ans_list[i][rand_index], args.post_max_sent_len)
		data_post_sents[2*i+1][-1], data_post_sent_masks[2*i+1][-1] = get_data_masks(ans_list[i][0], args.post_max_sent_len)
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

def build_utility_lstm(utility_post_list, utility_post_masks_list, post_max_sents, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):
	utility_post_out = [None]*post_max_sents
	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=utility_post_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=utility_post_masks_list[0])
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	#l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, hidden_dim, W=lasagne.init.GlorotNormal('relu'))
	# l_drop = lasagne.layers.DropoutLayer(l_emb, p=0.2)
	# l_lstm = lasagne.layers.LSTMLayer(l_drop, hidden_dim, mask_input=l_mask, )
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	utility_post_out[0] = lasagne.layers.get_output(l_lstm)
	utility_post_out[0] = T.mean(utility_post_out[0] * utility_post_masks_list[0][:,:,None], axis=1)
	for i in range(1, post_max_sents):
		l_in_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=utility_post_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=utility_post_masks_list[i])
		#l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=word_embeddings)
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, hidden_dim, W=l_emb.W)
		# l_drop_ = lasagne.layers.DropoutLayer(l_emb_, p=0.2)
		# l_lstm_ = lasagne.layers.LSTMLayer(l_drop_, hidden_dim, mask_input=l_mask_,\
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
		utility_post_out[i] = lasagne.layers.get_output(l_lstm_)
		utility_post_out[i] = T.mean(utility_post_out[i] * utility_post_masks_list[i][:,:,None], axis=1)
	
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
				# lasagne.layers.get_all_params(l_emb, trainable=True) 
		
	return utility_post_out, params

def build_evpi_model(word_embeddings, len_voc, word_emb_dim, args, freeze=False):

	# input theano vars
	
	utility_post_sents = T.itensor3()
	utility_post_sent_masks = T.ftensor3()
	utility_labels = T.ivector()

	utility_post_out, utility_post_lstm_params = build_utility_lstm(utility_post_sents, utility_post_sent_masks, \
																	args.post_max_sents, args.post_max_sent_len, \
																	word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)

	#utility_post_out = T.mean(utility_post_out, axis=0)
	utility_post_concatenate = T.concatenate(utility_post_out, axis=1)
	
	#l_utility_post_out = lasagne.layers.InputLayer(shape=(args.batch_size, args.hidden_dim), input_var=utility_post_out)
	l_utility_post_out = lasagne.layers.InputLayer(shape=(args.batch_size, (args.post_max_sents)*args.hidden_dim), input_var=utility_post_concatenate)
	# l_utility_post_drop = lasagne.layers.DropoutLayer(l_utility_post_out, p=0.2)
	# l_utility_post_dense = lasagne.layers.DenseLayer(l_utility_post_drop, num_units=args.hidden_dim,\
	l_utility_post_dense = lasagne.layers.DenseLayer(l_utility_post_out, num_units=args.hidden_dim,\
													nonlinearity=lasagne.nonlinearities.rectify)
	# for i in range(10):
	# 	l_utility_post_dense = lasagne.layers.DenseLayer(l_utility_post_dense, num_units=args.hidden_dim,\
	# 													nonlinearity=lasagne.nonlinearities.rectify)
	l_utility_post_dense2 = lasagne.layers.DenseLayer(l_utility_post_dense, num_units=1,\
													nonlinearity=lasagne.nonlinearities.sigmoid)
	utility_preds = lasagne.layers.get_output(l_utility_post_dense2)
	utility_loss = T.sum(lasagne.objectives.binary_crossentropy(utility_preds, utility_labels))

	utility_dense_params = lasagne.layers.get_all_params(l_utility_post_dense2, trainable=True)
							# lasagne.layers.get_all_params(l_utility_post_dense, trainable=True)
	utility_all_params = utility_post_lstm_params + utility_dense_params

	utility_loss += args.rho * sum(T.sum(l ** 2) for l in utility_all_params)

	utility_updates = lasagne.updates.adam(utility_loss, utility_all_params, learning_rate=args.learning_rate)
	
	utility_train_fn = theano.function([utility_post_sents, utility_post_sent_masks, utility_labels], \
									[utility_preds, utility_loss], updates=utility_updates)
	utility_dev_fn = theano.function([utility_post_sents, utility_post_sent_masks, utility_labels], \
									[utility_preds, utility_loss],)

	return utility_train_fn, utility_dev_fn

def utility_iterate_minibatches(post_sents, post_sent_masks, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(post_sents.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, post_sents.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield post_sents[excerpt], post_sent_masks[excerpt], labels[excerpt]

def validate(utility_val_fn, fold_name, epoch, utility_fold, args):
	start = time.time()
	utility_num_batches = 0
	utility_corr = 0
	utility_total = 0
	utility_cost = 0
	batch_size = args.batch_size
	
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
		utility_num_batches += 1
		
	lstring = '%s: epoch:%d, utility_cost:%f, utility_acc:%f time:%d' % \
				(fold_name, epoch, utility_cost*1.0/utility_num_batches, utility_corr*1.0/utility_total, time.time()-start)
	print lstring

def main(args):
	post_sent_vectors = p.load(open(args.post_sent_vectors_train, 'rb'))

	ques_list_vectors = p.load(open(args.ques_list_vectors_train, 'rb'))
	ans_list_vectors = p.load(open(args.ans_list_vectors_train, 'rb'))
	utility_post_sent_vectors = p.load(open(args.utility_post_sent_vectors_train, 'rb'))
	utility_labels = p.load(open(args.utility_labels_train, 'rb'))

	post_sent_vectors_test = p.load(open(args.post_sent_vectors_test, 'rb'))
	ques_list_vectors_test = p.load(open(args.ques_list_vectors_test, 'rb'))
	ans_list_vectors_test = p.load(open(args.ans_list_vectors_test, 'rb'))
	utility_post_sent_vectors_test = p.load(open(args.utility_post_sent_vectors_test, 'rb'))
	utility_labels_test = p.load(open(args.utility_labels_test, 'rb'))

	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab_size = len(word_embeddings)
	word_emb_dim = len(word_embeddings[0])
	freeze = False
	
	print 'vocab_size ', vocab_size, ' word_emb_dim ', word_emb_dim

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
	utility_train_fn, utility_dev_fn = build_evpi_model(word_embeddings, vocab_size, word_emb_dim, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start

	gc.collect()
	#train_fn, dev_fn, utility_train_fn, utility_dev_fn = None, None, None, None
	# train network
	for epoch in range(args.no_of_epochs):
		validate(utility_train_fn, 'Train', epoch, utility_train, args)
		validate(utility_dev_fn, '\t DEV', epoch, utility_dev, args)
		validate(utility_dev_fn, '\t TEST', epoch, utility_test, args)
		print "\n"

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_sent_vectors_train", type = str)
	argparser.add_argument("--ques_list_vectors_train", type = str)
	argparser.add_argument("--ans_list_vectors_train", type = str)
	argparser.add_argument("--utility_post_sent_vectors_train", type = str)
	argparser.add_argument("--utility_labels_train", type = str)
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
	argparser.add_argument("--post_max_sents", type = int, default = 10)
	argparser.add_argument("--post_max_sent_len", type = int, default = 10)
	argparser.add_argument("--ques_max_len", type = int, default = 10)
	argparser.add_argument("--ans_max_len", type = int, default = 10)
	argparser.add_argument("--_lambda", type = float, default = 0.5)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
