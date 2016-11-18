import sys
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

def generate_data(posts, post_max_len, questions, question_max_len, no_of_questions):
	data_posts = np.zeros((len(posts)/no_of_questions, post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((len(posts)/no_of_questions, post_max_len), dtype=np.int32)
	
	data_questions_list = np.zeros((len(questions)/no_of_questions, no_of_questions, question_max_len), dtype=np.int32)
	data_question_masks_list = np.zeros((len(questions)/no_of_questions, no_of_questions, question_max_len), dtype=np.int32)

	for i in range(len(posts)/no_of_questions):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i*no_of_questions], post_max_len)
		for j in range(no_of_questions):
			data_questions_list[i][j], data_question_masks_list[i][j] = get_data_masks(questions[i*no_of_questions+j], question_max_len)

	return data_posts, data_post_masks, data_questions_list, data_question_masks_list

def iterate_minibatches(posts, post_masks, questions, question_masks, labels, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(posts))
        np.random.shuffle(indices)
    for start_idx in range(0, len(posts) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield posts[excerpt], post_masks[excerpt], questions[excerpt], question_masks[excerpt], labels[excerpt]

def validate(val_fn, fold_name, epoch, fold, batch_size):
	start = time.time()
	num_batches = 0.
	cost = 0.
	acc = 0.
	posts, post_masks, questions_list, question_masks_list = fold
	labels = np.zeros((len(posts), 2), dtype=np.int32)
	for i in range(len(posts)):
		labels[i][0] = 1
	for c, cm, r, rm, l in iterate_minibatches(posts, post_masks, questions_list, question_masks_list, labels, \
												batch_size, shuffle=True):

		r_1 = r[:,0]
		rm_1 = rm[:,0].astype('float32')
		r_2 = r[:,1]
		rm_2 = rm[:,1].astype('float32')
		cm = cm.astype('float32')
		loss, probs = val_fn(c, cm, r_1, rm_1, r_2, rm_2, l)
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

def build_lstm(word_embeddings, len_voc, d_word, d_hidden, post_max_len, question_max_len, batch_size, lr, freeze=False):

    # input theano vars
    posts = T.imatrix(name='post')
    post_masks = T.matrix(name='post_mask')
    questions_1 = T.imatrix(name='question_1')
    question_masks_1 = T.matrix(name='question_mask_1')
    questions_2 = T.imatrix(name='question_2')
    question_masks_2 = T.matrix(name='question_mask_2')
    labels = T.imatrix(name='label')   
 
    # define network
    l_post = lasagne.layers.InputLayer(shape=(batch_size, post_max_len), input_var=posts)
    l_post_mask = lasagne.layers.InputLayer(shape=(batch_size, post_max_len), input_var=post_masks)
    l_question_1 = lasagne.layers.InputLayer(shape=(batch_size, question_max_len), input_var=questions_1)
    l_question_mask_1 = lasagne.layers.InputLayer(shape=(batch_size, question_max_len), input_var=question_masks_1)
    l_question_2 = lasagne.layers.InputLayer(shape=(batch_size, question_max_len), input_var=questions_2)
    l_question_mask_2 = lasagne.layers.InputLayer(shape=(batch_size, question_max_len), input_var=question_masks_2)

    #l_post_emb = lasagne.layers.EmbeddingLayer(l_post, len_voc, d_word, W=lasagne.init.GlorotNormal())
    #l_question_1_emb = lasagne.layers.EmbeddingLayer(l_question_1, len_voc, d_word, W=lasagne.init.GlorotNormal())
    l_post_emb = lasagne.layers.EmbeddingLayer(l_post, len_voc, d_word, W=word_embeddings)
    l_question_1_emb = lasagne.layers.EmbeddingLayer(l_question_1, len_voc, d_word, W=word_embeddings)
    l_question_2_emb = lasagne.layers.EmbeddingLayer(l_question_2, len_voc, d_word, W=l_question_1_emb.W)
    
    # now feed sequences of spans into VAN
    l_post_lstm = lasagne.layers.LSTMLayer(l_post_emb, d_hidden, \
									mask_input=l_post_mask, \
									#only_return_final=True, \
									peepholes=False,\
									)
    l_question_1_lstm = lasagne.layers.LSTMLayer(l_question_1_emb, d_hidden, \
									mask_input=l_question_mask_1, \
									#only_return_final=True, \
									peepholes=False,\
									)
     
    l_question_2_lstm = lasagne.layers.LSTMLayer(l_question_2_emb, d_hidden, \
									mask_input=l_question_mask_2, \
									#only_return_final=True, \
									ingate=lasagne.layers.Gate(W_in=l_question_1_lstm.W_in_to_ingate,\
																W_hid=l_question_1_lstm.W_hid_to_ingate,\
																b=l_question_1_lstm.b_ingate,\
																nonlinearity=l_question_1_lstm.nonlinearity_ingate),\
									outgate=lasagne.layers.Gate(W_in=l_question_1_lstm.W_in_to_outgate,\
																W_hid=l_question_1_lstm.W_hid_to_outgate,\
																b=l_question_1_lstm.b_outgate,\
																nonlinearity=l_question_1_lstm.nonlinearity_outgate),\
									forgetgate=lasagne.layers.Gate(W_in=l_question_1_lstm.W_in_to_forgetgate,\
																W_hid=l_question_1_lstm.W_hid_to_forgetgate,\
																b=l_question_1_lstm.b_forgetgate,\
																nonlinearity=l_question_1_lstm.nonlinearity_forgetgate),\
									cell=lasagne.layers.Gate(W_in=l_question_1_lstm.W_in_to_cell,\
																W_hid=l_question_1_lstm.W_hid_to_cell,\
																b=l_question_1_lstm.b_cell,\
																nonlinearity=l_question_1_lstm.nonlinearity_cell),\
									peepholes=False,\
									)

    #for i in range(DEPTH):
    #	l_post_lstm = lasagne.layers.DenseLayer(l_post_lstm, num_units=d_hidden,\
	#												nonlinearity=lasagne.nonlinearities.rectify)
    #	l_question_1_lstm = lasagne.layers.DenseLayer(l_question_1_lstm, num_units=d_hidden,\
	#												nonlinearity=lasagne.nonlinearities.rectify, \
	#												W=l_post_lstm.W)
    #	l_question_2_lstm = lasagne.layers.DenseLayer(l_question_2_lstm, num_units=d_hidden,\
	#												nonlinearity=lasagne.nonlinearities.rectify,\
	#												W=l_post_lstm.W)

    l_post_lstm = lasagne.layers.DropoutLayer(l_post_lstm)
    l_question_1_lstm = lasagne.layers.DropoutLayer(l_question_1_lstm)
    l_question_2_lstm = lasagne.layers.DropoutLayer(l_question_2_lstm)
        
    # residual learning
    #l_post_lstm = lasagne.layers.ElemwiseSumLayer([l_post_lstm, l_post_emb])
    #l_question_1_lstm = lasagne.layers.ElemwiseSumLayer([l_question_1_lstm, l_question_1_emb])
    #l_question_2_lstm = lasagne.layers.ElemwiseSumLayer([l_question_2_lstm, l_question_2_emb])
    
    # now get aggregate embeddings
    post_out = lasagne.layers.get_output(l_post_lstm)
    question_1_out = lasagne.layers.get_output(l_question_1_lstm)
    question_2_out = lasagne.layers.get_output(l_question_2_lstm)

    post_out = T.mean(post_out * post_masks[:,:,None], axis=1)
    question_1_out = T.mean(question_1_out * question_masks_1[:,:,None], axis=1)
    question_2_out = T.mean(question_2_out * question_masks_2[:,:,None], axis=1)
	
    # objective computation
    M = theano.shared(np.eye(d_hidden, dtype=np.float32))

    probs_1 = T.sum(T.dot(post_out, M)*question_1_out, axis=1)
    probs_2 = T.sum(T.dot(post_out, M)*question_2_out, axis=1)
 
    probs = lasagne.nonlinearities.softmax(T.stack([probs_1, probs_2], axis=1))
    loss = T.sum(lasagne.objectives.categorical_crossentropy(probs, labels))

    emb_post_params = lasagne.layers.get_all_params(l_post_emb, trainable=True)
    emb_question_params = lasagne.layers.get_all_params(l_question_1_emb, trainable=True)

    all_post_params = lasagne.layers.get_all_params(l_post_lstm, trainable=True)
    all_question_1_params = lasagne.layers.get_all_params(l_question_1_lstm, trainable=True)
    all_question_2_params = lasagne.layers.get_all_params(l_question_2_lstm, trainable=True)

    for i in range(len(all_post_params)):
        assert((all_question_1_params[i].get_value() == all_question_2_params[i].get_value()).all())
    updates = lasagne.updates.adam(loss, emb_post_params + emb_question_params + all_post_params + all_question_1_params + [M], learning_rate=lr)
    train_fn = theano.function([posts, post_masks, questions_1, question_masks_1, \
								questions_2, question_masks_2, labels], \
                               [loss, probs], updates=updates)
    val_fn = theano.function([posts, post_masks, questions_1, question_masks_1, \
								questions_2, question_masks_2, labels], \
                               [loss, probs])
    return train_fn, val_fn

if __name__ == '__main__':
	np.set_printoptions(linewidth=160)
	if len(sys.argv) < 4:
		print "usage: python baseline_post_ques_ranking.py posts.p questions.p no_of_questions word_embeddings.p batch_size"
		sys.exit(0)
	posts = p.load(open(sys.argv[1], 'rb'))
	questions = p.load(open(sys.argv[2], 'rb'))
	no_of_questions = int(sys.argv[3])
	word_embeddings = p.load(open(sys.argv[4], 'rb'))
	vocab_size = len(word_embeddings) 
	batch_size = int(sys.argv[5])
	d_word = 200
	d_hidden = 100
	freeze = False
	lr = 0.001
	n_epochs = 10 
	post_max_len = 100
	question_max_len = 20
	print 'vocab_size ', vocab_size, ', post_max_len ', post_max_len, ', question_max_len ', question_max_len

	start = time.time()
	print 'generating data'
	posts, post_masks, questions_list, question_masks_list = generate_data(posts, post_max_len, questions, question_max_len, no_of_questions)
	print 'done! Time taken: ', time.time()-start

	t_size = int(len(posts)*0.8)
	train = [posts[:t_size], post_masks[:t_size], questions_list[:t_size], question_masks_list[:t_size]]
	dev = [posts[t_size:], post_masks[t_size:], questions_list[t_size:], question_masks_list[t_size:]]

	print 'Size of training data: ', t_size
	print 'Size of dev data: ', len(posts)-t_size

	print 'compiling graph...'
	train_fn, val_fn = build_lstm(np.asarray(word_embeddings, dtype=np.float32), vocab_size, d_word, d_hidden, post_max_len, question_max_len, batch_size, lr, freeze=freeze)
	print 'done compiling'

	#N = int(len(train[0])*0.9)
	#train_90 = [train[0][:N], train[1][:N], train[2][:N], train[3][:N]]
	#train_10 = [train[0][N:], train[1][N:], train[2][N:], train[3][N:]]

	#M = int(len(dev[0])*0.9)
	#dev_90 = [dev[0][:M], dev[1][:M], dev[2][:M], dev[3][:M]]
	#dev_10 = [dev[0][M:], dev[1][M:], dev[2][M:], dev[3][M:]]

	# train network
	for epoch in range(n_epochs):

		validate(train_fn, 'Train', epoch, train, batch_size)
		#validate(val_fn, '\t Test on Train', epoch, train, batch_size)
		#validate(train_fn, 'TRAIN on train_90 ', epoch, train_90, batch_size) 
		#validate(train_fn, 'TRAIN on dev_90 ', epoch, dev_90, batch_size) 
		#validate(train_fn, 'Train on DEV', epoch, dev, batch_size)
		validate(val_fn, '\t DEV', epoch, dev, batch_size)
		#validate(val_fn, '\t \t TEST on train_10 ', epoch, train_10, batch_size)
		#validate(val_fn, '\t \t TEST on dev_10 ', epoch, dev_10, batch_size)
		print "\n"
