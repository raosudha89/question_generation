import sys
from nltk.tokenize import word_tokenize
import numpy as np
import cPickle as p
import random

def preprocess(text):
	return word_tokenize(text.lower())

def get_word_indices(words, vocab):
	word_indices = [0]*len(words)
	UNK = "<unk>"
	for i, w in enumerate(words):
		try:
			word_indices[i] = vocab[w]
		except:
			word_indices[i] = vocab[UNK]
	return word_indices

def generate_vectors(contents_str, vocab):
	contents = [None]*len(contents_str)
	for i, content_str in enumerate(contents_str):
		words = preprocess(content_str)
		indices = get_word_indices(words, vocab)
		contents[i] = indices
	return contents

def get_random_n(questions, N):
	random_questions = [None]*N
	k = 0
	for i in random.sample(range(0, len(questions)), N):
		random_questions[k] = questions[i]
		k += 1
	return random_questions

def generate_data(posts, questions):
	N = 2
	data_posts = [None]*len(posts)*N
	data_questions = [None]*len(posts)*N
	data_labels = [None]*len(posts)*N
	for i in range(len(posts)):
		data_posts[i*N] = posts[i]
		data_questions[i*N] = questions[i]
		data_labels[i*N] = 1
		for k in range(1,N):
			data_posts[i*N+k] = posts[i]
			data_questions[i*N+k] = questions[random.randint(0, len(questions)-1)]
			data_labels[i*N+k] = 0
	return data_posts, data_questions, data_labels

if __name__ == "__main__":
	if len(sys.argv) < 7:
		print "usage: python generate_data_for_lstm.py data_posts.p data_questions.p vocab.p out_posts.p out_questions.p out_labels.p"
		sys.exit(0)
	posts_str = p.load(open(sys.argv[1], 'rb'))
	questions_str = p.load(open(sys.argv[2], 'rb'))
	vocab = p.load(open(sys.argv[3], 'rb'))
	posts = generate_vectors(posts_str, vocab)
	questions = generate_vectors(questions_str, vocab)
	posts, questions, labels = generate_data(posts, questions)
	p.dump(posts, open(sys.argv[4], 'wb'))
	p.dump(questions, open(sys.argv[5], 'wb'))
	p.dump(labels, open(sys.argv[6], 'wb'))

