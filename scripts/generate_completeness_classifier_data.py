import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import numpy as np
import pdb
import cPickle as p

def preprocess(text):
	return word_tokenize(text.lower())

def get_word_indices(words, vocab):
	word_indices = [0]*len(words) 
	unk = "<unk>"
	for i, w in enumerate(words):
		try:
			word_indices[i] = vocab[w]
		except:
			word_indices[i] = vocab[unk]
	return word_indices

def generate_data(posthistory_file, vocab):
	posthistory_tree = ET.parse(posthistory_file)
	posts = []
	labels = []
	for posthistory in posthistory_tree.getroot():
		posthistory_typeid = posthistory.attrib['PostHistoryTypeId']
		if posthistory_typeid in ['2', '5']:
			text = posthistory.attrib['Text']
			words = preprocess(text)
			indices = get_word_indices(words, vocab)
			posts.append(indices)
			if posthistory_typeid == '2':
				labels.append(0)
			elif posthistory_typeid == '5':
				labels.append(1)
	return posts, labels

if __name__ == "__main__":
	if len(sys.argv) < 5:
		print "usage: python generate_completeness_classifier_data.py <posthistory.xml> <vocab.p> <output_completeness_posts.p> <output_completeness_labels.p>"
		sys.exit(0)
	posthistory_file = open(sys.argv[1], 'r')
	vocab = p.load(open(sys.argv[2], 'rb'))
	posts, labels = generate_data(posthistory_file, vocab)
	p.dump(posts, open(sys.argv[3], 'wb'))
	p.dump(labels, open(sys.argv[3], 'wb'))
