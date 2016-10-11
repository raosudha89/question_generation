import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import numpy as np
import pdb

SENTENCE_REDUCTION_FACTOR=0.1

def preprocess(text):
	return " ".join(word_tokenize(text.lower()))

def is_question(text):
	words = text.split(' ')
	#question_tokens = ['?', 'who', 'when', 'where', 'why', 'how', 'what']
	question_tokens = ['?']
	for tok in question_tokens: 
		if tok in words:
			return True
	return False

def extract_questions(posts_file, comments_file):
	posts_tree = ET.parse(posts_file)
	main_post_ids = []
	for post in posts_tree.getroot():
		postId = post.attrib['Id']
		postTypeId = post.attrib['PostTypeId'] 
		if postTypeId == '1':
			main_post_ids.append(postId)
	comments_tree = ET.parse(comments_file)
	questions = []
	for comment in comments_tree.getroot():
		postId = comment.attrib['PostId']
		if postId not in main_post_ids:
			continue	
		text = comment.attrib['Text']
		text = preprocess(text)
		if is_question(text):
			questions.append(text)
	return questions

def cluster_question_sentences(questions, word_vectors):
	word_vector_len = len(word_vectors[word_vectors.keys()[0]])
	question_sentence_vectors = np.empty((len(questions), word_vector_len))
	for i, question in enumerate(questions):
		words = question.split()
		question_word_vectors = np.zeros((len(words), word_vector_len))
		for j, w in enumerate(words):
			try:
				question_word_vectors[j] = word_vectors[w]
			except:
				pass
		question_sentence_vectors[i] = np.mean(question_word_vectors, axis=0)
	n_sentence_clusters = int(len(question_sentence_vectors)*SENTENCE_REDUCTION_FACTOR)
	kmeans = KMeans(n_clusters=n_sentence_clusters, random_state=0).fit(question_sentence_vectors)
	question_sentence_labels = kmeans.labels_
	question_clusters = [[] for i in range(n_sentence_clusters)]
	for i, question in enumerate(questions):
		question_clusters[question_sentence_labels[i]].append(questions[i])	
	pdb.set_trace()	

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "Usage: python cluster_questions.py <posts.xml> <comments.xml> <word_vectors>"
		sys.exit(0)
	posts_file = open(sys.argv[1], 'r')
	comments_file = open(sys.argv[2], 'r')
	word_vectors_file = open(sys.argv[3], 'r')
	word_vectors = {}
	for line in word_vectors_file.readlines():
		vals = line.rstrip().split(' ')
		word_vectors[vals[0]] = map(float, vals[1:])
	questions = extract_questions(posts_file, comments_file)
	cluster_question_sentences(questions, word_vectors)		

