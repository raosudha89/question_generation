import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import numpy as np
import pdb

WORD_REDUCTION_FACTOR=0.01
SENTENCE_REDUCTION_FACTOR=0.5

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

def cluster_question_words(word_vectors, questions):
	question_words = []
	question_word_vectors = []
	for question in questions:
		words = question.split()
		for w in words:
			if w not in question_words:
				try:
					v = word_vectors[w]
					question_words.append(w)
					question_word_vectors.append(v)
				except:
					pass
	print "No. of questions ", len(questions)
	print "No. of unique words in questions ", len(question_words)
	n_word_clusters = int(len(question_word_vectors)*WORD_REDUCTION_FACTOR)
	kmeans = KMeans(n_clusters=n_word_clusters, random_state=0).fit(question_word_vectors)
	question_labels = kmeans.labels_
	question_word_cluster_label = {}
	for i, w in enumerate(question_words):
		question_word_cluster_label[w] = question_labels[i]
	return question_word_cluster_label, n_word_clusters

def normalize(sentence_vector):
	total_wc = 0
	for wc in sentence_vector:
		total_wc += wc
	for i in range(len(sentence_vector)):
		sentence_vector[i] = sentence_vector[i]*1.0/total_wc
	return sentence_vector

def cluster_question_sentences(questions, question_word_cluster_label, n_word_clusters):
	question_sentence_vectors = np.zeros((len(questions), n_word_clusters+1)) #+1 for UNK word
	for i, question in enumerate(questions):
		for w in question.split():
			try:
				word_cluster_label = int(question_word_cluster_label[w])
			except:
				word_cluster_label = n_word_clusters
			question_sentence_vectors[i][word_cluster_label] += 1
		question_sentence_vectors[i] = normalize(question_sentence_vectors[i])
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
	question_word_cluster_label, n_word_clusters = cluster_question_words(word_vectors, questions)
	cluster_question_sentences(questions, question_word_cluster_label, n_word_clusters)		

