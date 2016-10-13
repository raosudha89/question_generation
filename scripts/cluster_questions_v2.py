import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
import numpy as np
import re
import pdb

SENTENCE_REDUCTION_FACTOR=0.5

def preprocess(text):
	return " ".join(word_tokenize(text.lower()))

def is_question(text):
	r = re.compile(r"(http://[^ ]+)")
	text = r.sub("", text) #remove urls so that ? is not identified in urls
	text = preprocess(text)
	if len(text.split()) > 25: #ignore long comments
		return False
	#remove content between quotes, braces so that ? is not identified in them
	text = re.sub("\".*?\"", "", text)
	text = re.sub("\(.*?\)", "", text)
	text = re.sub("\[.*?\]", "", text)
	words = text.split(' ')
	if '?' in words:
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
	question_comments = []
	for comment in comments_tree.getroot():
		postId = comment.attrib['PostId']
		if postId not in main_post_ids:
			continue	
		text = comment.attrib['Text']
		if is_question(text):
			text = preprocess(text)
			#sents = sent_tokenize(text)
			#question_content = ""
			#for sent in sents:
			#	if sent[-1] == '?':
			#		question_content += sent + " "
			#questions.append(question_content)
			questions.append(text)
			question_comments.append(text)
	return questions, question_comments

def cluster_question_sentences(questions, question_comments, word_vectors):
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
	#n_sentence_clusters = int(len(question_sentence_vectors)*SENTENCE_REDUCTION_FACTOR)
	#kmeans = KMeans(n_clusters=n_sentence_clusters, random_state=0).fit(question_sentence_vectors)
	#question_sentence_labels = kmeans.labels_
	dbscan = DBSCAN(eps=0.6, min_samples=1, n_jobs=4).fit(question_sentence_vectors)
	question_sentence_labels = dbscan.labels_
	n_sentence_clusters = len(set(question_sentence_labels)) - (1 if -1 in question_sentence_labels else 0)
	print "No. of questions", len(questions)
	print "No. of clusters ", n_sentence_clusters
	question_clusters = [[] for i in range(n_sentence_clusters)]
	for i, question in enumerate(questions):
		question_clusters[question_sentence_labels[i]].append(question_comments[i])	
	for cluster in question_clusters:
		if len(cluster) > 2:
			for sent in cluster: 
				print sent.encode('utf-8') + "\n"
			print
			print	

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
	questions, question_comments = extract_questions(posts_file, comments_file)
	cluster_question_sentences(questions, question_comments, word_vectors)		

