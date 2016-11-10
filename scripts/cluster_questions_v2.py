import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from sklearn.cluster import SpectralClustering
import numpy as np
import re
import string
import pdb

SENTENCE_REDUCTION_FACTOR=0.75

def preprocess(text):
	return " ".join(word_tokenize(text.lower()))

def get_question(text):
	r = re.compile(r"(http://[^ ]+)")
	text = r.sub("", text) #remove urls so that ? is not identified in urls
	text = preprocess(text)
	if '?' in text.split():
		text = text.split('?')[0]+ '?'
		words = text.split()
		if len(words) > 25: #ignore long comments
			return None
		return text
	return None

def extract_questions(posts_file, comments_file):
	posts_tree = ET.parse(posts_file)
	main_post_ids = []
	main_post_titles = []
	main_post_bodies = []
	unanswered_post_titles = []
	unanswered_post_bodies = []
	
	for post in posts_tree.getroot():
		postId = post.attrib['Id']
		postTypeId = post.attrib['PostTypeId'] 
		if postTypeId == '1':
			main_post_ids.append(postId)
			main_post_titles.append(post.attrib['Title'])
			main_post_bodies.append(post.attrib['Body'])
			if post.attrib['AnswerCount'] == "0":
				unanswered_post_titles.append(post.attrib['Title'])
				unanswered_post_bodies.append(post.attrib['Body'])

	print "No. of main posts ", len(main_post_ids)
	comments_tree = ET.parse(comments_file)
	questions = []
	post_titles = []
	post_bodies = []
	comment_count = 0
	for comment in comments_tree.getroot():
		postId = comment.attrib['PostId']
		if postId not in main_post_ids:
			continue
		i = main_post_ids.index(postId)
		comment_count += 1	
		text = comment.attrib['Text']
		question = get_question(text)
		if question:
			questions.append(question)	
			post_titles.append(main_post_titles[i])
			post_bodies.append(main_post_bodies[i])

	print "No. of comments to main posts ", comment_count
	print "No of question comments ", len(questions)
	return questions, post_titles, post_bodies, unanswered_post_titles, unanswered_post_bodies

def cluster_question_sentences(questions, post_titles, post_bodies, word_vectors, cluster_algo):
	word_vector_len = len(word_vectors[word_vectors.keys()[0]])
	question_sentence_vectors = np.empty((len(questions), word_vector_len))
	for i, question in enumerate(questions):
		words = question.split()
		question_word_vectors = np.zeros((len(words), word_vector_len))
		for j, w in enumerate(words):
			if w in string.punctuation:
				continue
			try:
				question_word_vectors[j] = word_vectors[w]
			except:
				pass
		question_sentence_vectors[i] = np.mean(question_word_vectors, axis=0)
	if cluster_algo == "kmeans":
		n_sentence_clusters = int(len(question_sentence_vectors)*SENTENCE_REDUCTION_FACTOR)
		kmeans = KMeans(n_clusters=n_sentence_clusters, n_jobs=-1, random_state=0).fit(question_sentence_vectors) #n_jobs=-1 runs #CPUs jobs in parallel
		question_sentence_labels = kmeans.labels_
	elif cluster_algo == "dbscan":
		dbscan = DBSCAN(eps=0.65, min_samples=1, n_jobs=-1).fit(question_sentence_vectors)
		question_sentence_labels = dbscan.labels_
		n_sentence_clusters = len(set(question_sentence_labels)) - (1 if -1 in question_sentence_labels else 0)
	elif cluster_algo == "spectral":
		n_sentence_clusters = int(len(question_sentence_vectors)*SENTENCE_REDUCTION_FACTOR)
		spectral = SpectralClustering(n_clusters=n_sentence_clusters, n_jobs=-1, eigen_solver='arpack', affinity="nearest_neighbors").fit(question_sentence_vectors)
		question_sentence_labels = spectral.labels_
	else:
		print "Unknown cluster algo ", cluster_algo
		sys.exit(0)
	print "No. of questions", len(questions)
	print "No. of clusters ", n_sentence_clusters
	question_clusters = [[] for i in range(n_sentence_clusters)]
	for i, question in enumerate(questions):
		question_clusters[question_sentence_labels[i]].append((questions[i], post_titles[i], post_bodies[i]))
	useful_question_clusters = []	
	for cluster in question_clusters:
		if len(cluster) > 5 and len(cluster) < len(questions)/4:
			useful_question_cluster = []
			for (question, post_title, post_body) in cluster: 
				#print "----------------------------------------------------------------"
				print question.encode('utf-8')
				#print post_title.encode('utf-8')
				#print post_body.encode('utf-8') + "\n"
				post_vector = get_post_vector(post_title, post_body, word_vectors)
				useful_question_cluster.append((question, post_title, post_body, post_vector))
			useful_question_clusters.append(useful_question_cluster)
			#print "----------------------------------------------------------------"
			#print "----------------------------------------------------------------"
			#print "----------------------------------------------------------------"
			print
			print
	return useful_question_clusters

def get_post_vector(post_title, post_body, word_vectors):
	word_vector_len = len(word_vectors[word_vectors.keys()[0]])
	post = preprocess(post_title) + " " + preprocess(post_body)
	words = post.split()
	post_word_vectors = np.zeros((len(words), word_vector_len))
	for j, w in enumerate(words):
		if w in string.punctuation:
			continue
		try:
			post_word_vectors[j] = word_vectors[w]
		except:
			pass
	post_vector = np.mean(post_word_vectors, axis=0)
	return post_vector

def predict_comment(question_clusters, unanswered_post_titles, unanswered_post_bodies, word_vectors):
	print "PREDICTIONS \n\n"
	for i in range(len(unanswered_post_titles)):
		unanswered_post_vector = get_post_vector(unanswered_post_titles[i], unanswered_post_bodies[i], word_vectors)
		highest_cosine_similarity = 0.0
		highest_similar_question = [None, None, None]
		for cluster in question_clusters:
			for (question, post_title, post_body, post_vector) in cluster:
				cosine_similarity = np.dot(unanswered_post_vector, post_vector)/(np.linalg.norm(unanswered_post_vector) * np.linalg.norm(post_vector))
				if cosine_similarity > highest_cosine_similarity:
					highest_cosine_similarity = cosine_similarity
					highest_similar_question = [question, post_title, post_body]	
		if highest_cosine_similarity > 0.95:
			print "UNANSWERED QUESTION:"
			print unanswered_post_titles[i].encode('utf-8')
			print unanswered_post_bodies[i].encode('utf-8')
			[question, post_title, post_body] = highest_similar_question
			print "QUESTION:"
			print question.encode('utf-8')
			print "FROM POST:"
			print post_title.encode('utf-8')
			print post_body.encode('utf-8') + "\n"

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "Usage: python cluster_questions.py <posts.xml> <comments.xml> <word_vectors> <cluster_algo>"
		sys.exit(0)
	posts_file = open(sys.argv[1], 'r')
	comments_file = open(sys.argv[2], 'r')
	word_vectors_file = open(sys.argv[3], 'r')
	cluster_algo = str(sys.argv[4])
	word_vectors = {}
	for line in word_vectors_file.readlines():
		vals = line.rstrip().split(' ')
		word_vectors[vals[0]] = map(float, vals[1:])
	questions, post_titles, post_bodies, unanswered_post_titles, unanswered_post_bodies = extract_questions(posts_file, comments_file)
	question_clusters = cluster_question_sentences(questions, post_titles, post_bodies, word_vectors, cluster_algo)
	predict_comment(question_clusters, unanswered_post_titles, unanswered_post_bodies, word_vectors)

