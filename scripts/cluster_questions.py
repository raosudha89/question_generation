import sys
import cPickle as p
from nltk.tokenize import word_tokenize
import pdb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from nltk.corpus import stopwords
stopword_list = stopwords.words('english')

SENTENCE_REDUCTION_FACTOR=0.5

def get_tokens(text):
	return word_tokenize(text.lower())

def get_embeddings(tokens, word_vectors):
	word_vector_len = len(word_vectors[word_vectors.keys()[0]])
	embeddings = np.zeros((len(tokens), word_vector_len))
	unk = "<unk>"
	for i, token in enumerate(tokens):
		try:
			embeddings[i] = word_vectors[token]
		except:
			embeddings[i] = word_vectors[unk]
	return embeddings

def cluster_questions(question_embeddings, cluster_algo):
	if cluster_algo == "kmeans":
		n_clusters = int(len(question_embeddings)*SENTENCE_REDUCTION_FACTOR)
		kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=0).fit(question_embeddings) #n_jobs=-1 runs #CPUs jobs in parallel
		question_labels = kmeans.labels_
	elif cluster_algo == "dbscan":
		dbscan = DBSCAN(eps=0.25, min_samples=1).fit(question_embeddings)
		question_labels = dbscan.labels_
		n_clusters = len(set(question_labels)) - (1 if -1 in question_labels else 0)
	else:
		print "Unknown cluster algo ", cluster_algo
		sys.exit(0)
	return question_labels, n_clusters

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "usage: python cluster_questions.py <data_questions.p> <word_vectors.txt> <cluster_algo> <output_data_normalized_questions.p>"
		sys.exit(0)
	questions = p.load(open(sys.argv[1], 'rb'))
	word_vectors_file = open(sys.argv[2], 'r')
	cluster_algo = str(sys.argv[3])
	word_vectors = {}
	for line in word_vectors_file.readlines():
		vals = line.rstrip().split(' ')
		word_vectors[vals[0]] = map(float, vals[1:])

	avg_question_embeddings = []
	for question in questions:
		question_toks = get_tokens(question)
		question_embeddings = get_embeddings(question_toks, word_vectors)
		weights = [0.5 if tok in stopword_list else 1.0 for tok in question_toks] 
		avg_question_embeddings.append(np.average(question_embeddings, axis=0, weights=weights))

	question_labels, n_clusters = cluster_questions(avg_question_embeddings, cluster_algo)	
	question_clusters = [[] for i in range(n_clusters)]
	for i, question in enumerate(questions):
		question_clusters[question_labels[i]].append(questions[i])
	
	normalized_question_per_label = [None]*n_clusters
	for i, cluster in enumerate(question_clusters):
		#if len(cluster) > 5 and len(cluster) < len(questions)/4:
		if True:	
			for question in cluster: 
				print question.encode('utf-8')
			print "----------------------------------------------------------------"
			question_len = [len(get_tokens(q)) for q in cluster]
			normalized_question_per_label[i] = cluster[question_len.index(min(question_len))]

	normalized_questions = [None]*len(questions)
	for i in range(len(questions)):
		normalized_questions[i] = normalized_question_per_label[question_labels[i]]
	
	p.dump(np.asarray(normalized_questions), open(sys.argv[4], 'wb'))
	
