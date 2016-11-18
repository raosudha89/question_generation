import sys
import cPickle as p
from nltk.tokenize import word_tokenize
import pdb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from nltk.corpus import stopwords
stopword_list = stopwords.words('english')

SENTENCE_REDUCTION_FACTOR=0.75

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

def cluster_posts(post_embeddings, cluster_algo):
	if cluster_algo == "kmeans":
		n_clusters = int(len(post_embeddings)*SENTENCE_REDUCTION_FACTOR)
		kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=0).fit(post_embeddings) #n_jobs=-1 runs #CPUs jobs in parallel
		post_labels = kmeans.labels_
	elif cluster_algo == "dbscan":
		dbscan = DBSCAN(eps=0.65, min_samples=3).fit(post_embeddings)
		post_labels = dbscan.labels_
		n_clusters = len(set(post_labels)) - (1 if -1 in post_labels else 0)
	else:
		print "Unknown cluster algo ", cluster_algo
		sys.exit(0)
	return post_labels, n_clusters

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "usage: python cluster_posts.py <data_posts.p> <word_vectors.txt> <cluster_algo>"
		sys.exit(0)
	posts = p.load(open(sys.argv[1], 'rb'))
	word_vectors_file = open(sys.argv[2], 'r')
	cluster_algo = str(sys.argv[3])
	word_vectors = {}

	print 'No. of posts:', len(posts)
	for line in word_vectors_file.readlines():
		vals = line.rstrip().split(' ')
		word_vectors[vals[0]] = map(float, vals[1:])

	avg_post_embeddings = []
	for post in posts:
		post_toks = get_tokens(post)
		post_embeddings = get_embeddings(post_toks, word_vectors)
		weights = [0.5 if tok in stopword_list else 1.0 for tok in post_toks] 
		avg_post_embeddings.append(np.average(post_embeddings, axis=0, weights=weights))
		#avg_post_embeddings.append(np.average(post_embeddings, axis=0))

	post_labels, n_clusters = cluster_posts(avg_post_embeddings, cluster_algo)	
	post_clusters = [[] for i in range(n_clusters)]
	for i, post in enumerate(posts):
		post_clusters[post_labels[i]].append(posts[i])
	
	print 'No. of clusters:', n_clusters
	for i, cluster in enumerate(post_clusters):
		if len(cluster) > 2:
		#if True:	
			for post in cluster: 
				print post.encode('utf-8')
			print "----------------------------------------------------------------"
	
