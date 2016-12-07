import sys
import xml.etree.ElementTree as ET
import re
from nltk.tokenize import word_tokenize
import pdb
import numpy as np
import datetime
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
import cPickle as p

SENTENCE_REDUCTION_FACTOR=0.75

def get_tokens(text):
	return word_tokenize(text.lower())

def get_question(text):
	r = re.compile(r"(http://[^ ]+)")
	text = r.sub("", text) #remove urls so that ? is not identified in urls
	text = " ".join(get_tokens(text))
	if '?' in text.split():
		text = text.split('?')[0]+ '?'
		words = text.split()
		if len(words) > 25: #ignore long comments
			return None
		return text
	return None

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

def get_similarity(a_embeddings, b_embeddings):
	avg_a_embedding = np.mean(a_embeddings, axis=0)
	avg_b_embedding = np.mean(b_embeddings, axis=0)
	cosine_similarity = np.dot(avg_a_embedding, avg_b_embedding)/(np.linalg.norm(avg_a_embedding) * np.linalg.norm(avg_b_embedding))
	return cosine_similarity

def collect_posts_log(posthistory_file, comments_file):
	posts_log = {} #{postId: [initial_post, edited_post, edit_comment, [question_comments]]
	posthistory_tree = ET.parse(posthistory_file)
	for posthistory in posthistory_tree.getroot():
		posthistory_typeid = posthistory.attrib['PostHistoryTypeId']
		postId = posthistory.attrib['PostId']
		if posthistory_typeid in ['2', '5']:
			if postId not in posts_log.keys():
				posts_log[postId] = [None, None, None, None, []]
			if posthistory_typeid == '2':
				posts_log[postId][0] = posthistory.attrib['Text']
			else:
				posts_log[postId][1] = posthistory.attrib['Text']
				posts_log[postId][2] = posthistory.attrib['Comment']
				posts_log[postId][3] = posthistory.attrib['CreationDate']
	
	comments_tree = ET.parse(comments_file)
	for comment in comments_tree.getroot():
		postId = comment.attrib['PostId']
		try:
			posts_log[postId]
		except:
			continue
		if not posts_log[postId][1]:
			continue
		comment_date_str = comment.attrib['CreationDate'].split('.')[0] #format of date e.g.:"2008-09-06T08:07:10.730" We don't want .730
		postedit_date_str = posts_log[postId][3].split('.')[0]
		comment_date = datetime.datetime.strptime(comment_date_str, "%Y-%m-%dT%H:%M:%S")
		postedit_date = datetime.datetime.strptime(postedit_date_str, "%Y-%m-%dT%H:%M:%S")
		if comment_date > postedit_date: #ignore comments posted after edit date
			continue
		text = comment.attrib['Text']
		question = get_question(text)
		if question:
			posts_log[postId][4].append(question)
			
	return posts_log

def cluster_questions(question_embeddings, cluster_algo):
	if cluster_algo == "kmeans":
		n_clusters = int(len(question_embeddings)*SENTENCE_REDUCTION_FACTOR)
		kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=0).fit(question_embeddings) #n_jobs=-1 runs #CPUs jobs in parallel
		question_labels = kmeans.labels_
	elif cluster_algo == "dbscan":
		#dbscan = DBSCAN(eps=0.65, min_samples=1, n_jobs=-1).fit(question_embeddings)
		dbscan = DBSCAN(eps=0.65, min_samples=1).fit(question_embeddings)
		question_labels = dbscan.labels_
		n_clusters = len(set(question_labels)) - (1 if -1 in question_labels else 0)
	else:
		print "Unknown cluster algo ", cluster_algo
		sys.exit(0)
	return question_labels, n_clusters

if __name__ == "__main__":
	if len(sys.argv) < 6:
		print "usage: python generate_incomplete_posts_labels.py <posthistory.xml> <comments.xml> <vectors.txt> <cluster_algo> <output_posts.p> <output_labels.p>"
		sys.exit(0)
	posthistory_file = open(sys.argv[1], 'r')
	comments_file = open(sys.argv[2], 'r')
	word_vectors_file = open(sys.argv[3], 'r')
	cluster_algo = str(sys.argv[4])
	word_vectors = {}
	for line in word_vectors_file.readlines():
		vals = line.rstrip().split(' ')
		word_vectors[vals[0]] = map(float, vals[1:])

	posts_log = collect_posts_log(posthistory_file, comments_file) # {postId: [initial_post, edited_post, edit_comment, [question_comments]]

	postIds = []
	avg_question_embeddings = []
	for postId in posts_log.keys():
		initial = posts_log[postId][0]
		final = posts_log[postId][1]
		if not final:
			continue
		if posts_log[postId][4] == []:
			continue
		delta_toks = set(get_tokens(final)) - set(get_tokens(initial))
		delta_embeddings = get_embeddings(delta_toks, word_vectors)
		max_similarity = 0.0
		delta_question = None
		delta_question_embeddings = None
		for question in posts_log[postId][4]:
			question_toks = set(get_tokens(question)) #because we are going to compare this with delta which is a set
			question_embeddings = get_embeddings(question_toks, word_vectors)
			curr_similarity = get_similarity(delta_embeddings, question_embeddings)
			if curr_similarity > max_similarity:
				delta_question = question
				max_similarity = curr_similarity
				delta_question_embeddings = question_embeddings
		if max_similarity < 0.5: #ignore comments that are not too similar
			continue
		#print initial.encode('utf-8')
		#print "----------------------------------------------------------------"
		#print final.encode('utf-8')
		#print "----------------------------------------------------------------"
		#print delta_question.encode('utf-8')
		#print max_similarity
		#print "----------------------------------------------------------------"
		#print
		#print
		posts_log[postId][4] = [delta_question]
		postIds.append(postId)
		avg_question_embeddings.append(np.mean(delta_question_embeddings, axis=0))

	question_cluster_labels, n_clusters = cluster_questions(np.asarray(avg_question_embeddings), cluster_algo)
	labels = np.asarray(question_cluster_labels)
	posts = [None]*len(labels)
	for i in range(len(labels)):
		posts[i] = posts_log[postIds[i]][0]

	posts = np.asarray(posts)
	p.dump(posts, open(sys.argv[5], 'wb'))
	p.dump(labels, open(sys.argv[6], 'wb'))

	postId_clusters = [[] for i in range(n_clusters)]
	for i, postId in enumerate(postIds):
		postId_clusters[question_cluster_labels[i]].append(postId)

	for cluster in postId_clusters:
		if len(cluster) < 4 or len(cluster) > len(posts)/4: #too few or too many items in a cluster
			continue
		for postId in cluster:
			print posts_log[postId][4][0].encode('utf-8')
		print "----------------------------------------------------------------"
			
