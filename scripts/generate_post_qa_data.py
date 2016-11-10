import sys
import xml.etree.ElementTree as ET
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
import pdb
import numpy as np
import datetime
import cPickle as p

domain_words = ['duplicate', 'upvote', 'downvote', 'vote']

def get_tokens(text):
	return word_tokenize(text.lower())

def get_question(text):
	r = re.compile(r"(http://[^ ]+)")
	text = r.sub("", text) #remove urls so that ? is not identified in urls
	tokens = get_tokens(text)
	if '?' in tokens:
		text = " ".join(tokens).split('?')[0]+ '?'
		words = text.split()
		if len(words) > 25: #ignore long comments
			return None
		for w in domain_words:
			if w in words:
				return None
		if words[0] == '@':
			text = " ".join(words[2:])
		else:
			text = " ".join(words)
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

def collect_posts_log(posts_file, posthistory_file, comments_file):
	main_post_ids = []
	posts_tree = ET.parse(posts_file)
	for post in posts_tree.getroot():
		postId = post.attrib['Id']
		postTypeId = post.attrib['PostTypeId'] 
		if postTypeId == '1':
			main_post_ids.append(postId)
	posts_log = {} #{postId: [initial_post, edited_post, edit_comment, [question_comments]]
	posthistory_tree = ET.parse(posthistory_file)
	for posthistory in posthistory_tree.getroot():
		posthistory_typeid = posthistory.attrib['PostHistoryTypeId']
		postId = posthistory.attrib['PostId']
		if postId not in main_post_ids: #only consider main posts
			continue
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

def get_delta(initial, final):
	orig_final = final
	s = SequenceMatcher(None, initial, final)
	m = s.find_longest_match(0, len(initial), 0, len(final))
	common = initial[m.a: m.a+m.size]
	while ' ' in initial and ' ' in common and len(common.split()) > 2:
		initial = initial[0:m.a] + ' ' + initial[m.a+m.size:]
		final = final[0:m.b] + ' ' + final[m.b+m.size:]
		s = SequenceMatcher(None, initial, final)
		m = s.find_longest_match(0, len(initial), 0, len(final))
		common = initial[m.a: m.a+m.size]
	if final == orig_final:
		return None
	return final

if __name__ == "__main__":
	if len(sys.argv) < 6:
		print "usage: python generate_post_qa_data.py <posts.xml> <posthistory.xml> <comments.xml> <vectors.txt> <output_posts.p> <output_questions.p> <output_answers.p>"
		sys.exit(0)
	posts_file = open(sys.argv[1], 'r')
	posthistory_file = open(sys.argv[2], 'r')
	comments_file = open(sys.argv[3], 'r')
	word_vectors_file = open(sys.argv[4], 'r')
	word_vectors = {}
	for line in word_vectors_file.readlines():
		vals = line.rstrip().split(' ')
		word_vectors[vals[0]] = map(float, vals[1:])

	posts_log = collect_posts_log(posts_file, posthistory_file, comments_file) # {postId: [initial_post, edited_post, edit_comment, [question_comments]]

	posts = []
	questions = []
	answers = []
	for postId in posts_log.keys():
		initial = posts_log[postId][0]
		final = posts_log[postId][1]
		if not final:
			continue
		if posts_log[postId][4] == []:
			continue
		delta = get_delta(initial, final)
		if not delta: #there is no reasonable difference
			continue
		delta_toks = delta.split()
		if len(delta_toks) < 3 or len(delta_toks) > 50 : #too few or too many words
			continue 
		delta_embeddings = get_embeddings(delta_toks, word_vectors)
		max_similarity = 0.0
		delta_question = None
		for question in posts_log[postId][4]:
			question_toks = get_tokens(question)
			question_embeddings = get_embeddings(question_toks, word_vectors)
			curr_similarity = get_similarity(delta_embeddings, question_embeddings)
			if curr_similarity > max_similarity:
				delta_question = question
				max_similarity = curr_similarity
		if max_similarity < 0.5: #ignore comments that are not too similar
			continue
		print initial.encode('utf-8')
		print "----------------------------------------------------------------"
		print final.encode('utf-8')
		print "----------------------------------------------------------------"
		print delta_question.encode('utf-8')
		print "----------------------------------------------------------------"
		print ' '.join(delta_toks).encode('utf-8')
		print max_similarity
		print "----------------------------------------------------------------"
		print
		print
		posts.append(initial)
		questions.append(delta_question)
		answers.append(' '.join(delta_toks))

	p.dump(posts, open(sys.argv[5], 'wb'))
	p.dump(questions, open(sys.argv[6], 'wb'))
	p.dump(answers, open(sys.argv[7], 'wb'))
