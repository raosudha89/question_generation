import sys
import xml.etree.ElementTree as ET
import cPickle as p
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import numpy as np
import re
import string
import pdb

domain_words = ['duplicate', 'upvote', 'downvote', 'vote', 'related']

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

def extract_posts_questions(posts_file, comments_file):
	posts_tree = ET.parse(posts_file)
	main_post_ids = []
	main_posts = []
	
	for post in posts_tree.getroot():
		postId = post.attrib['Id']
		postTypeId = post.attrib['PostTypeId'] 
		if postTypeId == '1':
			main_post_ids.append(postId)
			main_posts.append(post.attrib['Title']+ " "+ post.attrib['Body'])

	print "No. of main posts ", len(main_post_ids)
	comments_tree = ET.parse(comments_file)
	questions = []
	posts = []
	comment_count = 0
	for comment in comments_tree.getroot():
		postId = comment.attrib['PostId']
		if postId not in main_post_ids:
			continue
		i = main_post_ids.index(postId)
		post = main_posts[i]
		comment_count += 1	
		text = comment.attrib['Text']
		question = get_question(text)
		if question:
			questions.append(question)	
			posts.append(post)
	print "No. of comments to main posts ", comment_count
	print "No of question comments ", len(questions)
	return posts, questions

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "Usage: python generate_posts_questions.py <posts.xml> <comments.xml> <output_posts.p> <output_questions.p>"
		sys.exit(0)
	posts_file = open(sys.argv[1], 'r')
	comments_file = open(sys.argv[2], 'r')
	posts, questions = extract_posts_questions(posts_file, comments_file)
	p.dump(posts, open(sys.argv[3], 'wb'))
	p.dump(questions, open(sys.argv[4], 'wb'))

