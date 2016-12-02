import sys
import cPickle as p
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
import datetime
import pdb
from helper import *

class Post:

	def __init__(self, title, body, typeId):
		self.title = title
		self.body = body
		self.typeId = typeId

class PostParser:
	
	def __init__(self, filename):
		self.filename = filename
		self.posts = dict()
	
	def parse(self):
		posts_tree = ET.parse(self.filename)
		for post in posts_tree.getroot():
			postId = post.attrib['Id']
			postTypeId = post.attrib['PostTypeId']
			try:
				title = get_tokens(post.attrib['Title'])
			except:
				title = []
			body = get_tokens(post.attrib['Body'])
			self.posts[postId] = Post(title, body, postTypeId)

	def get_posts(self):
		return self.posts

class QuestionComment:

	def __init__(self, text, creation_date):
		self.text = text
		self.creation_date = creation_date

class CommentParser:

	def __init__(self, filename):
		self.filename = filename
		self.question_comments = defaultdict(list)

	def domain_words(self):
		return ['duplicate', 'upvote', 'downvote', 'vote', 'related']

	def get_question(self, text):
		r = re.compile(r"(http://[^ ]+)")
		text = r.sub("", text) #remove urls so that ? is not identified in urls
		tokens = get_tokens(text)
		if '?' in tokens:
			text = " ".join(tokens).split('?')[0]+ '?'
			words = text.split()
			if len(words) > 25: #ignore long comments
				return None
			for w in self.domain_words():
				if w in words:
					return None
			if words[0] == '@':
				text = words[2:]
			else:
				text = words
			return text
		return None

	def parse(self):
		comments_tree = ET.parse(self.filename)
		for comment in comments_tree.getroot():
			postId = comment.attrib['PostId']
			text = comment.attrib['Text']
			question = self.get_question(text)
			if question:
				creation_date = datetime.datetime.strptime(comment.attrib['CreationDate'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
				question_comment = QuestionComment(question, creation_date)
				self.question_comments[postId].append(question_comment)

	def get_question_comments(self):
		return self.question_comments

class PostHistory:
	def __init__(self):
		self.initial_post = None
		self.edited_post = None
		self.edit_comment = None
		self.edit_date = None

class PostHistoryParser:

	def __init__(self, filename):
		self.filename = filename
		self.posthistories = defaultdict(PostHistory)

	def parse(self):
		posthistory_tree = ET.parse(self.filename)
		for posthistory in posthistory_tree.getroot():
			posthistory_typeid = posthistory.attrib['PostHistoryTypeId']
			postId = posthistory.attrib['PostId']
			if posthistory_typeid == '2':
				self.posthistories[postId].initial_post = get_tokens(posthistory.attrib['Text'])
			elif posthistory_typeid == '5':
				self.posthistories[postId].edited_post = get_tokens(posthistory.attrib['Text'])
				self.posthistories[postId].edit_comment = get_tokens(posthistory.attrib['Comment'])
				self.posthistories[postId].edit_date = datetime.datetime.strptime(posthistory.attrib['CreationDate'].split('.')[0], "%Y-%m-%dT%H:%M:%S") 
							#format of date e.g.:"2008-09-06T08:07:10.730" We don't want .730
		for postId in self.posthistories.keys():
			if not self.posthistories[postId].edited_post:
				del self.posthistories[postId]
		
	def get_posthistories(self):
		return self.posthistories

if __name__ == "__main__":
	#post_parser = PostParser(filename=sys.argv[1])
	#post_parser.parse()
	#posts = post_parser.get_posts()
	#comment_parser = CommentParser(filename=sys.argv[1])
	#comment_parser.parse()
	#question_comments = comment_parser.get_question_comments()
	posthistory_parser = PostHistoryParser(filename=sys.argv[1])
	posthistory_parser.parse()
	posthistories = posthistory_parser.get_posthistories()
	pdb.set_trace()
