import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
#import theano, lasagne
import numpy as np

def preprocess(text):
	return " ".join(word_tokenize(text.lower()))

def is_question(text):
	r = re.compile(r"(http://[^ ]+)")
	text = r.sub("", text) #remove urls so that ? is not identified in urls
	text = preprocess(text)
	words = text.split()
	if 'duplicate' in words:
		return False
	if '?' in words:
		return True
	return False

def generate_data(posts_tree, comments_tree):
	post_comments_dict = {}
	for comment in comments_tree.getroot():
		postId = comment.attrib['PostId']
		post_comments_dict[postId] = comment
	data = []
	total_post_len = 0
	for post in posts_tree.getroot():
		postId = post.attrib['Id']
		postTypeId = post.attrib['PostTypeId']
		if postTypeId == '1': #it is a main post
			is_complete = True
			if post.attrib['CommentCount'] != '0': #if no comments then assume complete
				for comment in post_comments_dict[postId]:
					text = comment.attrib['Text']
					if is_question(text):
						is_complete = False
						continue
			title = post.attrib['Title']
			body = post.attrib['Body']
			post_content = preprocess(title) + " " + preprocess(body)
			total_post_len += len(post_content.split())
			if is_complete:
				data.append([post_content, 1])
			else:
				data.append([post_content, 0])
	print "Avg post len ", total_post_len*1.0/len(data)
	return data	

def iterate_minibatches(data, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(len(data))
		np.random.shuffle(indices)
	for start_idx in range(0, len(data) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield data[excerpt]	

if __name__ == "__main__":
	posts_file = open(sys.argv[1], 'r')
	posts_tree = ET.parse(posts_file)
	comments_file = open(sys.argv[2], 'r')
	comments_tree = ET.parse(comments_file)
	data = generate_data(posts_tree, comments_tree)
	num_data = len(data)
	train, dev = data[:int(num_data*0.8)], data[int(num_data*0.8):]
	

