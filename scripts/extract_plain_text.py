import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import pdb

def preprocess(text):
	return " ".join(word_tokenize(text.lower()))

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "usage: python extract_plain_text.py <Posts.xml> <Comments.xml> <output_plain_text.txt>"
		sys.exit(0)
	posts_file = open(sys.argv[1], 'r')
	comments_file = open(sys.argv[2], 'r')
	plain_text_file = open(sys.argv[3], 'w')
	posts_tree = ET.parse(posts_file)
	for post in posts_tree.getroot():
		if post.attrib.has_key('Title'):
			title = post.attrib['Title']
		body = post.attrib['Body']
		title = preprocess(title)
		plain_text_file.write(title.encode('utf-8') + " ")
		body = preprocess(body)
		plain_text_file.write(body.encode('utf-8') + " ")
	comments_tree = ET.parse(comments_file)
	for comment in comments_tree.getroot():
		text = comment.attrib['Text']
		text = preprocess(text)
		plain_text_file.write(text.encode('utf-8') + " ")
