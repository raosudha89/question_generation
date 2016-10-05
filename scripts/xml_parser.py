import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import pdb

def is_question(text):
	words = word_tokenize(text)
	#question_tokens = ['?', 'who', 'when', 'where', 'why', 'how', 'what']
	#for tok in question_tokens: 
	#	if tok in words:
	#		return True
	if '?' in words:
		return True
	return False

if __name__ == "__main__":
	posts_file = open(sys.argv[1], 'r')
	comments_file = open(sys.argv[2], 'r')
	posts_tree = ET.parse(posts_file)
	main_post_ids = []
	for post in posts_tree.getroot():
		postId = post.attrib['Id']
		postTypeId = post.attrib['PostTypeId'] 
		if postTypeId == '1':
			main_post_ids.append(postId)
	comments_tree = ET.parse(comments_file)
	for comment in comments_tree.getroot():
		postId = comment.attrib['PostId']
		if postId not in main_post_ids:
			continue	
		text = comment.attrib['Text']
		if is_question(text):
			try:
				print text
			except:
				continue
