import sys, os
import xml.etree.ElementTree as ET
import re
import pdb
import random

def read_raw_posts(raw_post_filename):
	raw_posts = {}
	raw_titles = {}
	with open(raw_post_filename) as h:
		i = 0
		for l in h.readlines():
			if i == 0 or i == 1 or '</posthistory>' in l:
				i += 1
				continue
			m = re.search(' PostId="([0-9]+)"', l)
			if m is None:
				raise Exception('re match failure on: ', l)
			post_id = int(m.group(1))
			m = re.search(' PostHistoryTypeId="([0-9]+)"', l)
			if m is None:
				raise Exception('re match failure on: ', l)
			post_type_id = int(m.group(1))
			if post_type_id in [1, 2]:  # initial title or initial post
				m = re.search(' Text="([^"]*)"', l)
				if m is None:
					raise Exception('re match failure on: ', l)
				if post_type_id == 1:
					raw_titles[post_id] = m.group(1)
				elif post_type_id == 2:
					raw_posts[post_id] = m.group(1)
	return raw_posts, raw_titles

def shuffle(q, r):
	shuffled_q = [None]*10
	shuffled_r = [None]*10
	indexes = range(10)
	random.shuffle(indexes)
	for j, index in enumerate(indexes):
		shuffled_q[j] = q[index]
		shuffled_r[j] = r[index]
			
	return shuffled_q, shuffled_r

if __name__ == '__main__':
	dataset_file = open(sys.argv[1], 'r')
	posthistory_xml_file = sys.argv[2]
	output_post_tsv_file = open(sys.argv[3], 'w')
	output_qa_tsv_file = open(sys.argv[4], 'w')
	
	data_questions = {}
	data_answers = {}
	main_question = False
	no_long_posts = 0
	for line in dataset_file.readlines():
		if 'Id:' in line:
			post_id = line.split(':')[1].strip('\n').strip()
		if 'Post:' in line:
			post = line.split(':', 1)[1].strip('\n').strip()
			if len(post.split()) > 300:
				no_long_posts += 1
				post_id = None
		if line.strip('\n').strip() == '':
			main_question = True
		if 'Question:' in line and post_id and main_question:
			try:
				data_questions[post_id]
			except:
				data_questions[post_id] = []
			question = line.split(':', 1)[1].strip('\n')
			data_questions[post_id].append(question)
		if 'Answer:' in line and post_id and main_question:
			try:
				data_answers[post_id]
			except:
				data_answers[post_id] = []
			answer = line.split(':', 1)[1].strip('\n')
			data_answers[post_id].append(answer)
			main_question = False

	print len(data_questions)
	print 'No. of long posts %d' % no_long_posts
	raw_posts, raw_titles = read_raw_posts(posthistory_xml_file)
	
	output_post_tsv_file.write('%s\t %s\t %s\n' % ("postId", "title", "post"))
	output_qa_tsv_file.write('%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t \
							       %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\n' % \
					  ("post_id","q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
								 "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"))

	for post_id in data_questions.keys():
		try:
			title = raw_titles[int(post_id)].replace('\t', ' ')
			post = raw_posts[int(post_id)].replace('\t', ' ').replace('\n', '</br>')
		except:
			pdb.set_trace()
		q = data_questions[post_id]
		a = data_answers[post_id]
		output_post_tsv_file.write('%s\t %s\t %s\n' % (post_id, title, post))
		output_qa_tsv_file.write('%s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t \
									   %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\t %s\n' % \
						  (post_id, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], \
									a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]))

