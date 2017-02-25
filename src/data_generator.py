import sys, os
import argparse
from parse import *
from post_ques_ans_generator import * 
from helper import *
import time
import numpy as np
import cPickle as p
import pdb
import random

def generate_utility_vectors(posts, posthistories, post_ques_answers, users, junior_max_rep, senior_min_rep, vocab, args):
	post_vectors = []
	post_sent_vectors = []
	post_ids = []
	labels = []
	for postId in posts:
		if postId in posthistories:
			if postId in post_ques_answers:
				post_vectors.append(get_indices(posts[postId].title + post_ques_answers[postId].post, vocab))
				post_sent_vectors.append(get_sent_vectors(posts[postId].title + post_ques_answers[postId].post, vocab))
				post_ids.append(postId)
				labels.append(0)
				post_vectors.append(get_indices(posts[postId].title + post_ques_answers[postId].post + post_ques_answers[postId].answer, vocab))
				post_sent_vectors.append(get_sent_vectors(posts[postId].title + post_ques_answers[postId].post + post_ques_answers[postId].answer, vocab))
				post_ids.append(postId)
				labels.append(1)
		else:
			if posts[postId].typeId == 1: #only main posts
				if not posts[postId].owner_userId: #author ID missing
					continue
				#posts that don't get a response and are written by "junior SE users"
				if posts[postId].answer_count == 0 and users[posts[postId].owner_userId].reputation < junior_max_rep:
					post_vectors.append(get_indices(posts[postId].title + posts[postId].body, vocab))
					post_sent_vectors.append(get_sent_vectors(posts[postId].title + posts[postId].body, vocab))
					post_ids.append(postId)
					labels.append(0)
				#posts that DO get a response and are written by "senior SE users"
				if posts[postId].accepted_answerId != None and \
						posts[postId].answer_count > 0 and \
							users[posts[postId].owner_userId].reputation > senior_min_rep:
					post_vectors.append(get_indices(posts[postId].title + posts[postId].body, vocab))
					post_sent_vectors.append(get_sent_vectors(posts[postId].title + posts[postId].body, vocab))
					post_ids.append(postId)
					labels.append(1)
	print "Size: ", len(post_vectors)
	p.dump(post_vectors, open(args.utility_post_vectors, 'wb'))
	p.dump(post_sent_vectors, open(args.utility_post_sent_vectors, 'wb'))
	p.dump(labels, open(args.utility_labels, 'wb'))
	p.dump(post_ids, open(args.utility_post_ids, 'wb'))

def get_sent_vectors(sents, vocab):
	sent_vectors = [None]*len(sents)
	for i, sent in enumerate(sents):
		sent_vectors[i] = get_indices(sent, vocab)
	return sent_vectors

def get_similar_docs(lucene_similar_docs):
	lucene_similar_docs_file = open(lucene_similar_docs, 'r')
	similar_docs = {}
	for line in lucene_similar_docs_file.readlines():
		parts = line.split()
		if len(parts) > 1:
			similar_docs[parts[0]] = parts[1:]
		else:
			similar_docs[parts[0]] = []
	return similar_docs

def generate_neural_vectors(post_ques_answers, posts, lucene_similar_posts, lucene_similar_questions, vocab, args):
	lucene_similar_posts = get_similar_docs(lucene_similar_posts)
	lucene_similar_questions = get_similar_docs(lucene_similar_questions)
	post_vectors = []
	post_sent_vectors = []
	ques_list_vectors = []
	ans_list_vectors = []
	post_ids = []
	N = args.no_of_candidates
	out_file = open(os.path.join(os.path.dirname(args.post_vectors), "lucene_post_ques_ans_list.log"), 'w') 
	for postId in lucene_similar_posts:
		candidate_postIds = lucene_similar_posts[postId]
		if len(candidate_postIds) < N:
			continue
		candidate_postIds[0] = postId #making sure first is the original post itself
		ques_list = [None]*N
		k = 0
		for j in range(len(candidate_postIds)):
			question_candidate_postIds = lucene_similar_questions[candidate_postIds[j]]
			if len(question_candidate_postIds) < N:
				continue
			question_candidate_postIds[0] = candidate_postIds[j]
			ques_list[k] = [None]*N
			for m in range(N):
				ques_list[k][m] = question_candidate_postIds[m]
			k += 1
			if k == N:
				break
		if k != N:
			continue
		post_ids.append(postId)
		post_vectors.append(get_indices(posts[postId].title + post_ques_answers[postId].post, vocab))
		post_sent_vectors.append(get_sent_vectors([posts[postId].title] + post_ques_answers[postId].post_sents, vocab))
		out_file.write("Id: " + str(postId) + '\n')
		out_file.write("Post: " + ' '.join(posts[postId].title) + ' ' + ' '.join(post_ques_answers[postId].post) + '\n\n')
		ques_list_vector = [None]*N
		ans_list_vector = [None]*N
		for k in range(N):
			ques_list_vector[k] = [None]*N
			ans_list_vector[k] = [None]*N
			for m in range(N):
				ques_list_vector[k][m] = get_indices(post_ques_answers[ques_list[k][m]].question_comment, vocab)
				out_file.write("Question: " + ' '.join(post_ques_answers[ques_list[k][m]].question_comment) + '\n')
				ans_list_vector[k][m] = get_indices(post_ques_answers[ques_list[k][m]].answer, vocab)
				out_file.write("Answer: " + ' '.join(post_ques_answers[ques_list[k][m]].answer) + '\n')
			out_file.write('\n')
		out_file.write('\n\n')
		ques_list_vectors.append(ques_list_vector)
		ans_list_vectors.append(ans_list_vector)

	print "Size: ", len(post_vectors)
	p.dump(post_ids, open(args.post_ids, 'wb'))
	p.dump(post_vectors, open(args.post_vectors, 'wb'))
	p.dump(post_sent_vectors, open(args.post_sent_vectors, 'wb'))
	p.dump(ques_list_vectors, open(args.ques_list_vectors, 'wb'))
	p.dump(ans_list_vectors, open(args.ans_list_vectors, 'wb'))

def write_data_log(post_ques_answers, posts, lucene_similar_posts, args):
	lucene_similar_posts = get_similar_posts(lucene_similar_posts)
	out_file = open(os.path.join(os.path.dirname(args.post_vectors), "lucene_post_ques_ans_list.log"), 'w') 
	N = args.no_of_candidates
	for postId in lucene_similar_posts:
		candidate_postIds = lucene_similar_posts[postId][:N]
		if len(candidate_postIds) < N:
			continue
		out_file.write("Id: " + str(postId) + '\n')
		out_file.write("Post: " + ' '.join(posts[postId].title) + ' ' + ' '.join(post_ques_answers[postId].post) + '\n\n')
		for j in range(N):
			out_file.write("Question: " + ' '.join(post_ques_answers[candidate_postIds[j]].question_comment) + '\n')
			out_file.write("Answer: " + ' '.join(post_ques_answers[candidate_postIds[j]].answer) + '\n')
		out_file.write('\n\n')

def generate_docs_for_lucene(post_ques_answers, posts, output_dir):
	for postId in post_ques_answers:
		f = open(os.path.join(output_dir, str(postId) + '.txt'), 'w')
		content = ' '.join(posts[postId].title).encode('utf-8') + ' ' + ' '.join(posts[postId].body).encode('utf-8')
		f.write(content)
		f.close()
		
def generate_ques_docs_for_lucene(post_ques_answers, output_dir):
	for postId in post_ques_answers:
		f = open(os.path.join(output_dir, str(postId) + '.txt'), 'w')
		content = ' '.join(post_ques_answers[postId].question_comment).encode('utf-8') 
		f.write(content)
		f.close()

def main(args):
	start_time = time.time()
	print 'Parsing posts...'
	post_parser = PostParser(args.posts_xml)
	post_parser.parse()
	posts = post_parser.get_posts()
	print 'Size: ', len(posts)
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Parsing posthistories...'
	posthistory_parser = PostHistoryParser(args.posthistory_xml)
	posthistory_parser.parse()
	posthistories = posthistory_parser.get_posthistories()
	print 'Size: ', len(posthistories)
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Loading vocab'	
	vocab = p.load(open(args.vocab, 'rb'))
	print 'Done! Time taken ', time.time() - start_time
	print
	
	start_time = time.time()
	print 'Parsing question comments...'
	comment_parser = CommentParser(args.comments_xml)
	# comment_parser.parse_all_comments()
	# question_comments = comment_parser.get_question_comments()
	# print 'Size: ', len(question_comments)
	
	comment_parser.parse_first_comment()
	question_comment = comment_parser.get_question_comment()
	print 'Size: ', len(question_comment)
	
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Loading word_embeddings'	
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	print 'Done! Time taken ', time.time() - start_time
	print

	# start_time = time.time()
	# print 'Parsing users...'
	# user_parser = UserParser(args.users_xml)
	# user_parser.parse()
	# users = user_parser.get_users()
	# junior_max_reputation, senior_min_reputation = user_parser.get_junior_senior_reputations()
	# print 'Size: ', len(users)
	# print 'Done! Time taken ', time.time() - start_time
	# print

	start_time = time.time()
	print 'Generating post_ques_ans...'
	post_ques_ans_generator = PostQuesAnsGenerator()
	# post_ques_answers = post_ques_ans_generator.generate(posts, question_comments, posthistories, vocab, word_embeddings)
	post_ques_answers = post_ques_ans_generator.generate(posts, question_comment, posthistories, vocab, word_embeddings)
	print 'Size: ', len(post_ques_answers)
	print 'Done! Time taken ', time.time() - start_time
	print
	
	# post_ques_ans_log_file = open(args.post_ques_ans_log, 'w')
	# for postId in post_ques_answers:
	# 	post_ques_ans_log_file.write('Id: %s\n' % postId)
	# 	post_ques_ans_log_file.write('Post: %s\n' % ' '.join(post_ques_answers[postId].post))
	# 	post_ques_ans_log_file.write('Ques: %s\n' % ' '.join(post_ques_answers[postId].question_comment))
	# 	post_ques_ans_log_file.write('Ans: %s\n\n' % ' '.join(post_ques_answers[postId].answer))
	
	generate_docs_for_lucene(post_ques_answers, posts, args.lucene_docs_dir)
	os.system('cd /fs/clip-amr/lucene && sh run_lucene.sh ' + os.path.dirname(args.post_vectors))

	generate_ques_docs_for_lucene(post_ques_answers, args.lucene_ques_docs_dir)
	os.system('cd /fs/clip-amr/lucene && sh run_ques_lucene.sh ' + os.path.dirname(args.post_vectors))

	start_time = time.time()
	print 'Generating neural vectors...'	
	generate_neural_vectors(post_ques_answers, posts, args.lucene_similar_posts, \
							args.lucene_similar_questions, vocab, args)
	print 'Done! Time taken ', time.time() - start_time
	print

	# start_time = time.time()
	# print 'Generating utility vectors'
	# generate_utility_vectors(posts, posthistories, post_ques_answers, users, junior_max_reputation, senior_min_reputation, vocab, args)
	# print 'Done! Time taken ', time.time() - start_time
	# print

	# start_time = time.time()
	# print 'Writing data log...'
	# write_data_log(post_ques_answers, posts, args.lucene_similar_posts, args)
	# print 'Done! Time taken ', time.time() - start_time
	# print

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--posts_xml", type = str)
	argparser.add_argument("--comments_xml", type = str)
	argparser.add_argument("--posthistory_xml", type = str)
	argparser.add_argument("--users_xml", type = str)
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--post_sent_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--ans_list_vectors", type = str)
	argparser.add_argument("--post_ids", type = str)
	argparser.add_argument("--utility_post_vectors", type = str)
	argparser.add_argument("--utility_post_sent_vectors", type = str)
	argparser.add_argument("--utility_labels", type = str)
	argparser.add_argument("--utility_post_ids", type = str)
	argparser.add_argument("--utility_ans_list_vectors", type = str)
	argparser.add_argument("--lucene_docs_dir", type = str)
	argparser.add_argument("--lucene_ques_docs_dir", type = str)	
	argparser.add_argument("--lucene_similar_posts", type = str)
	argparser.add_argument("--lucene_similar_questions", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--site_name", type = str)
	argparser.add_argument("--post_ques_ans_log", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)

