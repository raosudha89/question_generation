import sys, os
import argparse
from parse import *
from post_ques_ans_generator import * 
from helper import *
from BM25 import *
import time
import numpy as np
import cPickle as p
import pdb
import random

def generate_utility_vectors(posts, posthistories, vocab, args):
	N = len(posts) + len(posthistories)
	post_vectors = [None]*N
	labels = [0]*N
	i = 0
	for postId in posts:
		if postId in posthistories:
			post_vectors[i] = get_indices(posts[postId].title + posthistories[postId].initial_post, vocab)
			i += 1
			post_vectors[i] = get_indices(posts[postId].title + posthistories[postId].edited_post, vocab)
			labels[i] = 1
			i += 1
		else:
			post_vectors[i] = get_indices(posts[postId].title + posts[postId].body, vocab)
			labels[i] = 1
			i += 1
	p.dump(post_vectors, open(args.utility_post_vectors, 'wb'))
	p.dump(labels, open(args.utility_labels, 'wb'))

def get_similar_posts(lucene_similar_posts):
	lucene_similar_posts_file = open(lucene_similar_posts, 'r')
	similar_posts = {}
	for line in lucene_similar_posts_file.readlines():
		parts = line.split()
		if len(parts) > 1:
			similar_posts[parts[0]] = parts[1:]
		else:
			similar_posts[parts[0]] = []
	return similar_posts

def generate_neural_vectors(post_ques_answers, lucene_similar_posts, vocab, args):
	lucene_similar_posts = get_similar_posts(lucene_similar_posts)

	start_time = time.time()
	print 'Generating final vectors...'	
	post_vectors = []
	ques_list_vectors = []
	ans_list_vectors = []
	N = args.no_of_candidates
	for postId in lucene_similar_posts:
		candidate_postIds = lucene_similar_posts[postId][:N]
		if len(candidate_postIds) < N:
			continue
		post_vectors.append(get_indices(post_ques_answers[postId].post, vocab))
		ques_list = [None]*N
		ans_list = [None]*N
		for j in range(N):
			ques_list[j] = get_indices(post_ques_answers[candidate_postIds[j]].question_comment, vocab)
			ans_list[j] = get_indices(post_ques_answers[candidate_postIds[j]].answer, vocab)
		ques_list_vectors.append(ques_list)
		ans_list_vectors.append(ans_list)

	print 'Done! Time taken ', time.time() - start_time
	print 'Size ', len(post_vectors)
	print

	p.dump(post_vectors, open(args.post_vectors, 'wb'))
	p.dump(ques_list_vectors, open(args.ques_list_vectors, 'wb'))
	p.dump(ans_list_vectors, open(args.ans_list_vectors, 'wb'))

def write_data_log(post_ques_answers, lucene_similar_posts, args):
	lucene_similar_posts = get_similar_posts(lucene_similar_posts)

	start_time = time.time()
	print 'Writing data log...'
	out_file = open(os.path.join(os.path.dirname(args.post_vectors), "lucene_post_ques_ans_list.log"), 'w') 
	N = args.no_of_candidates
	for postId in lucene_similar_posts:
		candidate_postIds = lucene_similar_posts[postId][:N]
		if len(candidate_postIds) < N:
			continue
		out_file.write("Post: " + ' '.join(post_ques_answers[postId].post) + '\n\n')
		for j in range(N):
			out_file.write("Post: " + ' '.join(post_ques_answers[candidate_postIds[j]].post) + '\n')
			out_file.write("Question: " + ' '.join(post_ques_answers[candidate_postIds[j]].question_comment) + '\n')
			out_file.write("Answer: " + ' '.join(post_ques_answers[candidate_postIds[j]].answer) + '\n\n')
		out_file.write('\n\n')
	print 'Done! Time taken ', time.time() - start_time
	print

def generate_docs_for_lucene(post_ques_answers, posts, output_dir):
	for postId in post_ques_answers:
		f = open(os.path.join(output_dir, str(postId) + '.txt'), 'w')
		content = ' '.join(posts[postId].title).encode('utf-8') + ' ' + ' '.join(posts[postId].body).encode('utf-8')
		f.write(content)
		f.close()

def main(args):
	start_time = time.time()
	print 'Parsing posts...'
	post_parser = PostParser(args.posts_xml)
	post_parser.parse()
	posts = post_parser.get_posts()
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Parsing posthistories...'
	posthistory_parser = PostHistoryParser(args.posthistory_xml)
	posthistory_parser.parse()
	posthistories = posthistory_parser.get_posthistories()
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Loading vocab'	
	vocab = p.load(open(args.vocab, 'rb'))
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Generating utility vectors and labels'
	generate_utility_vectors(posts, posthistories, vocab, args)
	print 'Done! Time taken ', time.time() - start_time
	print
	
	return

	start_time = time.time()
	print 'Parsing comments...'
	comment_parser = CommentParser(args.comments_xml)
	comment_parser.parse()
	question_comments = comment_parser.get_question_comments()
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Loading word_embeddings'	
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Generating post_ques_ans...'
	post_ques_ans_generator = PostQuesAnsGenerator()
	post_ques_answers = post_ques_ans_generator.generate(posts, question_comments, posthistories, vocab, word_embeddings)
	print 'Done! Time taken ', time.time() - start_time
	print

	generate_docs_for_lucene(post_ques_answers, posts, args.lucene_docs_dir)
	os.system('cd /fs/clip-amr/lucene && sh run_lucene.sh ' + args.site_name)
	generate_neural_vectors(post_ques_answers, args.lucene_similar_posts, vocab, args)
	write_data_log(post_ques_answers, args.lucene_similar_posts, args)

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--posts_xml", type = str)
	argparser.add_argument("--comments_xml", type = str)
	argparser.add_argument("--posthistory_xml", type = str)
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--ans_list_vectors", type = str)
	argparser.add_argument("--utility_post_vectors", type = str)
	argparser.add_argument("--utility_labels", type = str)
	argparser.add_argument("--lucene_docs_dir", type = str)	
	argparser.add_argument("--lucene_similar_posts", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--no_of_candidates", type = int, default = 20)
	argparser.add_argument("--site_name", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)

