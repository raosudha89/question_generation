import sys
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

def main(args):
	start_time = time.time()
	print 'Parsing posts...'
	post_parser = PostParser(args.posts_xml)
	post_parser.parse()
	posts = post_parser.get_posts()
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Parsing comments...'
	comment_parser = CommentParser(args.comments_xml)
	comment_parser.parse()
	question_comments = comment_parser.get_question_comments()
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
	print 'Loading word_embeddings, vocab'	
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	vocab = p.load(open(args.vocab, 'rb'))
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Generating post_ques_ans...'
	post_ques_ans_generator = PostQuesAnsGenerator()
	post_ques_answers = post_ques_ans_generator.generate(posts, question_comments, posthistories, vocab, word_embeddings)
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Generating similar posts using BM25...'
	post_titles_as_queries = []
	postIds_as_query_ids = []
	posts_as_documents = {}
	questions_as_queries = []
	questions_as_documents = {}
	for postId in post_ques_answers.keys():
		post_titles_as_queries.append(' '.join(posts[postId].title).encode('utf-8'))
		postIds_as_query_ids.append(postId)
		posts_as_documents[postId] = ' '.join(posts[postId].title).encode('utf-8') + ' ' + ' '.join(posts[postId].body).encode('utf-8')
		questions_as_queries.append(' '.join(post_ques_answers[postId].question_comment).encode('utf-8'))
		questions_as_documents[postId] = ' '.join(post_ques_answers[postId].question_comment).encode('utf-8')

	posts_BM25 = BM25(post_titles_as_queries, postIds_as_query_ids, posts_as_documents)
	similar_posts = posts_BM25.run()
	
	questions_BM25 = BM25(questions_as_queries, postIds_as_query_ids, questions_as_documents)
	similar_questions = questions_BM25.run()

	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Generating final vectors...'	
	post_vectors = [None]*len(post_ques_answers)
	ques_list_vectors = [None]*len(post_ques_answers)
	ans_list_vectors = [None]*len(post_ques_answers)

	i = 0
	N = args.no_of_candidates
	for postId in post_ques_answers.keys():
		candidate_postIds = similar_posts[postId][:N-1]
		if not candidate_postIds:
			continue
		post_vectors[i] = get_indices(post_ques_answers[postId].post, vocab)
		ques_list_vectors[i] = [None]*N
		ques_list_vectors[i][0] = get_indices(post_ques_answers[postId].question_comment, vocab)
		ans_list_vectors[i] = [None]*N
		ans_list_vectors[i][0] = get_indices(post_ques_answers[postId].answer, vocab)
		for j in range(N-1):
			ques_list_vectors[i][j+1] = get_indices(post_ques_answers[candidate_postIds[j]].question_comment, vocab)
			ans_list_vectors[i][j+1] = get_indices(post_ques_answers[candidate_postIds[j]].answer, vocab)
		i+=1

	print 'Done! Time taken ', time.time() - start_time
	print

	p.dump(similar_posts, open('similar_posts.p', 'wb'))
	p.dump(similar_questions, open('similar_questions.p', 'wb'))
	p.dump(posts, open('posts.p', 'wb'))
	p.dump(post_ques_answers, open('post_ques_answers.p', 'wb'))
	p.dump(post_vectors, open(args.post_vectors, 'wb'))
	p.dump(ques_list_vectors, open(args.ques_list_vectors, 'wb'))
	p.dump(ans_list_vectors, open(args.ans_list_vectors, 'wb'))

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--posts_xml", type = str)
	argparser.add_argument("--comments_xml", type = str)
	argparser.add_argument("--posthistory_xml", type = str)
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--ans_list_vectors", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--no_of_candidates", type = int, default = 5)
	args = argparser.parse_args()
	print args
	print ""
	main(args)

