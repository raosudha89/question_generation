__author__ = 'Nick Hirakawa'


from parse import *
import argparse
import sys
import time
sys.path.insert(0, '/fs/clip-amr/BM25/src')
from query import QueryProcessor
import operator
import pdb
from collections import defaultdict
import cPickle as p
import multiprocessing

def get_similar_posts(post_titles_as_queries, post_titles_as_queries_ids, posts_as_corpus): 
	proc = QueryProcessor(post_titles_as_queries, posts_as_corpus)
	results = proc.run()
	qid = 0
	similar_posts = defaultdict(list)
	for result in results:
		sorted_x = sorted(result.iteritems(), key=operator.itemgetter(1))
		sorted_x.reverse()
		for postId, score in sorted_x[:20]:
			#print post_titles_as_queries_ids[qid], postId, score
			similar_posts[post_titles_as_queries_ids[qid]].append(postId)
		#print 
		qid += 1
	return similar_posts

def main():
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
	print 'Generating similar posts using BM25...'
	posts_as_corpus = dict()
	post_titles_as_queries = []
	post_titles_as_queries_ids = []
	for postId in posthistories.keys():
		try:
			posts[postId]
		except:
			continue # post is not a main post
		if posts[postId].typeId == '1':
			posts_as_corpus[postId] = " ".join(posts[postId].title + posts[postId].body).encode('utf-8').split()
			post_titles_as_queries.append(" ".join(posts[postId].title).encode('utf-8').split())
			post_titles_as_queries_ids.append(postId)
	num_cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool()
	count = len(post_titles_as_queries)/num_cores
	tasks = [(post_titles_as_queries[i*count : i*count+count], post_titles_as_queries_ids[i*count : i*count+count], posts_as_corpus) for i in range(num_cores)]
	results = [pool.apply_async(get_similar_posts, t) for t in tasks]
	similar_posts = results[0].get()
	for i in range(1, len(results)):
		similar_posts.update(results[i].get())
	p.dump(similar_posts, open(args.similar_posts, 'wb'))	
	print 'Done! Time taken ', time.time() - start_time
	print

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--posts_xml", type = str)
	argparser.add_argument("--posthistory_xml", type = str)
	argparser.add_argument("--similar_posts", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main()
