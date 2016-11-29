__author__ = 'Nick Hirakawa'


from parse import *
import sys
sys.path.insert(0, '/fs/clip-amr/BM25/src')
from query import QueryProcessor
import operator
import pdb
from collections import defaultdict
import cPickle as p

def main():
	pp = PostParser(filename=sys.argv[1])
	pp.parse()
	posts = pp.get_posts()
	posts_as_corpus = dict()
	post_titles_as_queries = []
	post_titles_as_queries_ids = []
	for postId, post in posts.iteritems():
		if post.typeId == '1':
			posts_as_corpus[postId] = " ".join(post.title + post.body).encode('utf-8').split()
			post_titles_as_queries.append(" ".join(post.title).encode('utf-8').split())
			post_titles_as_queries_ids.append(postId)
	proc = QueryProcessor(post_titles_as_queries[:10], posts_as_corpus)
	results = proc.run()
	qid = 0
	similar_posts = defaultdict(list)
	for result in results:
		sorted_x = sorted(result.iteritems(), key=operator.itemgetter(1))
		sorted_x.reverse()
		for postId, score in sorted_x[:10]:
			#print post_titles_as_queries_ids[qid], postId, score
			similar_posts[post_titles_as_queries_ids[qid]].append(postId)
		#print 
		qid += 1
	p.dump(similar_posts, open(sys.argv[2], 'wb'))	

if __name__ == '__main__':
	main()
