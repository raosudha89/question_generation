import sys
sys.path.insert(0, '/fs/clip-amr/BM25/src')
from query import QueryProcessor
import multiprocessing
import operator
from collections import defaultdict
import pdb

def get_similar_documents(queries, query_ids, documents): 
	proc = QueryProcessor(queries, documents)
	results = proc.run()
	i = 0
	similar_documents = defaultdict(list)
	for result in results:
		sorted_x = sorted(result.iteritems(), key=operator.itemgetter(1))
		sorted_x.reverse()
		for qid, score in sorted_x[:20]:
			similar_documents[query_ids[i]].append(qid)
		i += 1
	return similar_documents

class BM25:
	def __init__(self, queries, query_ids, documents):
		self.queries = queries
		self.query_ids = query_ids
		self.documents = documents

	def run(self):	
		num_cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool()
		count = len(self.queries)/num_cores
		tasks = [(self.queries[i*count : i*count+count], self.query_ids[i*count : i*count+count], self.documents) for i in range(num_cores)]
		tasks += [(self.queries[count*num_cores :], self.query_ids[count*num_cores :], self.documents)]
		results = [pool.apply_async(get_similar_documents, t) for t in tasks]
		similar_documents = results[0].get()
		for i in range(1, len(results)):
			similar_documents.update(results[i].get())
		return similar_documents
		
