from invdx import build_data_structures
from rank import score_BM25
import operator


class QueryProcessor:
	def __init__(self, queries, corpus):
		self.queries = queries
		self.index, self.dlt = build_data_structures(corpus)

	def run(self):
		results = []
		for query in self.queries:  # for each query
			results.append(self.run_query(query))
		return results

	def run_query(self, query):
		query_result = dict()
		for term in query:  # for each word in query
			if term in self.index:  # doc words in `self.index`, self.index[word][docid]
				doc_dict = self.index[term]  # retrieve index entry
				"""
				for each document and its (the current query word's) word frequency, 
				"""
				for docid, freq in doc_dict.items():
					score = score_BM25(
						n=len(doc_dict),
						f=freq,
						qf=1,
						r=0,
						N=len(self.dlt),
						dl=self.dlt.get_length(docid),
						avdl=self.dlt.get_average_length())  # calculate score
					if docid in query_result:  # this document has already been scored once
						query_result[docid] += score
					else:
						query_result[docid] = score
		"""For current query word, each doc score."""
		return query_result