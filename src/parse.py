import re


class QueryParser:

	def __init__(self, filename):
		self.filename = filename
		self.queries = []

	def parse(self):
		with open(self.filename) as f:
			lines = ''.join(f.readlines())
		self.queries = [x.rstrip().split() for x in lines.split('\n')[:-1]]
		"""
		print(self.queries): 
			[['portabl', 'oper', 'system'], 
			['code', 'optim', 'for', 'space', 'effici'], 
			['parallel', 'algorithm'], 
			['distribut', 'comput', 'structur', 'and', 'algorithm'], 
			['appli', 'stochast', 'process'], 
			['perform', 'evalu', 'and', 'model', 'of', 'comput', 'system'], 
			['parallel', 'processor', 'in', 'inform', 'retriev']]
		"""

	def get_queries(self):
		return self.queries


class CorpusParser:

	def __init__(self, filename):
		self.filename = filename
		self.regex = re.compile('^#\s*\d+')
		self.corpus = dict()

	def parse(self):
		with open(self.filename) as f:
			s = ''.join(f.readlines())
		blobs = s.split('#')[1:]
		for x in blobs:
			text = x.split()
			docid = text.pop(0)
			self.corpus[docid] = text

		"""
		print(self.corpus["1"]):
			['preliminari', 'report', 'intern', 'algebra', 'languag', 'cacm', 'decemb', '1958', 'perli', 'a', 'j', 
			'samelson', 'k', 'ca581203', 'jb', 'march', '22', '1978', '8', '28', 'pm', '100', '5', '1', '123', '5', 
			'1', '164', '5', '1', '1', '5', '1', '1', '5', '1', '1', '5', '1', '205', '5', '1', '210', '5', '1', 
			'214', '5', '1', '1982', '5', '1', '398', '5', '1', '642', '5', '1', '669', '5', '1', '1', '6', '1', 
			'1', '6', '1', '1', '6', '1', '1', '6', '1', '1', '6', '1', '1', '6', '1', '1', '6', '1', '1', '6', '1', 
			'1', '6', '1', '1', '6', '1', '165', '6', '1', '196', '6', '1', '196', '6', '1', '1273', '6', '1', '1883', 
			'6', '1', '324', '6', '1', '43', '6', '1', '53', '6', '1', '91', '6', '1', '410', '6', '1', '3184', '6', '1']
		"""

	def get_corpus(self):
		return self.corpus


if __name__ == '__main__':
	qp = QueryParser('text/queries.txt')
	print(qp.get_queries())