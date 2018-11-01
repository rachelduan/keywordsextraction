"""
简单无向图textrank,选用所有词性分词实现
self_textrank_analyse
"""
import pandas as pd
import numpy as np
import jieba

class TextRank():
	def __init__(self, sentence, width, d, stopwords_path):
		self.window_width = width
		self.sentence = sentence
		self.d = d
		self.stopwords = [word.strip() for word in open(stopwords_path, encoding = 'utf-8')]
		self.vertices = {}

	def cut(self):
		words_seg = jieba.lcut(self.sentence)
		self.words = [word for word in words_seg if word not in self.stopwords]

	def vertices_dict(self):
		word_list_len = len(self.words)
		for index, word in enumerate(self.words):
			
			linked_vertices = []

			left_margin = index-self.window_width
			if left_margin < 0:left_margin = 0
			right_margin = index+self.window_width+1
			if right_margin > word_list_len: right_margin = word_list_len

			linked_vertices = linked_vertices + self.words[left_margin:index] + self.words[index+1:right_margin]

			if word not in self.vertices.keys():
				self.vertices[word] = linked_vertices
			else:
				self.vertices[word] = self.vertices[word] + linked_vertices

	def edge_array(self):
		vertices_list = list(self.vertices.keys())
		vertices_list_len = len(vertices_list)
		
		self.edge = np.zeros([vertices_list_len, vertices_list_len])

		edge_sum = 0
		for index, vertice in enumerate(vertices_list):
			linked = self.vertices[vertice]
			for v in linked:
				self.edge[index, vertices_list.index(v)] += 1
				edge_sum += 1
		
		self.edge = self.edge/edge_sum

	def cal_text_rank(self):
		max_interate_times = 50
		key_len = len(self.vertices.keys())
		self.text_score = np.ones([key_len, 1])
		for iter_times in range(0,max_interate_times):
			self.text_score = 1-self.d + self.d * np.dot(self.edge, self.text_score)
		
		text_rank_index_ascend = np.argsort(np.reshape(self.text_score, key_len)).tolist()
		rank_descend = text_rank_index_ascend[::-1]

		self.text_rank = [list(self.vertices.keys())[index] for index in rank_descend]

	def keywords_textrank(self):
		jieba.load_userdict('my_dict')
		self.cut()
		self.vertices_dict()
		self.edge_array()
		self.cal_text_rank()
		return self.text_rank




data = pd.read_excel('../qa_20181030.xlsx')['query']
print(data.head())
jieba.load_userdict('my_dict')
textrank_keys = []
for line in data:
	rank = TextRank(line, 1, 0.85, 'stopwords.txt')
	textrank_keys.append(' '.join(rank.keywords_textrank()))
pd.DataFrame({'text_rank_keywords':textrank_keys}).to_csv('../text_rank_keywords.csv')




