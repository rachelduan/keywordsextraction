import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_excel('qa_20181030.xlsx')['query']
print(data.head())

jieba.load_userdict('my_dict')

"""
未去除停用词
jieba_analyse = data.apply(lambda x: pd.Series([' '.join(jieba.lcut(x)), 
	                                            ' '.join(jieba.analyse.extract_tags(x))], 
	                                            index = ['jieba_segment', 'jieba_keywords']))
"""

"""
jieba.analyse.extract_tags() 基于tfidf的关键词提取
"""
stopwords = [word.strip() for word in open('stopwords.txt', encoding = 'utf-8')]
jieba_segment, jieba_keys = [],[]
for line in data:
	segment = ' '.join([word for word in jieba.lcut(line) if word not in stopwords])
	jieba_segment.append(segment)
	jieba_keys.append(' '.join(jieba.analyse.extract_tags(segment)))

jieba_analyse = pd.DataFrame({'jieba_segment': jieba_segment, 'jieba_keywords':jieba_keys})

"""
通过TfidfVectorizer进行关键词提取
"""
tfidf_keys = []
vectirizer = TfidfVectorizer()
tfidf = vectirizer.fit_transform(jieba_segment).toarray()
words_list = vectirizer.get_feature_names()

for index, words in enumerate(tfidf):
	index_sort_inverse = np.argsort(words).tolist()
	keys = []
	for i in range(-1*len(jieba_segment[index].split(' ')), 0):
		key = words_list[index_sort_inverse[i]]
		if key in jieba_segment[index].split(' '):
			keys.append(key)
	tfidf_keys.append(' '.join(keys))
tfidf_analyse = pd.DataFrame({'tfidf_keywords':tfidf_keys})

"""
textrank是基于pagerank算法的
建立文本无向图，边权重由词语共现频率决定
通过textrank权重计算公式迭代直到收敛，对权重进行排序产生keywords
http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
textrank算法代码可以参考：https://github.com/summanlp/textrank
"""
textrank = []
jieba_posseg = []
key_word_pos = ['x', 'ns', 'n', 'vn', 'v', 'l', 'j', 'nr', 'nrt', 'nt', 'nz', 'nrfg', 'm', 'i', 'an', 'f', 't',
                        'b', 'a', 'd', 'q', 's', 'z', 'Ag', 'dg', 'h','k', 'Ng','ns', 'nt', 'nz', 'p','un','y','c']
for line in jieba_segment:
	posseg = jieba.posseg.cut(line)
	pos = ''
	for word,flag in posseg:
		if word != ' ':
			pos = pos + word + '/' + flag + ' '
	jieba_posseg.append(pos.strip())
	rank = jieba.analyse.TextRank()
	rank.span = 5
	textr = rank.textrank(line,topK=4, allowPOS=key_word_pos)
	textrank.append(' '.join(textr))
textrank_analyse = pd.DataFrame({'textrank_keywords':textrank})
jieba_pos = pd.DataFrame({'jieba_pos':jieba_posseg})

"""
简单无向图textrank实现
"""
self_textrank_analyse = pd.read_csv('text_rank_keywords.csv')

pd.concat([data, jieba_pos, jieba_analyse, tfidf_analyse, textrank_analyse, self_textrank_analyse], axis = 1).to_csv('keywords_extraction.csv')



