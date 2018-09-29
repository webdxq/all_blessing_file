#-*-coding:utf-8-*-

import csv
import re
import sys
import os
import chardet
import pandas as pd
import numpy as np
import codecs
from multiprocessing.dummy import Pool as ThreadPool
from pandas.core.frame import DataFrame
reload(sys)
sys.setdefaultencoding('utf8')


def RmMassyCodes(char_code):
	# print tag,'|',taglist
	return char_code >= u'\u4e00' and char_code<=u'\u9fa5'


def mkdir(path):
	folder = os.path.exists(path) 	
	if not folder:         	
		os.makedirs(path) 	
		print "---  new folder...  ---"		
		print "---  OK  ---" 	
	else:		
		print "---  This folder has existed!  ---"		

def SearchNoMatchSentence(keyword_file,sentence_file):
	# all_sentences = open(sentence_file,'r').readlines()
	filelist =  os.listdir(keyword_file)
	mkdir('rmkeyword')
	# all_sentences = [line.decode('utf-8') for line in all_sentences]
	# print all_sentences
	# for i in all_sentences:
	# 	# print chardet.detect(i)
	# 	print i
	sentences = pd.read_csv(sentence_file,names=['sentence','tags'])
	print sentences.info()
	def noMatchSentence(i):
		# print 
		print keyword_file+'/'+ i
		# keywords = open(keyword_file+'/'+ i,'r').readlines()
		keywords = []
		with open(keyword_file+'/'+ i,'r') as rkf:
			for line in rkf:
				if line[0][:3] == codecs.BOM_UTF8:  # 判断是否为带BOM文件
					line = line[3:]
				line = line.strip().decode('utf-8')
				keywords.append(line)
		print len(keywords)
		print keywords[0]

		# while chardet.detect(keywords[0])['encoding']!='utf-8':
		# 	print 'here'
		# 	keywords[0] = keywords[0][1:]
		# keywords[0] = keywords[0][3:]
		# print keywords[0]
		# sys.exit()
		print i ,type(i),keywords[0]
		tag = i[0]
		# selected_sentences = sentences[sentences['tags'].isin([tag])]
		# split_sentences = sentences['tags'].str.split(',',expand=True)
		# tags = sentences['tags']
		# tagfind = np.array(list(map(find_tag,tags.values)))
		# print len(tagfind)
		# print tagfind
		# print sentences.tags
		sentences['bools'] = sentences.tags.apply(lambda x: True if tag in x else False)
		# for i in tags:
		# 	if tag in i:

		# print selected_sentences

		# with open()
		selected_sentences = sentences[sentences['bools'].isin([True])]
		print "selected_sentences",selected_sentences
		del selected_sentences['bools']
		# selected_sentences.drop(['bools'],axis=1)
		# selected_sentences['bools'] = selected_sentences.bools.apply(lambda x : False)
		print selected_sentences
		row_lines = selected_sentences.iloc[:,0].size
		# pd.concat([selected_sentences, pd.DataFrame(columns=['bool'])])
		# selected_sentences.reindex(columns=['sentence'])
		match_sentence = [False for i in range(row_lines)]

		# for i,keyword in enumerate(keywords):
		# 	# print keyword.split(',')[0]
		# 	print keyword
		# 	keywords[i] = keyword.split(',')[0].decode('utf-8')

		# for keyword in keywords:
		# 	print keyword
		# keywords = [keyword[0].decode('utf-8') for keyword in keywords]
		# print keywords
		keywords = list(set(keywords))
		keywords_match_count = [0 for i in range(len(keywords))]
		for j,keyword in enumerate(keywords):
			# print keyword
			for i, sentence in enumerate(selected_sentences['sentence']):
				# print chardet.detect(sentence)
				if chardet.detect(sentence)['encoding']=='utf-8':
					# print sentence
					sentence = sentence.decode('utf-8')
				else:
					# print sentence
					sentence = sentence[1:]
					while chardet.detect(sentence)['encoding']!='utf-8':
						sentence = sentence[1:]
						# print sentence
					sentence = sentence.decode('utf-8')
					# print sentence
					# sentence = sentence[1:].decode('utf-8')
					# print [sentence]
				# print keyword,sentence
				# sentence = sentence.decode('utf-8')
				if keyword in sentence:
					keywords_match_count[j] += 1
					# print keyword in sentence,i,match_sentence[i]
					# print selected_sentences.iloc[2]['bools']
					# if selected_sentences.iloc[2]['bools']==False:
						# selected_sentences.iloc[2]['bools'] = True
						# selected_sentences.set_value('2','bools',True) 
					if match_sentence[i] == False:
						match_sentence[i] = True
					# break
				# break
			# print match_sentence
		selected_sentences.insert(selected_sentences.shape[1],'bools', match_sentence)
		nomatch_sentences = selected_sentences[selected_sentences['bools'].isin([False])]
		print selected_sentences
		if not nomatch_sentences.empty:
			print '************************************************'
			print 'not empty'
			print nomatch_sentences
			nomatch_sentences.to_csv(tag+'new_not_find.csv',index=False,encoding="utf-8")
		countkey = {"key" : keywords, "count" : keywords_match_count}
		data=DataFrame(countkey)
		data.to_csv('rmkeyword/'+tag+'count_keyword.csv',index=False,encoding="utf-8")
		return

	
	# print sentences.size
	print filelist
	# pool = ThreadPool(len(filelist))
	# pool.map(noMatchSentence, filelist)
	# pool.close()
	# pool.join()
	for i in range(7):

		noMatchSentence(filelist[i])
	# noMatchSentence('1MidAutumnFes_tag_keyword.csv')	
	
	# print sentences['tags']
	# print sentences
	
		
if __name__ == '__main__':
	args = sys.argv[1:]
	keyword_file = args[0]
	sentence_file = args[1]
	print keyword_file," ",sentence_file
	SearchNoMatchSentence(keyword_file,sentence_file)