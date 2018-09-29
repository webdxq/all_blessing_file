#-*-coding:utf-8-*-

import csv
import re
import sys
import os
import chardet
import codecs
import pandas as pd
import numpy as np
import collections
reload(sys)
sys.setdefaultencoding('utf-8')

keywords =  pd.read_excel("xlsx_file/关键词.xlsx")
# blessings =  pd.read_excel("xlsx_file/祝福語.xlsx")
# map_k2b =  pd.read_excel("xlsx_file/祝福语关键字映射表.xlsx")

del keywords['gmt_create']
del keywords['gmt_modified']

# for i in range():
filetype = [pd.DataFrame(columns=['id','theme_id','content']) for i in range(5)]
# print filetype[0]
# print filetype[1]
# print filetype[2]
# print keywords['theme_id'].dtypes
filetype_list = ['0','1','4','5','9']
for i,types in enumerate(filetype_list):
	# print types
	# print keywords.loc[keywords['theme_id'] == int(types)]
	filetype[i] = pd.concat([filetype[i], keywords.loc[keywords['theme_id'] == int(types)]],ignore_index= True) 
	# print filetype[i]

keywords_list = [np.array(filetype[i]['content']).tolist() for i in range(5)]
# print keywords_list
# intersection_pf = pd.DataFrame([])
# print intersection_pf
weigth_wf = open('keywords_weigth.txt','w')

all_new_list = []
with open('intersection_key.txt','w') as wf:
	for i in range(5):
		# all_words = list(keywords_list[i])
		all_words = []
		for j in range(5):
			# if i == j:
			# 	continue
			iset = set(keywords_list[i]).intersection(set(keywords_list[j]))
			iset = list(iset)
			print filetype_list[i],filetype_list[j],len(iset)
			wf.write(filetype_list[i]+'-'+filetype_list[j]+',')
			wf.write(",".join(iset))
			wf.write("\n")
			all_words = all_words + iset
			# print len(iset),len(all_words)

		# print all_words
		# for word in all_words:
		# 	print word,
		# print '\n'
		counter = collections.Counter(all_words)  
		count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
		words, counts = zip(*count_pairs)
		words = list(words)
		counts = list(counts)
		# print counts
		# for i in counts:
		weight_before = [1/float(count) for count in counts]
		# print weight_before
		sum_counts = sum(weight_before)
		weight = [str(weights/sum_counts) for weights in weight_before]
		# print weight
		weigth_wf.write(filetype_list[i]+',')
		weigth_wf.write(",".join(words))
		weigth_wf.write("\n")
		weigth_wf.write('counts,')
		weigth_wf.write(",".join(weight))
		weigth_wf.write("\n")
		new_key_word_list = [None for k in range(len(keywords_list[i]))]
		# new_key_word_list = [new_key_word_list[keywords_list[i].index(words[r])] = weight[r] for r in range(len(keywords_list[i]))]
		print filetype[i]
		print len(words),len(keywords_list[i]),len(new_key_word_list)
		for r in range(len(keywords_list[i])):
			# print r
			# if keywords_list[r] not in words:
			# 	print keywords_list[r]
			# 
			try:
				new_key_word_list[keywords_list[i].index(words[r])] = weight[r]
			except IndexError:
				print keywords_list[i][r]
				# print keywords_list[i][r],new_key_word_list[r]
				# print 
				counter_test = collections.Counter(keywords_list[i])  
				count_pairs_test = sorted(counter_test.items(), key=lambda x: -x[1])  
				print count_pairs_test[0][0]

				sys.exit()

		for r in range(len(keywords_list[i])):
			print r
			if keywords_list[i][r] not in words:
				print 'here!',keywords_list[r]

		all_new_list = all_new_list+new_key_word_list
		print len(all_new_list)
		new_counts = 0
		for i,new_key in enumerate(all_new_list):
			if None == new_key:
				new_counts += 1
		print new_counts
weigth_wf.close()
		# intersection_pf.insert(intersection_pf.shape[1], filetype_list[i]+'-'+filetype_list[j], pd.Series(iset))
# can_find = pd.DataFrame(columns=['blessing_id','keyword_id'])

# for index, row in map_k2b.iterrows():
# 	print blessings.loc[blessings['id'] == row['blessing_id']]
# 	if row['blessing_id'] not in blessings['id']:
# 		temp = pd.DataFrame([[row['blessing_id'],row['keyword_id']]],columns=['blessing_id','keyword_id'])
# 					# print temp
# 		can_find = pd.concat([can_find,temp],ignore_index= True)

# intersection_pf.to_csv('intersection_key.csv',encoding="utf-8")
# print keywords
# print blessings
# print map_k2b
