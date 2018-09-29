#-*-coding:utf-8-*-
import csv
import re
import sys
import os
import chardet
import pandas as pd
import numpy as np
import codecs
import numpy
reload(sys)
sys.setdefaultencoding('utf-8')

def mkdir(path):
	folder = os.path.exists(path) 	
	if not folder:         	
		os.makedirs(path) 	
		print "---  new folder...  ---"		
		print "---  OK  ---" 	
	else:		
		print "---  This folder has existed!  ---"		

def rm_repeat_keyword(originalfiles, rmfiles, sentence_file):
	print originalfiles
	print rmfiles
	save = pd.DataFrame(columns=['theme_id','content'])
	mkdir('final_data')
	for i in range(7):
		print rmfiles[i]
		print originalfiles[i][0],rmfiles[i][0]
		assert originalfiles[i][0] == rmfiles[i][0]
		# data = pd.read_csv("new_keyword/" + originalfiles[i],names=['content','other'])#['content']
		# data1 = pd.read_csv("new_rmkeyword/" + rmfiles[i])
		data = pd.read_csv("keyword/" + originalfiles[i],names=['content','other'])#['content']
		data1 = pd.read_csv("rmkeyword/" + rmfiles[i])
		data1[['count']] = data1[['count']].astype(int)
		# print data
		# for i in data:
		# 	# print i
		# 	if i in np.array(data1.loc[data1['count'] == 0]['key']).tolist():
		# 		print i
		# print data.shape
		temp = data.loc[~data['content'].isin(np.array(data1.loc[data1['count'] == 0]['key']).tolist())]
		# temp.drop(['other'])
		del temp['other']
		# print temp.shape[0]
		temp.insert(0,'theme_id',np.array([rmfiles[i][0] for j in range(temp.shape[0])]))
		print temp.shape

		# save = temp
		save = pd.concat([save,temp],ignore_index= True)
	print save
	print save.info()
	# save.to_csv('total_keywords.csv',encoding="utf-8")
	sentences = pd.read_csv(sentence_file,names=['sentence','tags'])
	print sentences.info()
	map_blessing_keyword = pd.DataFrame(columns=['blessing_id','keyword_id'])
	cannotfind_sentence = []
	for i in range(7):
		tag = originalfiles[i][0]
		print tag
		sentences['bools'] = sentences.tags.apply(lambda x: True if tag in x else False)
		# print sentences
		selected_sentences = sentences[sentences['bools'].isin([True])]
		# print selected_sentences
		del selected_sentences['bools']
		# selected_sentences.drop(['bools'],axis=1)
		# selected_sentences['bools'] = selected_sentences.bools.apply(lambda x : False)
		print selected_sentences
		temp_keyword = save.loc[save['theme_id'] == tag]
		a = [0 for i in range(temp_keyword.shape[0])]
		pos = 0
		for index, row in temp_keyword.iterrows():
			# print chardet.detect(row['content'])
			
			for sentence_index, sentence_row in selected_sentences.iterrows():
				# print chardet.detect(sentence_row['sentence'])
				# print sentence_row['sentence']
				try:
					if chardet.detect(sentence_row['sentence'])['encoding']=='utf-8':
					# print sentence
						sentence_row['sentence'] = sentence_row['sentence'].decode('utf-8')
					else:
						# print sentence
						sentence_row['sentence'] = sentence_row['sentence'][1:]
						while chardet.detect(sentence_row['sentence'])['encoding']!='utf-8':
							sentence_row['sentence'] = sentence_row['sentence'][1:]
							# print sentence_row['sentence']
						sentence_row['sentence'] = sentence_row['sentence'].decode('utf-8')
					
				except TypeError as e:
					# print e.message
					pass
				if row['content'] in sentence_row['sentence']:
					# print pos
					a[pos] += 1
					temp = pd.DataFrame([[sentence_index,index]],columns=['blessing_id','keyword_id'])
					# print temp
					map_blessing_keyword = pd.concat([map_blessing_keyword,temp],ignore_index= True)
					# 
					# print sentence_row['sentence']
					# chardet.detect(sentence_row['sentence'])['encoding']!='unicode'
				# 	sys.exit()
			pos += 1
		cannotfind_sentence = cannotfind_sentence+a
		print map_blessing_keyword	
		# print cannotfind_sentence
		print len(cannotfind_sentence)
	save['count'] = cannotfind_sentence
	save.to_csv('final_data/all_keyword_count.csv',encoding="utf-8")
	map_blessing_keyword.to_csv('final_data/map_blessing_keyword.csv',encoding="utf-8")

		# print temp_keyword
		# temp.drop(['other'])
		# print temp

		# for i in data1.loc[data1['count'] == 0]['key']:

		# # s = pd.Series(data1[['count']])
		# s = data1.pop([['rm_count']])
		# print data1
		# print data1[['rm_count']]
		# data1['count'] = pd.to_numeric(data1['count'])
		# print data1.info()
		# print pd.to_numeric(s)
		# with open("rm_keword/" + rmfiles[i],'r') as rf:
		# 	f_csv = csv.reader(rf)
		# 	for i in f_csv:
		# 		print type(i[0])



if __name__ == '__main__':
	# args = sys.argv[1:]
	# originalfiles = args[0]
	# rmfiles = args[1]
	# originalfiles = ['0_keyword_t_new.csv', '1_keyword_t_new.csv', '4_keyword_t_new.csv', '5_keyword_t_new.csv', '9_keyword_t_new.csv']
	# rmfiles = ['0_count_keyword.csv', '1_count_keyword.csv', '4_count_keyword.csv', '5_count_keyword.csv', '9_count_keyword.csv']
	# sentence_file = 'refined_CHValentine_test.csv'
	originalfiles = ['0MidAutumnFes_tag_keyword.csv','1MidAutumnFes_tag_keyword.csv','2MidAutumnFes_tag_keyword.csv','3MidAutumnFes_tag_keyword.csv','4MidAutumnFes_tag_keyword.csv','5MidAutumnFes_tag_keyword.csv','6MidAutumnFes_tag_keyword.csv']
	rmfiles = ['0new_count_keyword.csv','1new_count_keyword.csv','2new_count_keyword.csv','3new_count_keyword.csv','4new_count_keyword.csv','5new_count_keyword.csv','6new_count_keyword.csv']
	sentence_file = 'MidAutumnFes_refine_final_v3.csv'

	rm_repeat_keyword(originalfiles, rmfiles, sentence_file)