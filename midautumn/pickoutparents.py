#-*-coding:utf-8-*-
import csv
import re
import sys
import os
import chardet
import codecs
from jieba import posseg
import collections 
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from pandas.core.frame import DataFrame
stop = [line.strip().decode('utf-8', 'ignore') for line in open('/home/pingan_ai/dxq/project/im2txt_re/stop_words.txt').readlines()]
taglist = ['ag','c','e','f','mg','o','p','g','r','tg','u','ud','ug','vn','y','zg']
# csv_writers = codecs.open('MidAutumnFes_tag_1_pick.csv','w')
# csv_writers.write(codecs.BOM_UTF8)
# writer = csv.writer(csv_writers)
# stringList = ['爸爸','妈妈','父母','父亲','母亲']
# pattens = [re.compile(i.decode('utf-8')) for i in stringList]

# with open('MidAutumnFes_tag_1.csv', 'r') as rf:
# 	rd = csv.reader(rf)
# 	for i,lines in enumerate(rd):
# 		line = lines[0].decode('utf-8')
# 		# print line
# 		stop = False
# 		# for patten in pattens
# 		j = 0
# 		while stop == False and j != len(stringList)-1:
# 			j += 1
# 			if pattens[j].search(line):
# 				# print patten.search(line)
# 				print line
# 				writer.writerow([lines[0],lines[1]])
# 				stop = True
# 				# continue
# 	
def PickParents(lines):
	stringList = [u'爸爸',u'妈妈',u'父母',u'父亲',u'母亲',u'爸妈']
	# pattens = [re.compile(i.decode('utf-8')) for i in stringList]			
	for patten in stringList:
		# print patten.search(lines)
		# if patten.search(lines):
		print patten in lines
		if patten in lines:

			return True
	return False

def isdigit(string):
	# r1 = r'\d'
	if re.match(r'\d',string):
		# print False
		return False 
	# print True
	return True

def rm_stopwords(sentence_list):
	outStr = [] 
	for word in sentence_list:
		if word not in stop and isdigit(word) :  
			# print(word,end=',')
			outStr.append(word) 
	  # outStr += ' ' 
	# print('\n')
	return outStr

def get_keywords(file):
	all_words_t = {}
	filename = file.split('/')[-1].split('.')[0]
	print filename
	with open(file,'r') as rf:
		f_csv = csv.reader(rf)
		for i,line in enumerate(f_csv):
			# temp = jieba.lcut(line[0],HMM=True)
			w5=posseg.lcut(line[0],HMM=True)
			# rm_stop = rm_stopwords(w5)
			# print '**********************************************'
			# print line[0],line[1]
			# print '/'.join(temp)
			# print '------------------------------'
			print line[0]
			for j,tag in w5:
				# print type(j)
				# print j,tag
				if j in stop or tag in taglist or len(j)<=1:
					continue
				if not isdigit(j):
					continue

				# print j,tag
				all_words_t[j] = tag
				# print j+'/',
			# print '\n'

			print '\n'
			# all_words += [word for word in rm_stop]
	# counter = collections.Counter(all_words)
	# count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
	# print ' '.join(counter)
	# print len(counter)
	with open(filename+'keyword.csv','w') as wf:
		wf.write(codecs.BOM_UTF8)
		writer = csv.writer(wf)
		for key in all_words_t:
			print key
			# print i, all_words_t[i]
			writer.writerow([ key.encode('utf-8'), all_words_t[key]])
	
def OnlyOlders(x):
	if ('0' in x or '1' in x or '2' in x) and ('3' not in x or '4' not in x):
		return True
	return False

def olders(x):
	if ('0' in x or '1' in x or '2' in x) and ('3' in x or '4' in x):
		return True
	return False
def PopTags(tags,tag):
	try:
		tags.pop(tags.index(tag))
	except Exception as e:
		print e.message
		pass
def find_older(file):
	filename = file.split('/')[-1].split('.')[0]
	pattern = re.compile(u'你')
	sentences = pd.read_csv(file,names=['sentence','tags'])
	sentences['bools'] = sentences.tags.apply(lambda x: True if olders(x) else False)
	# sentences['_bools'] = sentences.tags.apply(lambda x: True if olders(x) else False)
	selected_sentences = sentences[sentences['bools'].isin([True])]
	print "selected_sentences",selected_sentences
	count = 0
	add_rows = pd.DataFrame(columns=['sentence','tags',"bools"])
	for index, row in selected_sentences.iterrows():
		if chardet.detect(row['sentence'])['encoding']=='utf-8':
					# print sentence
			row['sentence'] = row['sentence'].decode('utf-8')
		else:
			# print sentence
			row['sentence'] = row['sentence'][1:]
			while chardet.detect(row['sentence'])['encoding']!='utf-8':
				row['sentence'] = row['sentence'][1:]
				# print row['sentence']
			row['sentence'] = row['sentence'].decode('utf-8')
		# row['sentence'] = re.sub(pattern, u'您', row['sentence'], count=0, flags=0)
		if u'你' in row['sentence']: 
			# count += 1
			# print row["sentence"]
			# print row
			tags = row['tags'].split(',')
			# sentence = row["sentence"].values
			# print sentence
			# flag = 	row["bools"].values	
			# print flag
			PopTags(tags,'0')	
			PopTags(tags,'1')	
			PopTags(tags,'2')	
			tags = ",".join(tags)

			# temp['tags'] = tags
			temp = pd.DataFrame([[row["sentence"],tags,row["bools"]]],columns=['sentence','tags','bools'])
			# print temp
			# print temp["sentence"],temp["tags"]
			add_rows = pd.concat([add_rows,temp],ignore_index= True)
			# print temp
			
		
		# break
	print add_rows
	print count
	sentences = pd.concat([add_rows,sentences],ignore_index= True)
	for index, row in sentences.iterrows():
		try:
			if chardet.detect(row['sentence'])['encoding']=='utf-8':
					# print sentence
				row['sentence'] = row['sentence'].decode('utf-8')
			else:
				# print sentence
				row['sentence'] = row['sentence'][1:]
				while chardet.detect(row['sentence'])['encoding']!='utf-8':
					row['sentence'] = row['sentence'][1:]
					# print row['sentence']
				row['sentence'] = row['sentence'].decode('utf-8')
		except Exception as e:
			print e.message
			pass
		
		if u'你' in row['sentence'] and olders(row['tags']):
			tags1 = row['tags'].split(',')
			row['sentence'] = re.sub(pattern, u'您', row['sentence'], count=0, flags=0)
			PopTags(tags1,'3')	
			PopTags(tags1,'4')
			row['tags'] = ",".join(tags1)	
		if '1' in row['tags'] and len(row['tags'])>1:
			tags1 = row['tags'].split(',')
			if PickParents(row['sentence']):
				print 'here'
				pass
			else:
				PopTags(tags1,'1')
				row['tags'] = ",".join(tags1)
		if OnlyOlders(row['tags']):
			row['sentence'] = re.sub(pattern, u'您', row['sentence'], count=0, flags=0)

	print sentences
	del sentences['bools']
	sentences.to_csv(filename+'_v3.csv',index=False,encoding="utf-8")
	# print "******************************************************"		
	# print "selected_sentences",selected_sentences
if __name__ == '__main__':
	file = sys.argv[1:][0]
	# all_words = []
	find_older(file)
	