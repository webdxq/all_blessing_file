#-*-coding:utf-8-*-
import csv
import re
import sys
import os
import chardet
import codecs
import jieba
import collections 
import sys;
reload(sys);
sys.setdefaultencoding("utf8")

stop = [line.strip().decode('utf-8', 'ignore') for line in open('/home/pingan_ai/dxq/project/im2txt_re/stop_words.txt').readlines()]
def rm_stopwords(sentence_list):
	outStr = [] 
	for word in sentence_list:
		if word not in stop:  
			# print(word,end=',')
			outStr.append(word) 
	  # outStr += ' ' 
	# print('\n')
	return outStr

file = 'all_tech.txt'
filename = file.split('/')[-1].split('.')[0]
print filename
all_words = [] 
with open(file,'r') as rf:
	for i,line in enumerate(rf):
		# if chardet.detect(line)['encoding'] != "utf-8":
		# 	continue
		print line
		temp = jieba.lcut(line,HMM=True)
		rm_stop = rm_stopwords(temp)
		all_words += [word for word in rm_stop]
		# w5=jieba.posseg.lcut(line[0],HMM=True)
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
with open(filename+'_words_frequency.csv','w') as wf:
	wf.write(codecs.BOM_UTF8)
	writer = csv.writer(wf)
	for i in count_pairs:
		# print i, all_words[i]
		if u'\u4e00' <= i[0] <=u'\u9fa5':
			writer.writerow([ i[0].encode('utf-8'),i[1]])