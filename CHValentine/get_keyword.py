#-*-coding:utf-8-*-

import csv
import re
import sys
import os
import chardet
import codecs
import jieba
import collections 
import jieba.posseg
import sys;
reload(sys);
sys.setdefaultencoding("utf8")
jieba.load_userdict("add_words.txt") 
stop = [line.strip().decode('utf-8', 'ignore') for line in open('/home/DongXiaoqun/project/stopwords.txt').readlines()]
taglist = ['ag','c','e','f','mg','o','p','g','r','tg','u','ud','ug','vn','y','zg']
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
			outStr.append(word) 
	return outStr

def mkdir(path):
	folder = os.path.exists(path) 	
	if not folder:         	
		os.makedirs(path) 	
		print "---  new folder...  ---"		
		print "---  OK  ---" 	
	else:		
		print "---  This folder has existed!  ---"		

def get_keywords(file):
	all_words = []
	all_words_t = {}
	filename = file.split('/')[-1].split('.')[0]
	print filename
	with open(file,'r') as rf:
		f_csv = csv.reader(rf)
		for i,line in enumerate(f_csv):
			# temp = jieba.lcut(line[0],HMM=True)
			w5=jieba.posseg.lcut(line[0],HMM=True)
			rm_stop = rm_stopwords(temp)
			print line[0]
			for j,tag in w5:
				# print type(j)
				if j in stop or tag in taglist or len(j)<=1:
					continue
				if not isdigit(j):
					continue
				# print j,tag
				all_words_t[j] = tag
				print j+'/',
			# print '\n'
			print '\n'
			all_words += [word for word in rm_stop]
	counter = collections.Counter(all_words)
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
	# print ' '.join(counter)
	# print len(counter)
	mkdir(filename)
	with open(filename+'/'+filename+'_test_t.csv','w') as wf:
		wf.write(codecs.BOM_UTF8)
		writer = csv.writer(wf)
		for i in all_words_t:
			# print i, all_words_t[i]
			writer.writerow([ i, all_words_t[i]])

	with open(filename+'/'+filename+'_test.csv','w') as wf:
		wf.write(codecs.BOM_UTF8)
		writer = csv.writer(wf)
		for i in count_pairs:
			# print i, all_words[i]
			writer.writerow([ i[0].encode('utf-8'),i[1]])
if __name__ == '__main__':
	file = sys.argv[1:][0]
	get_keywords(file)
