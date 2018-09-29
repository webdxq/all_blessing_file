#-*-coding:utf-8-*-
import csv
import re
import sys
import os
import chardet
import codecs
import jieba
import collections 

#www.31dx.com.csv  www.aizhufu.cn.csv  www.glook.cn.csv
keep_tag = ['0','1','2','3','4','5','9']
tag_name = ['missing','humor','boyfriends','girlfriends','lover','friends','confession']
keep_tag_index = dict(zip(keep_tag, range(len(keep_tag))) )
keep_tag_vocal = [[] for i in range(len(keep_tag))]
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
def tag_keyword(tag, word):
 	if tag in keep_tag:
 		print tag
 		# if word not in keep_tag_vocal[keep_tag_index[tag]]:
		keep_tag_vocal[keep_tag_index[tag]].append(word)

if __name__ == '__main__':
	file = sys.argv[1:][0]
	# r1 = re.compile('白色情人节')
	print keep_tag_index
	with open(file,'r') as rf:
		f_csv = csv.reader(rf)
		for i,line in enumerate(f_csv):
			# temp = re.sub(r1, '七夕情人节'，line[0])
			temp = jieba.lcut(line[0])
			# rm_stop = rm_stopwords(temp)
			# print rm_stop
			# print '/'.join(rm_stop)
			# tag_keyword()
			# map(tag_keyword,line[1],rm_stop)
			map(tag_keyword,line[1],temp)
			# if i == 37:
			# 	temp = jieba.lcut(line[0])
			# 	rm_stop = rm_stopwords(temp)
			# 	# print rm_stop
			# 	# print '/'.join(rm_stop)
			# 	# tag_keyword()
			# 	map(tag_keyword,line[1],rm_stop)
				# break
		# print str(keep_tag_vocal[4]).decode("unicode-escape")
	# sys.exit()
	for i,vocal in enumerate(keep_tag_vocal):
		print '***********************************'
		counter = collections.Counter(vocal)
		count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
		with open(tag_name[i]+'key_word.csv','w') as wf:
			wf.write(codecs.BOM_UTF8)
			writer = csv.writer(wf)
			writer.writerow([tag_name[i],i])
			for j in count_pairs:
				# print i[0],i[1]
				# if j[0]>= u'\u4e00' and j[0]<=u'\u9fa5':
				writer.writerow([j[0].encode('utf-8'),j[1]])
		# print type(count_pairs)
		# print count_pairs
		
		# print count_pairs
		# words, _ = zip(*count_pairs)  
