#-*-coding:utf-8-*-

import csv
import codecs
import os
import sys
import re
import chardet

def rm_sentence(tag):
	# keep_tag = ['0','1','2','3','4','5','9']
	keep_tag = ['0','1','2','3','4','5','6']
	if tag in keep_tag:
		print tag
		return True
	else:
		return False

if __name__ == '__main__':
	file = sys.argv[1:][0]
	filename = file.split('.')[0]
	# tagname = ["思念","搞笑","送男生朋友","送女生朋友","送对象","送朋友","其他","诗歌","对联","表白"]
	tagname = ["客户","父母","师长","爱人","朋友","搞笑","思念"]
	tagcount = [0 for i in range(len(tagname))]
	combine = 0
	# r1 = re.compile('短信')
	wf = open(filename +'_keep.csv','w')
	wf.write(codecs.BOM_UTF8)
	writer = csv.writer(wf)
	# print temp
	with open(file,'r') as rf:
		count = 0		
		f_csv = csv.reader(rf)
		for i,line in enumerate(f_csv):
			# print i,line[0].decode('utf-8')
			# print i,line[1]
			# print re.sub(r1, '信息', line[0])
			# if re.search(r1, line[0]):
			# 	temp = re.sub(r1, '信息', line[0])
			# 	print i,temp
			# print chardet.detect(line[0])
			tags = line[1].split(',')
			# if '4' in tags and '5' in tags:
			# 	combine += 1
			# # print tags[0]
			# if '4' in tags:
			# 	tagcount[4]+=1
			# if '5' in tags:
			# 	tagcount[5]+=1
			# if '1' in tags:
			# 	tagcount[1]+=1
			# if '0' in tags:
			# 	tagcount[0]+=1
			# if '9' in tags:
			# 	tagcount[9]+=1
			# if '2' in tags:
			# 	tagcount[2]+=1
			# if '3' in tags:
			# 	tagcount[3]+=1
			if map(rm_sentence, tags)[0]:
				print map(rm_sentence, tags)
				writer.writerow(line)
			# if i == 20:
			# 	break

			# if line[1]=='99':
			# 	count+=1
				# print line[1]
		# for i,name in enumerate(tagname):
		# 	print name,tagcount[i]