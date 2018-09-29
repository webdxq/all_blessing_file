#-*- coding:utf-8 -*-

import csv
import re
import sys
import os
# reload(sys)
# sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()
import codecs
import chardet

dirname = "/home/pingan_ai/dxq/corpus/"
# dirname = "/home/pingan_ai/dxq/project/sentence_douwnload/aizhufu"
files = os.listdir(dirname)
FestivalList = ['七夕','情人节','白色情人节','思念']
Festival_fileList = ['CHValentine','Valentine','WhiteValentin','Missing']
# FestivalList = ['圣诞','中秋','七夕','元宵','春节','元旦','新年']
# Festival_fileList = ['Christmas','MidAutumnFes','CHValentine','LanternFes','SpringFes','CHNewYear','Newyears']

FileWriter = [None for i in range(len(FestivalList))]
pattens = [re.compile(i.decode('utf-8')) for i in FestivalList]
for i, festival in enumerate(FestivalList):
	# FileWriter[i] = open('/home/pingan_ai/dxq/clean_corpus/'+Festival_fileList[i]+'.csv','a')
	FileWriter[i] = open('/home/pingan_ai/dxq/CHValentine/'+Festival_fileList[i]+'.csv','a')
	FileWriter[i].write(codecs.BOM_UTF8)
# tagList = ['朋友','你们','姐妹','哥们']
# tagPatten = [re.compile(i.decode('utf-8')) for i in tagList]
# tagPatten = re.compile('朋友')
r1 = u'[\u2E80-\u9FFF，《》！？、；：“”’‘ \t\d]+'
# otherWriter = open('/home/pingan_ai/dxq/CHValentine/otherpeople.csv','a')
# otherWriter.write(codecs.BOM_UTF8)
# owriter = csv.writer(otherWriter)
print files
for file in files:
	print file
	if file[-1] == 'y':
		# print file
		continue
	with open(dirname+'/'+file,'r') as rf:
		f_csv = csv.reader(rf)
		# print f_csv
		for line in f_csv:
			# print chardet.detect(line[0])
			# print line[0]
			# print str(line[0].decode('utf-8').split()).decode("unicode-escape")
			try:
				temp = line[0].decode('utf-8').split()
			except IndexError:
				print line
				continue
			if temp==[]:
				continue
			index = -1
			# print type(temp[0])
			# print patten.match(temp[0])
			# print re.findall(u'春节',temp[0])
			for i, patten in enumerate(pattens):
				# print patten
				if patten.match(temp[0]):
					index = i
			if index!=-1:
				print index,temp[0]
				writer = csv.writer(FileWriter[index])
				try:
					temp_str = ''.join(re.findall(r1, temp[1]))
					if len(temp_str)>14:
						# if tagPatten.match(temp[1]):
						# 	owriter.writerow([(''.join(re.findall(r1, temp[1]))).encode('utf-8'),index])

						writer.writerow([temp_str.encode('utf-8'),index])
				except IndexError:
					# print str(temp).decode("unicode-escape")
					continue
			# break
			# print line[0]
	# with codecs.open(file,'rb', 'gb2312') as rf:
	# 	f_csv = csv.reader(rf)
	# 	print f_csv
	# 	for line in f_csv:
	# 		print line
	# 		# print line[0].decode('GB2312').encode('utf-8')