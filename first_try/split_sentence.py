#-*-coding:utf-8-*-

import re
import csv
import codecs

file_path = '/home/pingan_ai/dxq/first_try/first_run.csv'

FileWriter = open('/home/pingan_ai/dxq/first_try/first_run_split.csv','a')
FileWriter.write(codecs.BOM_UTF8)
writer = csv.writer(FileWriter)
writer.writerow(['src','trg','theta'])
patten = u'[，。！？；]'
with open(file_path,'r') as rf:
	reader = csv.reader(rf)
	for i,line in enumerate(reader):

		# line = line.decode('utf-8')
		# print line
		try:
			temp = re.split(patten,line[0].decode('utf-8'))
		except IndexError:
			print line
			break
		print line[1]
		# if i%100 == 0:
		# print str(temp).decode("unicode-escape")
		for j in range(len(temp)):
			if j == len(temp)-2 or j == len(temp)-1:
				continue
			if len(temp[j])>0 and len(temp[j+1])>0:
				writer.writerow([temp[j].encode('utf-8'),temp[j+1].encode('utf-8'),line[1]])
		# if i == 6:
		# 	break
