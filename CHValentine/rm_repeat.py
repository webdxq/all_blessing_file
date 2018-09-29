#-*-coding:utf-8-*-
import csv
import re
import sys
import os
import chardet
import pandas as pd
import numpy as np
import codecs
# reload(sys)
# sys.setdefaultencoding('utf-8')

def rm_repeat(rmfile):
	# originallist = open(originalfile, 'r').readlines()
	# originallist_unicode = map(lambda x: x[0].decode('utf-8'),originallist)
	# ”；
	filename = rmfile.split('.')[0]
	wf = open(filename+'noendsentence.txt','w')
	wf_notcsv = open(filename+'_com.csv','w')
	wf_notcsv.write(codecs.BOM_UTF8)
	writer = csv.writer(wf_notcsv)
	a = [u"。", u"！",u"？"]
	b = [u'，', u'；', u"“",u"：",u":",u";",u"\""]
	with open(rmfile, 'r') as rf:
		f_csv = csv.reader(rf)
		for i in f_csv:
			# print i[0]
			# print i 	
			i[0] = i[0].decode('utf-8')
			# 
			if i[0][-1] not in a:
				print i[0][-1]
				if i[0][-1] in b:
					wf.write(i[0])
					wf.write(',')
					wf.write(i[1])
					wf.write('\n')
					i[0] = i[0][-1]+u'！'
				else:
					wf.write(i[0])
					wf.write(',')
					wf.write(i[1])
					wf.write('\n')
					i[0] = i[0]+'！'
			if len(i[0])>100:
				wf.write("******************\n")
				wf.write(str(len(i[0])))
				wf.write('\n')
				wf.write(i[0])
				wf.write(',')
				wf.write(i[1])
				wf.write('\n')
				wf.write("******************\n")
				print len(i[0])
				continue
			writer.writerow([i[0].encode('utf-8'),i[1]])

if __name__ == '__main__':
	args = sys.argv[1:]
	rmfile = args[0]
	# originalfile = args[1]
	rm_repeat(rmfile)
	print 'done'