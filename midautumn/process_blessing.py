#-*-coding:utf-8-*-
import csv
import re
import sys
import os
import chardet
import codecs
import jieba
import collections 

'''
神武英明盖世无双，人见人爱花见花开，打遍天下无敌手，情场杀手鬼见愁玉面飞龙，人称美貌无双心地善良，晕倒一片的我，祝领导中秋快乐,"0,5"
月儿圆圆，心儿暖暖，中秋佳节到了，送你一个用温馨浪漫体贴关怀炮制，咬一口充满了甜蜜的什锦月饼，祝领导永远幸福开心健康衷心祝福。,0
'''
less_wf = codecs.open('length_less_20.txt','w','utf-8')
long_wf = codecs.open('length_longer_100.txt','w','utf-8')
# refine_wf = codecs.open('MidAutumnFes_refine_v2_keep.csv','w','utf-8')
# refine_wf.write(codecs.BOM_UTF8)
# writer = csv.reader(refine_wf)
tags = ['0','1','2','3','4','5','6']
writers = [codecs.open('MidAutumnFes_tag_%s.csv'%i,'w') for i in tags]
fwriters = [None for i in range(len(tags))]
for i,writer in enumerate(writers):
	writer.write(codecs.BOM_UTF8)
	fwriters[i] = csv.writer(writer)
with open('MidAutumnFes_refine_v2_keep.csv', 'r') as rf:
	rd = csv.reader(rf)
	for i,lines in enumerate(rd):
		print lines[0]
		# line = lines.strip('\n').split(',')[0]

		# # line_temp = line.split(',')
		# if len(line) < 20:
		# 	less_wf.write(lines)
		# elif len(line) > 100:
		# 	print len(line)
		# 	long_wf.write(lines)
		# else:
		# 	refine_wf.write(lines)
			# writer.writerow()
		
		# print tags 
		for j,tag in enumerate(tags):
			if tag in lines[1]:
				fwriters[j].writerow([lines[0],tag])


less_wf.close()
long_wf.close()