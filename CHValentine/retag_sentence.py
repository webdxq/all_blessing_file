#-*-coding:utf-8-*-

import csv
import codecs
import os
import sys
import re
# import chardet

if __name__ == '__main__':
	file = sys.argv[1]
	number = int(sys.argv[2])
	print file
	name = file.split('/')[-1].split('.')[0]
	print name 
	FileWriter = open(name+'_reviewed.csv','a')
	FileWriter.write(codecs.BOM_UTF8)
	writer = csv.writer(FileWriter)
	r1 = u'[\u2E80-\u9FFF，《》！？、；：“”’‘ \t\d]+'
	r2 = re.compile('\D')
	with open(file,'r') as rf:
		f_csv = csv.reader(rf)
		for i,line in enumerate(f_csv):
			if i < number:
				# print i
				continue
			else:
				print '-----------------------------------------------------------'
				
				line[0] = ''.join(re.findall(r1, line[0].decode('utf-8')))
				print i,line[0]
				print 'tag ',line[1]
				text_modify = raw_input("文本，修改：y; 不修改：Enter| 输入：")
				print text_modify
				# exit_code = raw_input()
				# if exit_code == 'e':
				# 	sys.exit()
				# else:

				while text_modify != '':
					with open('temp_modify.txt','w') as wf:
						wf.write(line[0].encode('utf-8'))
					if text_modify == 'y':
						text_input = raw_input('输入修改句子：').decode(sys.stdin.encoding)
						if text_input == '':
							text_modify = text_input
						break
					else:
						print 'wrong input'
						text_modify = raw_input("文本，修改：y; 不修改：Enter| 输入：")
				if text_modify == '':
					text_input = line[0]
				tag_modify = raw_input("标签，思念：0，搞笑：1， 送男生朋友：2，送女生朋友：3，送对象：4，送朋友：5，中性：6:，诗歌：7，对联：8 ，表白：9| 输入：")
				while tag_modify != '':
					if re.match('[0-9]*$|[0-9],[0-9]',tag_modify) != None:
						tag_input = tag_modify
						# print type(tag_input)
						break
					else:
						tag_modify = raw_input("重新输入标签：")
						# print type(tag_input)
				if tag_modify == '':
					tag_input = line[1]
				writer.writerow([text_input.encode('utf-8'),tag_input])
				# break

