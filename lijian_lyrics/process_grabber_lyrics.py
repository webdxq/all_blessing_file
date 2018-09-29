#-*-coding:utf-8-*-
import jieba
import chardet
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
remove_list = [u"作曲",u"作词",u"音乐总监",u"音响总监",u"编曲",u"笛子",\
				u"吉他",u"键盘",u"贝斯",u"鼓手",u"打击乐",u"和音",u"Program",\
				u"弦乐",u"间奏",u"MIDI制作",u"领唱",u"原唱",u"Cover混音",u"中文填词"\
				,u"音乐制作统筹",u"混音",u"架子鼓",u"首席",u"制作人",u"后期",u"单簧管"\
				,u"录音师",u"音乐推广营销",u"合唱编写",u"Bass",u"钢琴",u"PGM",u"…"\
				,u"Arragement",u"Programing",u"Guitars",u"Keyboards",u"Hormony",\
				u"Recording Engineer",u"Mixing Engineer",u"艺人经纪",u"同期录音",u"缩混"\
				,u"打击乐",u"伴唱",u"化妆师",u"特别鸣谢",u"经纪助理",u"鼓",u"贝斯"\
				,u"Flute",u"童声合唱",u"Drums",u"录音",u"Environment sound effect"\
				,u"English Horn",u"Accordion",u"Accordion",u"Trumpet",u"Aditionnal Drums"\
				,u"Electric Guitar",u"AditionalDrums",u"Synth",u"Synthesizer"\
				u"艺人经纪",u"PA调音师",u"Monitor调音师",u"吉他",u"贝斯",u"键盘/合成器"\
				,u"手风琴",u"鼓",u"打击乐",u"伴唱",u"PGM程序",u"吉他技师",u"长号",u"小号"\
				,u"大提琴",u"笛子",u"经纪助理",u"总成音",u"成音助理",u"后期处理"\
				,u"后期制作处理",u"Bass & Organ",u"Harmony",u"Music",u"Keyboads"\
				,u"Arrangement"]
remove_list2 = [u"Recorded By",u"Mixed By",u"Recorded & Mixed By"]
remove_list_clean = list(set(remove_list))
def Remove(line):
	for words in remove_list_clean:
		if words in line and u"：":
			# print line
			return True
	for words in remove_list2:
		if words in line:
			# print line
			return True
	return False

def HasReapeatWord(string):
    flag = False
    for i,char in enumerate(string):
        # print i
        s = i
        m = i+1
        e = i+2 
        # print string[s],string[m],string[e]
        if flag:
            return True
        elif e >= (len(string)-1):
            return False
        else:
            if string[s] == string[m] and string[m] == string[e]:
            	print string
                flag = True
            else:
                continue

origin = 'pushu_lyrics.txt'
clean_txt = origin.split('.')[0] + '_clean.txt'
writefile = open(clean_txt,'w')
with open(origin,'r') as rf:
	for i,line in enumerate(rf):
		if line == "\n":
			print i
			continue
		if Remove(line):
			continue
		if HasReapeatWord(line):
			continue
		if chardet.detect(line)['encoding'] != "utf-8":
			writefile.write(line)
			# writefile.write('\n')
			print line
			continue
		if len(line) == 1:
			continue
		# print line
		# print chardet.detect(line)
		line = line.decode('UTF-8')
		
		if u"：" in line:
			line = line.split(u'：')[1]
			print line 
		if u":" in line:
			line = line.split(':')[1]
			print line 
		temps = line.split()
		for temp in temps:
			writefile.write(temp.encode('utf-8'))
			writefile.write('\n')

writefile.close()


        # line = line.strip(u'\n')
        # print line
        
        # for words in remove_list:
        # 	if words in line:
        # 		print line
        # 		continue

