#encoding=utf-8
import requests
import json
import re
import os
from bs4 import BeautifulSoup

headers = {
	'Referer':'https://music.163.com',
	'Host': 'music.163.com',
	'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36',
	'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
}

def get_top50(artist_id):
	url = "http://music.163.com/artist?id="+str(artist_id)

	s = requests.session()
	s = BeautifulSoup(s.get(url,headers=headers).content,"lxml")

	artist_name = s.title

	main = s.find('ul',{'class':'f-hide'})
	main = main.find_all('a')

	song = {}
	song['artist_name'] = artist_name.text
	song['list'] = main

	return song

def get_lyric(song_id):
	list = song_id.split('=')
	id = list[1]
	url = "http://music.163.com/api/song/lyric?id="+str(id)+"&lv=1&kv=1&tv=-1"

	s = requests.session()
	s = BeautifulSoup(s.get(url,headers=headers).content,"lxml")
	json_obj = json.loads(s.text)

	final_lyric = ""
	if( "lrc" in json_obj):
		inital_lyric = json_obj['lrc']['lyric']
		regex = re.compile(r'\[.*\]')
		final_lyric = re.sub(regex,'',inital_lyric).strip()

	return final_lyric

def makedir(dir_name):
	folder = os.path.exists(dir_name)

	if not folder:
		os.makedirs(dir_name)
		print("creat dir success")
	else:
		print("this folder has existed")

def ConvertStrToFile(dir_name,filename,str):
	if (str == ""):
		return
	filename = filename.replace('/','')
	with open(dir_name+"//"+filename+".txt",'w') as f:
		f.write(str.encode('utf-8'))
		

def MergeStrToFile(f,str):
	if (str == ""):
		return
	f.write('LyricsStart\n')
	f.write(str.encode('utf-8'))
	f.write('\nLyricsEnd')
	f.write('\n')
def get_top50_lyric(artist_id):
	songlist = get_top50(artist_id)

	artist_name = songlist['artist_name']
	idlist = songlist['list']

	makedir(artist_name)

	wf = open('name.txt','w')
	for music in idlist:
		print music['href']
		wf.write((music.text).encode('utf-8'))
		wf.write('\n')
		print(get_lyric(music['href']))
		# ConvertStrToFile(artist_name,music.text,get_lyric(music['href']))
		with open(artist_name+".txt",'a') as f:
			MergeStrToFile(f,get_lyric(music['href']))
		print("File "+music.text+" is writing on the disk")
		
	print("All files have created successfully")
	wf.close()
#print(get_top50(1007170))
get_top50_lyric(3695)
#print(get_lyric("song?id=542667185"))