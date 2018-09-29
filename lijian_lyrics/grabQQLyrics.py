#-*-coding:utf-8-*-

def download(url, user_agent='wswp', num_retries=2):
    if url is None:
        return None
    print('Downloading:', url)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    request = urllib.request.Request(url, headers=headers)  # 设置用户代理wswp(Web Scraping with Python)
    try:
        html = urllib.request.urlopen(request).read().decode('utf-8')
    except urllib.error.URLError as e:
        print('Downloading Error:', e.reason)
        html = None
        if num_retries > 0:
            if hasattr(e, 'code') and 500 <= e.code < 600:
                # retry when return code is 5xx HTTP erros
                return download(url, num_retries-1)  # 请求失败，默认重试2次,
    return html

def music_scrapter(html, page_num=0):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        mod_songlist_div = soup.find_all('div', class_='mod_songlist')
        songlist_ul = mod_songlist_div[1].find('ul', class_='songlist__list')
        '''开始解析li歌曲信息'''
        lis = songlist_ul.find_all('li')
        for li in lis:
            a = li.find('div', class_='songlist__album').find('a')
            music_url = a['href']  # 单曲链接
            urls.add_new_url(music_url)  # 保存单曲链接
            # print('music_url:{0} '.format(music_url))
        print('total music link num:%s' % len(urls.new_urls))
        next_page(page_num+1)
    except TimeoutException as err:
        print('解析网页出错:', err.args)
        return next_page(page_num + 1)
    return None

def get_music():
     try:
        while urls.has_new_url():
            # print('urls count:%s' % len(urls.new_urls))
            '''跳转到歌曲链接，获取歌曲详情'''
            new_music_url = urls.get_new_url()
            print('url leave count:%s' % str( len(urls.new_urls) - 1))
            html_data_info = download(new_music_url)
            # 下载网页失败，直接进入下一循环，避免程序中断
            if html_data_info is None:
                continue
            soup_data_info = BeautifulSoup(html_data_info, 'html.parser')
            if soup_data_info.find('div', class_='none_txt') is not None:
                print(new_music_url, '   对不起，由于版权原因，暂无法查看该专辑！')
                continue
            mod_songlist_div = soup_data_info.find('div', class_='mod_songlist')
            songlist_ul = mod_songlist_div.find('ul', class_='songlist__list')
            lis = songlist_ul.find_all('li')
            del lis[0]  # 删除第一个li
            # print('len(lis):$s' % len(lis))
            for li in lis:
                a_songname_txt = li.find('div', class_='songlist__songname').find('span', class_='songlist__songname_txt').find('a')
                if 'https' not in a_songname_txt['href']:  #如果单曲链接不包含协议头，加上
                    song_url = 'https:' + a_songname_txt['href']
                song_name = a_songname_txt['title']
                singer_name = li.find('div', class_='songlist__artist').find('a').get_text()
                song_time =li.find('div', class_='songlist__time').get_text()
                music_info = {}
                music_info['song_name'] = song_name
                music_info['song_url'] = song_url
                music_info['singer_name'] = singer_name
                music_info['song_time'] = song_time
                collect_data(music_info)
     except Exception as err:  # 如果解析异常，跳过
         print('Downloading or parse music information error continue:', err.args)

def write_to_excel(self, content):
    try:
        for row in content:
            self.workSheet.append([row['song_name'], row['song_url'], row['singer_name'], row['song_time']])
        self.workBook.save(self.excelName)  # 保存单曲信息到Excel文件
    except Exception as arr:
        print('write to excel error', arr.args)