# -*- coding:utf-8 -*-
import sys
import importlib

importlib.reload(sys)

import pandas as pd
import os
import yt_dlp
import random
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# 搜索接口
url = 'https://www.douyin.com/aweme/v1/web/search/item/'

search_keyword = '丽江'  # 搜索词
pageSize = 10  # 接口用，一页多少条
needCount = 10  # 需要的数量，pageSize的整数倍，不要超过maxCount

maxCount = 30  # 限制最大数量

# 获取当前日期和时间
now = datetime.now()
formatted_datetime = now.strftime('%Y年%m月%d日%H时%M分%S秒')

# 保存链接
file = 'F:\\douyin\\' + search_keyword + '-' + formatted_datetime + '.xlsx'

# cookie设置，从浏览器中拷贝
# 网页 https://www.douyin.com/search/%E5%A4%A7%E6%98%8E%E5%B1%B1?aid=0e3b6537-25af-4178-a544-d03ee89420ad&type=video
# 接口 https://www.douyin.com/aweme/v1/web/search/item/
cookie = 'ttwid=1|dDQ9BfZ8fF_bkoruTF6sbNjsB4hUWytdIueLTxLxSIg|1719221618|6775e5240f9e7162e0a31a43f6e827a94b0a12b05a060599519be1b63f84e7b3; UIFID_TEMP=b4fba672f3367a6b793dc899f2c6d1b4408dacb6e3db75976f87f84aa96ca132d2aac2d4c5200e71e073e3b5702e1682ccb227ee63ced814c4eaea5f1b598b32cd7ddaeb58b0e35a137a4f85c6779914; s_v_web_id=verify_lxss5ue1_Jmn5uuLh_dMUF_4wG9_941J_9J0L48UJhzRn; douyin.com; device_web_cpu_core=12; device_web_memory_size=8; architecture=amd64; home_can_add_dy_2_desktop="0"; dy_swidth=1920; dy_sheight=1080; csrf_session_id=430e4d2baec9238f69560d2547115edd; fpk1=U2FsdGVkX1/SaVsOVxZKCPiYhL9cY03NbqMYNqI/FCBN73JZWL/UrYseuRcDsN0R+VNfpxwoxzoTbKgXq/7V3A==; fpk2=5586416403aec5e1793de81a0e867574; odin_tt=88094437121e58046c52fed6012892e0ce220999931d97fb61cefa5ffdfe1c9b1beab6e27323a4f99ea19c7471b75a4efb7e0a017196a1eb2b4ecd9a8da18eba3f37a0447cf9ad8ca248a6bb168395c9; passport_csrf_token=f6f5130af4cd662aa9881fe18606c889; passport_csrf_token_default=f6f5130af4cd662aa9881fe18606c889; FORCE_LOGIN={"videoConsumedRemainSeconds":180}; bd_ticket_guard_client_web_domain=2; SEARCH_RESULT_LIST_TYPE="single"; x-web-secsdk-uid=8c6b064e-4dfa-4db8-a71d-8b05ba39e860; UIFID=b4fba672f3367a6b793dc899f2c6d1b4408dacb6e3db75976f87f84aa96ca132d2aac2d4c5200e71e073e3b5702e1682edcdafb8d536bc6f522fa5e109fc98afb2803a7af610586de5564b5f60c223649d80069ed903d8820efd44ef2c18862e37209f5c061bd273b14f9d41b58ae847b1c6b6506a3d9f25952538a563d94fda06da5706f4465e43beb6e48606ebe5c55e3cd8cc59e6a6348d2c351fc866e84b; strategyABtestKey="1719300376.097"; volume_info={"isUserMute":false,"isMute":true,"volume":0.5}; __ac_signature=_02B4Z6wo00f01mLaUywAAIDDAdCTRf.RvtZi-leAAP7bc6; stream_recommend_feed_params="{\"cookie_enabled\":true,\"screen_width\":1920,\"screen_height\":1080,\"browser_online\":true,\"cpu_core_num\":12,\"device_memory\":8,\"downlink\":10,\"effective_type\":\"4g\",\"round_trip_time\":50}"; bd_ticket_guard_client_data=eyJiZC10aWNrZXQtZ3VhcmQtdmVyc2lvbiI6MiwiYmQtdGlja2V0LWd1YXJkLWl0ZXJhdGlvbi12ZXJzaW9uIjoxLCJiZC10aWNrZXQtZ3VhcmQtcmVlLXB1YmxpYy1rZXkiOiJCT2txRXArRWhEVSs5V0tJaCt0RWduQ1piS2tzNTI1Q0lDdDRtQzUrSmFsY1o1d1FheGF1dlNZTlJTVmFyNHhWeUlqRGV5OTZvSVhscUFNK1RRMDVzNDQ9IiwiYmQtdGlja2V0LWd1YXJkLXdlYi12ZXJzaW9uIjoxfQ==; download_guide="3/20240625/0"; WallpaperGuide={"showTime":1719221652787,"closeTime":0,"showCount":1,"cursor1":13,"cursor2":0}; stream_player_status_params="{\"is_auto_play\":0,\"is_full_screen\":0,\"is_full_webscreen\":0,\"is_mute\":1,\"is_speed\":1,\"is_visible\":0}"; IsDouyinActive=true; msToken=QmiuYBMMFHJcpajzqkeb7xp6yh18cFO21Yx4GZTCzy_3BKQX4Fgwgw7FHRABvUln7mOeqN81169VDDdGqTv3VBmFX5i-8EPoRXtKR0zu0bN7sFNCpYn3'

# # 随机选择一个user-agent
# user_agents = [
#     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
#     'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0',
#     'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) CriOS/56.0.2924.75 Mobile/14E5239e Safari/602.1',
#     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393'
# ]
ua = UserAgent()
# random_user_agent = random.choice(user_agents)
random_user_agent = ua.firefox

# 请求头
h1 = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cookie': cookie,
    'referer': 'https://www.douyin.com/search/%E5%A4%A7%E6%98%8E%E5%B1%B1?type=video',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'content-type': 'application/json; charset=utf-8',
    'user-agent': random_user_agent
}


# h2 = {
#     'cookie': cookie,
#     'referer': 'https://www.douyin.com/search/%E5%A4%A7%E6%98%8E%E5%B1%B1?type=video',
#     'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1'
# }

class CustomError(Exception):
    """自定义异常类"""
    pass


if needCount > maxCount:
    raise CustomError("所需数量不能超过" + maxCount)

if needCount < pageSize:
    pageSize = needCount

# 请求参数
params = {
    "device_platform": "webapp",
    "aid": "6383",
    "channel": "channel_pc_web",
    "search_channel": "aweme_video_web",
    "enable_history": "1",
    "keyword": search_keyword,
    "search_source": "switch_tab",
    "query_correct_type": "1",
    "is_filter_search": "0",
    "from_group_id": "",
    "offset": 0,
    "count": pageSize,
    "search_id": "20240626103443B38013E4CB13514676FB",
    "need_filter_settings": "1",
    "list_type": "single",
    "update_version_code": "170400",
    "pc_client_type": "1",
    "version_code": "170400",
    "version_name": "17.4.0",
    "cookie_enabled": "true",
    "screen_width": "414",
    "screen_height": "896",
    "browser_language": "zh-CN",
    "browser_platform": "Win32",
    "browser_name": "Mobile Safari",
    "browser_version": "13.0.3",
    "browser_online": "true",
    "engine_name": "WebKit",
    "engine_version": "605.1.15",
    "os_name": "iOS",
    "os_version": "13.2.3",
    "cpu_core_num": "12",
    "device_memory": "8",
    "platform": "iPhone",
    "downlink": "10",
    "effective_type": "4g",
    "round_trip_time": "50",
    "webid": "7384000566172599859",
    "msToken": "yJqrUwFy1RCr7T8lBwEEsS-SJjnKW9by6ND_0sI3QDzyM61OjzAvocsULWJ3Fn9DAAJObKC7E_K8LGqZLN62YiUjujxv7yTaRh9kpMzYvCKTeE4WaJcg",
    "a_bogus": "Dfm0QmhvDkVkvDSk5U5LfYpq6fq3Y2YI0CPYMD2fLdf4Og39HMTt9exEJ3GvRMEjNs/DIeYjy4hjOpPMEOAn0ZwXHWfKl2Ak-g00t-PD-9Uj-HhHuy8snsJP4vE3tee/svrIi/igw7lHFmupAnAJ5kIlO62-zo0/9-u="
}


def get_proxy_ips():
    # url = 'https://www.zdaye.com/free/'
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    # }
    # response = requests.get(url, headers=headers)
    # soup = BeautifulSoup(response.text, 'html.parser')
    # proxy_ips = ['183.234.215.11:8443', '47.96.176.130:59394', '47.93.249.121:8118', '120.25.1.15:7890', '121.227.31.32:8118', '49.7.11.187:80', '118.190.142.208:80', '39.105.27.30:3128', '223.100.178.167:9091', '106.227.87.11:3128', '60.205.132.71:80', '122.136.212.132:53281', '39.191.223.9:3128', '47.93.121.200:80', '39.106.60.216:3128', '121.37.107.186:1080', '221.231.13.198:1080', '202.117.115.6:80', '203.110.176.69:8111', '125.77.25.177:8080']
    proxy_ips = [  # ip存放地址
        '112.19.241.37:19999',
        '58.210.227.210:8088',
    ]

    # proxy_ips = []
    # for row in soup.find_all('tr'):
    #     columns = row.find_all('td')
    #     if len(columns) >= 2:
    #         ip = columns[0].text
    #         port = columns[1].text
    #         proxy_ips.append(f'{ip}:{port}')
    return proxy_ips


# 代理ip请求
proxy_ips = get_proxy_ips()


def get_random_proxy():
    ip = random.choice(proxy_ips)
    print(ip)
    return ip


proxy = {
    'https': 'http://' + get_random_proxy()
}

index = 0

if needCount % pageSize == 0:
    index = int(needCount / pageSize)
else:
    index = int(needCount / pageSize) + 1

print(index)
# 定义空列表
video_no_list = []  # 视频编号
title_list = []  # 视频标题
author_name_list = []  # 作者昵称
link_web_list = []  # 视频网页链接
link_1080_list = []  # 视频链接
link_720_list = []  # 视频链接

for num in range(index):
    print(num)
    # 发送请求
    params['offset'] = num * pageSize
    response = requests.get(url, headers=h1, params=params, stream=True, proxies=proxy)
    # print(response.status_code)
    print(response.text)
    # 以json格式接收返回数据
    json_data = response.json()
    video_list = json_data['data']
    for v in video_list:
        video_no = v['aweme_info']['aweme_id']
        title = v['aweme_info']['desc']
        linkWeb = "https://www.douyin.com/video/" + v['aweme_info']['aweme_id']
        author_name = v['aweme_info']['author']['nickname']
        link_1080 = v['aweme_info']['video']['bit_rate'][0]['play_addr']['url_list'][0]
        if len(v['aweme_info']['video']['bit_rate']) > 1:
            link_720 = v['aweme_info']['video']['bit_rate'][1]['play_addr']['url_list'][0]
        else:
            link_720 = ''
        print('视频标题:' + title)
        video_no_list.append(video_no)
        title_list.append(title)
        link_web_list.append(linkWeb)
        author_name_list.append(author_name)
        link_1080_list.append(link_1080)
        link_720_list.append(link_720)
        # 下载1080视频
        # 保存路径及文件名
        output_path = 'F:\\douyin\\' + search_keyword + '-' + formatted_datetime + '-' + v['aweme_info'][
            'aweme_id'] + '.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'merge_output_format': 'mp4',
            'http_headers': h1
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link_1080])
        # 延迟5秒
        # time.sleep(5)

# 保存视频信息到DF
df = pd.DataFrame(
    {
        '视频编号': video_no_list,
        '视频标题': title_list,
        '作者昵称': author_name_list,
        '视频网页链接': link_web_list,
        '视频链接1080': link_1080_list,
        '视频链接720': link_720_list
    }
)
# 保存csv excel
if os.path.exists(file):
    header = False
else:
    header = True
df.to_csv(file, mode='secrets.toml+', index=False, header=header, encoding='utf_8_sig')

exit()
