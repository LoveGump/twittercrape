import tweepy
import requests
import os

# Twitter API 认证信息
CONSUMER_KEY = "***"
CONSUMER_SECRET = "***"
ACCESS_TOKEN = "***"
ACCESS_TOKEN_SECRET = "***"

# 初始化认证
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


def download_media(url, filename):
    """下载媒体文件（图片/视频）"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"下载完成: {filename}")
    else:
        print(f"下载失败: {url}")


def crawl_tweets(keyword, count=10):
    """爬取指定关键词的推文"""
    tweets = api.search_tweets(q=keyword, tweet_mode="extended", count=count)

    for i, tweet in enumerate(tweets):
        print(f"\n=== 推文 {i + 1} ===")
        print("用户:", tweet.user.screen_name)
        print("文本:", tweet.full_text)

        # 保存文本
        with open(f"tweet_{i + 1}.txt", "w", encoding="utf-8") as f:
            f.write(tweet.full_text)

        # 提取媒体（图片/视频）
        if 'media' in tweet.entities:
            media = tweet.extended_entities['media']
            for j, item in enumerate(media):
                if item['type'] == 'photo':
                    # 下载图片
                    url = item['media_url']
                    filename = f"tweet_{i + 1}_image_{j + 1}.jpg"
                    download_media(url, filename)
                elif item['type'] == 'video':
                    # 下载视频（选择最高质量）
                    video_info = item['video_info']['variants']
                    video_url = max(
                        [v for v in video_info if v['content_type'] == 'video/mp4'],
                        key=lambda x: x.get('bitrate', 0)
                    )['url']
                    filename = f"tweet_{i + 1}_video_{j + 1}.mp4"
                    download_media(video_url, filename)


if __name__ == "__main__":
    crawl_tweets("cryptocurrency", count=100)
