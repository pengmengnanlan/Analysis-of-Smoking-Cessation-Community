import tweepy
import csv
import json
import time

# file "twitter_credentials.json" consists of four access keys of Twitter APIs
# we do not provide the keys here
with open('twitter_credentials.json') as cred_data:
	info = json.load(cred_data)
	consumer_key = info['CONSUMER_KEY']
	consumer_secret = info['CONSUMER_SECRET']
	access_key = info['ACCESS_KEY']
	access_secret = info['ACCESS_SECRET']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# maximum_number_of_tweets_to_be_extracted = int(input('Enter the number of tweets that you want to extract- '))
hashtag = input("Enter the hashtag you want to scrape- ")

with open('./tweets_with_hashtag_'+ hashtag + '.csv', 'a', encoding = 'utf-8') as the_file:
	writer = csv.writer(the_file)
	writer.writerow(['userId',"userName", 'created_at', 'message',"location","retweeted_user_id","followers"])
	backoff_counter = 1
	while True:
		try:
			for tweet in tweepy.Cursor(api.search, q='#'+hashtag, rpp=100).items():
				#writer.writerow(['id', 'created_at', 'text'])
				#writer.writerows
				# the_file.write(str(tweet.text.encode('utf-8'))+ '\n')
				ids = []
				for page in tweepy.Cursor(api.followers_ids, screen_name=tweet.user.screen_name).pages():
					ids.extend(page)
					time.sleep(5)
					# print(len(page),"added")
				print(len(ids))
				try:
					#do some thing you need
					retweeted_status = str(tweet.retweeted_status.user.id)
				except AttributeError as e:
					#error: has not attribute
					retweeted_status = "none"
					
				writer.writerow([tweet.user.id,tweet.user.screen_name ,tweet.created_at, tweet.text.encode('utf-8'),tweet.user.location,retweeted_status,str(ids)])
		except tweepy.TweepError as e:
			print(e.reason)
			time.sleep(60)
			continue
		
# print('Extracted '+str(maximum_number_of_tweets_to_be_extracted)+' tweets with hashtag #' + hashtag)
