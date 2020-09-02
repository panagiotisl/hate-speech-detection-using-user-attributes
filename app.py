import tweepy
import csv
import login_credentials as creds


dataset_file = "dataset/NAACL_SRW_2016.csv"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(creds.CONSUMER_KEY, creds.CONSUMER_SECRET)
auth.set_access_token(creds.ACCESS_TOKEN, creds.ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)



#tweets = [[]]
#
# with open(dataset_file, 'r') as csvfile:
#     lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
#     for row in lineReader:
#         try:
#             tweet = api.get_status(row[0])
#             tweets.append(tweet.text)
#         except tweepy.TweepError as e:
#             print("Tweet with id: " + row[0] + " not found")



