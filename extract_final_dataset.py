import tweepy
import csv
import login_credentials as creds

# Initial dataset file
dataset_file = "dataset/NAACL_SRW_2016.csv"

# Tweets that no longer exist or not accessible
bad_tweets = "dataset/bad_tweets.csv"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(creds.CONSUMER_KEY, creds.CONSUMER_SECRET)
auth.set_access_token(creds.ACCESS_TOKEN, creds.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

with open(dataset_file, 'r') as read_file, open(bad_tweets, "w", newline='') as write_file:
    lineReader = csv.reader(read_file, delimiter=',')
    writer = csv.writer(write_file, delimiter=',')
    for row in lineReader:
        try:
            tweet = api.get_status(row[0])
        except tweepy.TweepError as e:
            writer.writerow(row)
            write_file.flush()

read_file.close()
write_file.close()








