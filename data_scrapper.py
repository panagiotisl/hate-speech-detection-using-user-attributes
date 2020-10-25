import csv
import tweepy
from configuration import log_config, twitter_login_config

logger = log_config.logger

# store date in data_file
data_file = "dataset/data.csv"

# number of tweets to keep
TWEETS_NUM = 100

# Authenticate to Twitter
auth = tweepy.OAuthHandler(twitter_login_config.CONSUMER_KEY, twitter_login_config.CONSUMER_SECRET)
auth.set_access_token(twitter_login_config.ACCESS_TOKEN, twitter_login_config.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)


def fetch_data():
    with open(data_file, "a", encoding="utf-8", newline='') as write_file:
        fieldnames = [
            "tweet_id", "tweet_text"
        ]
        writer = csv.DictWriter(write_file, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        # search for tweets with controversial terms
        for tweet in tweepy.Cursor(api.search, q="#trump", lang="en", tweet_mode="extended").items(TWEETS_NUM):
            writer.writerow({"tweet_id": tweet.id,
                             "tweet_text": tweet.full_text
                             })
            write_file.flush()


fetch_data()

