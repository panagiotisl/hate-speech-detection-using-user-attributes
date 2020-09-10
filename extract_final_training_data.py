import tweepy
import csv
from configuration import log_config, twitter_login_config

logger = log_config.logger

# Dataset file
dataset_file = "dataset/training_data_previous_attributes.csv"

# New dataset file with more data for training
tweet_data_file = "dataset/final_training_data.csv"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(twitter_login_config.CONSUMER_KEY, twitter_login_config.CONSUMER_SECRET)
auth.set_access_token(twitter_login_config.ACCESS_TOKEN, twitter_login_config.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

with open(dataset_file, 'r') as read_file, open(tweet_data_file, "w", encoding="utf-8") as write_file:
    lineReader = csv.reader(read_file, delimiter=',')
    fieldnames = [
        "created_at", "tweet_id", "tweet_text", "tweet_hashtags", "tweet_retweets", "tweet_reply_to_user",
        "user_id", "user_name", "user_screen_name", "user_descr",
        "user_following", "user_followers", "user_location", "user_total_tweets", "user_created_at", "category"
        ]
    writer = csv.DictWriter(write_file, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    for row in lineReader:
        try:
            tweet = api.get_status(row[0])
            hashtags = []
            for hashtag in tweet.entities['hashtags']:
                hashtags.append(hashtag['text'])
            writer.writerow({"created_at": tweet.created_at,
                             "tweet_id": tweet.id,
                             "tweet_text": tweet.text,
                             "tweet_hashtags": hashtags,
                             "tweet_retweets": tweet.retweet_count,
                             "tweet_reply_to_user": tweet.in_reply_to_user_id,
                             "user_id": tweet.user.id,
                             "user_name": tweet.user.name,
                             "user_screen_name": tweet.user.screen_name,
                             "user_descr": tweet.user.description,
                             "user_location": tweet.user.location,
                             "user_following": tweet.user.friends_count,
                             "user_followers": tweet.user.followers_count,
                             "user_total_tweets": tweet.user.statuses_count,
                             "user_created_at": tweet.user.created_at,
                             "category": row[0]
                             })
        except tweepy.TweepError as e:
            logger.warning(f"Tweet with id {row[0]} caused exception {e}")

read_file.close()
