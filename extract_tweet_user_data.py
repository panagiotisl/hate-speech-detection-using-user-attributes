import tweepy
import csv
from configuration import log_config, twitter_login_config

logger = log_config.logger

# Old Dataset file
dataset_file = "dataset/data.csv"

# New dataset file
tweet_data_file = "dataset/tweet_user_data.csv"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(twitter_login_config.CONSUMER_KEY, twitter_login_config.CONSUMER_SECRET)
auth.set_access_token(twitter_login_config.ACCESS_TOKEN, twitter_login_config.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
with open(dataset_file, 'r') as read_file, open(tweet_data_file, "w", encoding="utf-8", newline='') as write_file:
    lineReader = csv.reader(read_file, delimiter=',')
    fieldnames = [
        "tweet_id", "tweet_retweets", "tweet_likes", "tweet_is_reply",
        "user_id", "user_following", "user_followers", "user_total_tweets", "is_hate", "is_positive_negative_neutral",
        "tweet_text"
    ]
    writer = csv.DictWriter(write_file, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    tweet_parsing_errors = 0
    total = 0
    for row in lineReader:
        total += 1
        try:
            tweet = api.get_status(row[0], tweet_mode="extended")
        except tweepy.TweepError as e:
            tweet_parsing_errors += 1
        else:
            tweet_is_reply = 0 if tweet.in_reply_to_status_id is None else 1
            writer.writerow({"tweet_id": tweet.id,
                             "tweet_retweets": tweet.retweet_count,
                             "tweet_likes": tweet.favorite_count,
                             "tweet_is_reply": tweet_is_reply,
                             "user_id": tweet.user.id,
                             "user_following": tweet.user.friends_count,
                             "user_followers": tweet.user.followers_count,
                             "user_total_tweets": tweet.user.statuses_count,
                             "is_hate": row[8],
                             "is_positive_negative_neutral": row[9],
                             "tweet_text": tweet.full_text
                             })
            write_file.flush()
        finally:
            if total % 500 == 0:
                logger.info(f"processed {total} tweets with {tweet_parsing_errors} total exceptions in tweets parsing.")
    logger.info(f"processed {total} tweets with {tweet_parsing_errors} total exceptions in tweets parsing.")
read_file.close()


