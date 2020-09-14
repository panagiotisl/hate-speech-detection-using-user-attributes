import tweepy
import csv
from configuration import log_config, twitter_login_config

logger = log_config.logger

# Old Dataset file
dataset_file = "dataset/NAACL_SRW_2016.csv"

# New dataset file
tweet_data_file = "dataset/tweet_user_data.csv"


# Encode tweet label as 0 for 'none', 1 for 'racism' and 2 for 'sexism'
def tweet_label_encoding(label):
    if label == "none":
        return 0
    elif label == "racism":
        return 1
    else:
        return 2


# Authenticate to Twitter
auth = tweepy.OAuthHandler(twitter_login_config.CONSUMER_KEY, twitter_login_config.CONSUMER_SECRET)
auth.set_access_token(twitter_login_config.ACCESS_TOKEN, twitter_login_config.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
with open(dataset_file, 'r') as read_file, open(tweet_data_file, "w", encoding="utf-8", newline='') as write_file:
    lineReader = csv.reader(read_file, delimiter=',')
    # fieldnames = [
    #     "tweet_id", "tweet_text", "tweet_hashtags", "tweet_retweets", "tweet_reply_to_user",
    #     "user_id", "user_name", "user_screen_name", "user_description",
    #     "user_following", "user_followers", "user_total_tweets", "tweet_label"
    #     ]
    fieldnames = [
        "tweet_id", "tweet_retweets", "user_id", "user_following", "user_followers", "user_total_tweets", "tweet_label"
    ]
    writer = csv.DictWriter(write_file, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    tweet_parsing_errors = 0
    total = 0
    for row in lineReader:
        total += 1
        try:
            tweet = api.get_status(row[0])
        except tweepy.TweepError as e:
            tweet_parsing_errors += 1
        else:
            # hashtags = []
            # for hashtag in tweet.entities['hashtags']:
            #     hashtags.append(hashtag['text'])
            # tweet label encoding
            tweet_label = tweet_label_encoding(row[1])
            writer.writerow({"tweet_id": tweet.id,
                             # "tweet_text": tweet.text,
                             # "tweet_hashtags": hashtags,
                             "tweet_retweets": tweet.retweet_count,
                             # "tweet_reply_to_user": tweet.in_reply_to_user_id,
                             "user_id": tweet.user.id,
                             # "user_name": tweet.user.name,
                             # "user_screen_name": tweet.user.screen_name,
                             # "user_description": tweet.user.description,
                             "user_following": tweet.user.friends_count,
                             "user_followers": tweet.user.followers_count,
                             "user_total_tweets": tweet.user.statuses_count,
                             "tweet_label": tweet_label
                             })
            write_file.flush()
        finally:
            if total % 600 == 0:
                logger.info(f"processed {total} tweets with {tweet_parsing_errors} total exceptions in tweets parsing.")
    logger.info(f"processed {total} tweets with {tweet_parsing_errors} total exceptions in tweets parsing.")
read_file.close()


