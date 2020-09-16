import numpy as np
import pandas as pd

df = pd.read_csv("dataset/tweet_user_data.csv")

# select attributes from "user_following", "user_followers", "user_total_tweets", "tweet_retweets"
# i.e single attribute
# df = df[["tweet_retweets", "tweet_label"]]
# i.e combination of attributes
df = df[["user_following", "user_followers", "user_total_tweets", "tweet_label"]]

prediction_column = "tweet_label"

X = np.array(df.drop([prediction_column], 1))
y = np.array(df[prediction_column])

# test size (default 80/20)
test_size = 0.2

# how many train cycles (default 20)
cycles = 20

# KNN
knn_neighbors = 15

