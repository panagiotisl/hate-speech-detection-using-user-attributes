import numpy as np
import pandas as pd

# #################################################################################################################
#   FEATURES SELECTION
###################################################################################################################
#   select attributes from
#   "user_following", "user_followers", "user_total_tweets", "tweet_retweets", "tweet_likes", "tweet_is_reply"
#   single attribute --> i.e attribute_columns = [f1]
#   combination of attributes --> i.e attribute_columns = [f1, f3]
# #################################################################################################################

f1 = "tweet_retweets"
f2 = "tweet_likes"
f3 = "user_following"
f4 = "user_followers"
f5 = "user_total_tweets"
f6 = "tweet_is_reply"

# #################################################################################################################
#   LABEL/CLASS SELECTION
###################################################################################################################
#   select column to predict from
#   "is_hate", "is_positive_negative_neutral"
# #################################################################################################################

p1 = "is_hate"
p2 = "is_positive_negative_neutral"

# features
attribute_columns = [f3]

# label
prediction_column = p2

# columns for the dataset
columns = attribute_columns + [prediction_column]

# read csv and keep only the columns needed
df = pd.read_csv("dataset/tweet_user_data_final.csv")
df = df[columns]

# create numpy arrays for the datasets
X = np.array(df.drop([prediction_column], 1))
# X[:, 0] = np.where(X[:, 0] < 2, 1, 100)
# X[:, 1] = np.where(X[:, 1] < 2, 1, 100)
# X[:, 2] = np.where(X[:, 2] < 750, 1, 10000)
# X[:, 3] = np.where(X[:, 3] < 12000, 1, 100000)
# X[:, 4] = np.where(X[:, 4] < 15000, 1, 100000)

y = np.array(df[prediction_column])
