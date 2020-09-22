import numpy as np
import pandas as pd

# ##################################################################################################
#   FEATURES SELECTION
#   select attributes from "user_following", "user_followers", "user_total_tweets", "tweet_retweets"
#   single attribute --> i.e attribute_columns = [f1]
#   combination of attributes --> i.e attribute_columns = [f1, f3]
# ##################################################################################################

f1 = "tweet_retweets"
f2 = "user_following"
f3 = "user_followers"
f4 = "user_total_tweets"

# features
attribute_columns = [f1, f2]

# label
prediction_column = "tweet_label"

# columns for the dataset
columns = attribute_columns + [prediction_column]

# read csv and keep only the columns needed
df = pd.read_csv("dataset/tweet_user_data.csv")
df = df[columns]

# create numpy arrays for the datasets
X = np.array(df.drop([prediction_column], 1))
y = np.array(df[prediction_column])

