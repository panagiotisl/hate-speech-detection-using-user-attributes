import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection
from sklearn.utils import shuffle

# columns = [
#         "tweet_id", "tweet_text", "tweet_hashtags", "tweet_retweets", "tweet_reply_to_user",
#         "user_id", "user_name", "user_screen_name", "user_descr",
#         "user_following", "user_followers", "user_total_tweets", "tweet_label"
#         ]

columns = [
        "tweet_id", "tweet_retweets", "user_id", "user_following", "user_followers", "user_total_tweets", "tweet_label"
        ]

df = pd.read_csv("dataset/tweet_user_data.csv", names=columns)

df = df[["tweet_retweets", "tweet_label"]]

prediction_column = "tweet_label"

X = np.array(df.drop([prediction_column], 1))
y = np.array(df[prediction_column])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

clf = linear_model.LinearRegression()

clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)

print(acc)
print(clf.intercept_)
print(clf.coef_)


