# Hate speech detection
This project is based on the paper 
["Are They Our Brothers? Analysis and Detection of Religious Hate Speech in the Arabic Twittersphere"](https://ieeexplore.ieee.org/document/8508247), 
Albadi, Nuha and Kurdi, Maram and Mishra, Shivakant (2018).
 
In this research, we will try to expand the knowledge that the previous authors had acquired 
and we will attempt to follow a new approach. Specifically, we will try to detect hate speech in
tweets, analyzing the user's profile using any metrics we may find useful. Therefore, we move
to a Social Network analysis.

### Dataset
For our research we 'll be using the dataset from the previously mentioned papers.
Some data from the original dataset were inaccessible. Therefore, we use `extract_tweet_user_data.py` to obtain our new dataset
`tweet_user_data.csv`. The dataset includes the following information:
* tweet_id
* tweet_retweets
* user_following
* user_followers
* user_total_tweets
* tweet_label
Finally, we removed duplicate tweets and kept only distinct users in our dataset to avoid biased results.

### Usage
To use one of the algorithms (i.e SVM, KNN) you have to do the following:
1. go to `data_config.py` and:
    * choose the feature(s) that you want to train and test your data with.
2. go to `classification_config.py` and:
    * choose how many train cycles you want to run (def. 20) changing the appropriate variable.
    * choose the size of your test dataset (def. 80/20) changing the appropriate variable.
    * choose classification algorithm.
2. simply run `classification.py` to get the results.
