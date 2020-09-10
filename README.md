# Hate speech detection
This project is based on the papers:
 * [Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter](https://www.aclweb.org/anthology/N16-2013/)
 * [Deep Learning for Hate Speech Detection in Tweets](https://dl.acm.org/doi/10.1145/3041021.3054223)

In this research, we will try to expand the knowledge that the previous authors had acquired 
and we will attempt to follow a new approach. Specifically, we will try to detect hate speech in
tweets, analyzing the user's profile using any metrics we may find useful. Therefore, we move
from NLP analysis to Social Network analysis.

### Dataset
For our research we 'll be using the [dataset](https://github.com/zeerakw/hatespeech) from the previously mentioned papers.
Some data from the original dataset were inaccessible. Therefore, we use `remove_non_existent_tweets.py`
to find all the tweets that caused an exception. Most of the time, the exception was 
*'code': 63, 'message': 'User has been suspended'*. After extracting the `bad_tweets.csv` file,
we compare it with `NAACL_SRW_2016.csv`, we remove the tweets that caused exceptions
and we obtain the dataset `training_data_previous_attributes.csv`, which contains the tweets that still exist.

From this file, we examine each tweet ID and we use the twitter API to extract all the information needed for our 
research and we will save it to our final training data file `final_training_data.csv`.
