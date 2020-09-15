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
Some data from the original dataset were inaccessible. Therefore, we use `extract_tweet_user_data.py` to obtain our new dataset
`tweet_user_data.csv`.