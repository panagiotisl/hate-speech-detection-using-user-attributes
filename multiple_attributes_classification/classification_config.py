# vectors dimension for the embedding
EMBEDDING_DIM = 50
# validation data size (i.e 0.2 = 20%)
VALIDATION_SIZE = 0.25
# epochs to train the neural network
EPOCHS = 7
# test size (i.e 0.2 = 20%)
TEST_SIZE = 0.25

# choose classification attributes below and add them to the 'chosen_attributes_list'
f1 = "tweet_retweets"
f2 = "tweet_likes"
f3 = "user_following"
f4 = "user_followers"
f5 = "user_total_tweets"
f6 = "tweet_is_reply"
chosen_attributes_list = [f1, f2, f3, f5, f5, f6]
