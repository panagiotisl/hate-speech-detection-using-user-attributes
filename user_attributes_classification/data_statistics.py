import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

f1 = "tweet_retweets"
f2 = "tweet_likes"
f3 = "user_following"
f4 = "user_followers"
f5 = "user_total_tweets"
f6 = "tweet_is_reply"
label = "is_hate"

columns = [f1, f2, f3, f4, f5, f6, label]


# HOW MANY REPLIES
def produce_replies_plot(df):
    fig = sns.barplot(x=df[f6].value_counts().index, y=df[f6].value_counts()).get_figure()
    fig.savefig("../statistics_plots/" + f6 + ".png")


# GENERAL INFORMATION
def dataframe_general_info(df):
    print(df.describe)


# HOW MANY OUTLIERS (PLOTS)
def produce_outliers_plots(df):
    for c in columns:
        if c != "tweet_is_reply":
            fig = sns.boxplot(x=df[c]).get_figure()
            fig.savefig("../statistics_plots/outliers_plots/" + c + ".png")


# SCATTER PLOTS
def produce_scatter_plots(df):
    for c in columns:
        if c != "tweet_is_reply" and c != "is_hate":
            fig = sns.scatterplot(x=df.index, y=df[c], hue=df[label]).get_figure()
            fig.savefig("../statistics_plots/" + c + ".png")


def produce_is_hate_plot(df):
    fig = sns.barplot(x=df[label].value_counts().index, y=df[label].value_counts()).get_figure()
    fig.savefig("../statistics_plots/" + label + ".png")


# ZEROS IN EACH FEATURE
def zeros_per_column(df):
    print(df[f1].loc[df[f1] == 0].count()/df[f1].count())
    print(df[f2].loc[df[f2] == 0].count()/df[f2].count())
    print(df[f3].loc[df[f3] == 0].count()/df[f3].count())
    print(df[f4].loc[df[f4] == 0].count()/df[f4].count())
    print(df[f5].loc[df[f5] == 0].count()/df[f5].count())


# ARE THERE NULL VALUES?
def check_null_values(df):
    nul = 0
    for col in df.columns:
        if df[col].isnull().any():
            print(col, 'has null values')
            nul = 1
    if nul == 0:
        print('There are no Null values present')


df = pd.read_csv("../dataset/tweet_user_data_final.csv")
df = df[columns]

dataframe_general_info(df)
zeros_per_column(df)
check_null_values(df)
produce_scatter_plots(df)
produce_outliers_plots(df)
produce_replies_plot(df)
produce_is_hate_plot(df)
