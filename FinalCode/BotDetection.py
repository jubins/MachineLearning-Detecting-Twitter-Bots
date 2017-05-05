import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
mpl.rcParams['patch.force_edgecolor'] = True
warnings.filterwarnings("ignore")
# %matplotlib inline


# Cross-validating the training data
def perform_cross_validation(train_df):
    X = train_df
    y = train_df.ix[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return (X_train, X_test, y_train, y_test)

# Gives heatmap of null values
def get_heatmap(df):
    # This function gives heatmap of all NaN values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.tight_layout()
    return plt.show()


def twitter_bot_predictor(df):
    # creating copy of dataframe
    train_df = df.copy()

    # converting id to int
    train_df['id'] = train_df.id.apply(lambda x: int(x))

    # check if screen_name_count has bot
    condition = (train_df.screen_name.str.contains("bot", case=False) == True)
    predicted_df = train_df[condition]  # 206 these all are bots
    predicted_df.bot = 1
    predicted_df = predicted_df[['id', 'bot']]

    # check if listed_count>16000
    listed_count_df = train_df[~condition]
    listed_count_df.listed_count = listed_count_df.listed_count.apply(lambda x: 0 if x == 'None' else x)
    listed_count_df.listed_count = listed_count_df.listed_count.apply(lambda x: int(x))
    condition = (listed_count_df.listed_count > 16000)  # these all are nonbots
    # listed_count_df[condition].shape #these all are nonbots
    predicted_df1 = listed_count_df[condition][['id', 'bot']]
    predicted_df1.bot = 0
    predicted_df = pd.concat([predicted_df, predicted_df1])

    # check if the user is verified
    verified_df = listed_count_df[~condition]
    condition = (verified_df.verified == 'TRUE')  # these all are nonbots
    predicted_df1 = verified_df[condition][['id', 'bot']]
    predicted_df1.bot = 0
    predicted_df = pd.concat([predicted_df, predicted_df1])

    # check if description contains bot
    description_df = verified_df[~condition]
    condition = description_df.description.str.contains("bot", case=False, na=False)  # these all are bots
    predicted_df1 = description_df[condition][['id', 'bot']]
    predicted_df1.bot = 1
    predicted_df = pd.concat([predicted_df, predicted_df1])

    # check if status contain bot
    status_df = description_df[~condition]
    condition = (status_df.status.str.contains("bot", case=False, na=False))  # these all are bots
    predicted_df1 = status_df[condition][['id', 'bot']]
    predicted_df1.bot = 1
    predicted_df = pd.concat([predicted_df, predicted_df1])

    # check if description contains buzzfeed
    buzzfeed_df = status_df[~condition]
    condition = (buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False))  # these all are nonbots
    predicted_df1 = buzzfeed_df[buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False)][['id', 'bot']]
    predicted_df1.bot = 0
    predicted_df = pd.concat([predicted_df, predicted_df1])

    # check if the name contains bot or screenname contains b0t
    name_df = buzzfeed_df[~condition]
    condition = ((name_df.name.str.contains("bot", case=False, na=False)) |
                 (name_df.screen_name.str.contains("b0t", case=False, na=False)))  # these all are bots
    predicted_df1 = name_df[condition][['id', 'bot']]
    predicted_df1.bot = 1
    predicted_df = pd.concat([predicted_df, predicted_df1])

    # check if screename contains truth, name contains truth and screenname contains anony
    truth_df = name_df[~condition]  # shape: 83

    condition = ((truth_df.screen_name.str.contains("truth", case=False, na=False)) |
                 (truth_df.name.str.contains("truth", case=False, na=False)) |
                 (truth_df.screen_name.str.contains("anony", case=False, na=False)))  # these all are bots

    predicted_df1 = truth_df[condition][['id', 'bot']]
    predicted_df1.bot = 1
    predicted_df = pd.concat([predicted_df, predicted_df1])

    # check if statuses
    statuses_count_df = truth_df[~condition]
    condition = ((statuses_count_df.description.str.contains('cannabis', case=False, na=False)) |
                 (statuses_count_df.description.str.contains('mishear', case=False, na=False)))  # these all are bots
    predicted_df1 = statuses_count_df[condition][['id', 'bot']]
    predicted_df1.bot.replace(to_replace=np.nan, value='1', inplace=True)
    predicted_df = pd.concat([predicted_df, predicted_df1])

    predicted_df1 = statuses_count_df[~condition][['id', 'bot']]  # these all are nonbots
    predicted_df1.bot = 0
    predicted_df = pd.concat([predicted_df, predicted_df1])

    return predicted_df


filepath = 'https://raw.githubusercontent.com/jubins/ML-TwitterBotDetection/master/FinalCode/kaggle_data/'
train_df = pd.read_csv(filepath + 'training_data_2_csv_UTF.csv')

#Performing cross validation the training data
(X_train, X_test, y_train, y_test) = perform_cross_validation(train_df)

#Running our own twitter_bot_predictor algorithm
#print (twitter_bot_predictor(X_train).values)

# test_df = pd.read_csv(filepath + 'test_data_4_students.csv', sep='\t')
# predicted_df = twitter_bot_predictor(test_df)
# preparing subission file
# final_df.to_csv('submission.csv', index=False)
print(X_train)
