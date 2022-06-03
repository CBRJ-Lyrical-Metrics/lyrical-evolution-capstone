# Typical Everydayers...
import pandas as pd 
import numpy as np

# The Viz Folks...
import seaborn as sns
import matplotlib.pyplot as plt

# # Make deepcopy
# import copy

# Modeling help...
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Quieeet!!! Y'all can't stop me now...
import warnings
warnings.filterwarnings('ignore')

# Let me see it AAAALLLL!!!
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

# set default style for charts
plt.rc('figure', figsize=(13, 7))
plt.style.use('fivethirtyeight')


######################################## Topics Visuals ########################################

# Most popular topics...
def topic_popularity(df):
    df.topic_name.value_counts().plot(kind = 'bar')
    return

def show_topic_counts():
    pd.DataFrame(df.topic_name.value_counts())
    return

def all_topics_prevalence(df):   
    ax = sns.countplot(data = df, x = 'decade', hue = 'topic_name', ec = 'black')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad=0.)
    plt.title('Topics\' Prevalence Over the Decades')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
    plt.xlabel('Decade of Song')
    plt.ylabel('Song Count')
    plt.show()
    return

def relationship_bar(df):   
    # create a variable that stores a list relationship topics
    relationships = ['affection','breakups','love', 'breakup', 
                     'sex', 'heartache', 'jealousy']
    # make a copy
    train2 = df.copy()
    # add a column to the dataframe where any topic that is a relationship topic is gathered and all 
    # others are represented by 'other'
    train2['relationship_topics'] = np.where(train2.topic_name.isin(relationships), train2.topic_name, 'other')
    # drop anything that isn't a relationship topic
    train2 = train2.loc[train2['relationship_topics'] != 'other']
    train2.groupby('decade').relationship_topics.value_counts(normalize = True).unstack().plot(kind = 'bar', width = 1, ec = 'black')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad=0.)
    plt.title('Relationship Topics\' Prevalence Over the Decades')
    plt.xlabel('Decade of Song')
    plt.ylabel('Song Topic Count')
    plt.show()
    return

def relationships_swarm(df):  
    df5 = df.copy()
    df5['relationship_topics'] = np.where(df5['topic_name'].isin(['affection','love', 'sex', 
                                                         'heartache', 'jealousy','breakups']), df5['topic_name'], None)
    ax = sns.swarmplot(data = df5, x = 'relationship_topics', y = 'date')
    ax.set(title = 'While Love Has Remained A Constant Topic, Sex Has Replaced Affection In Lyrics')
    plt.ylabel('Decades')
    plt.xlabel('Relationship Topics')
    return

def love_swarm(df):  
    df5 = df.copy()
    # train3 = train3.sample(3_000)
    df5['love_v_sex'] = np.where(df5['topic_name'].isin(['affection','love', 'sex']), df5['topic_name'], 
                                                    None)
    ax = sns.swarmplot(data = df5, x = 'love_v_sex', y = 'date')
    ax.set(title = 'While Love Has Remained A Constant Topic, Sex Has Replaced Affection In Lyrics')
    plt.ylabel('Decades')
    plt.xlabel('Topics of Intimacy')
    return

def touch_swarm(df):  
    df6 = df.copy()
    # train3 = train3.sample(3_000)
    df6['affection_v_sex'] = np.where(df6['topic_name'].isin(['affection','sex']), df6['topic_name'], 
                                                    None)
    ax = sns.swarmplot(data = df6, x = 'affection_v_sex', y = 'date')
    ax.set(title = 'It Appears That Lyricists Have Become More Open To Full-on \'Sex\' Topics\nAs Opposed To More Subtle Lyrics Talking About \'Affection\'')
    plt.ylabel('Decades')
    plt.xlabel('Topics of Intimacy')
    return

def vice_bar(df):   
    # create a variable that stores a list relationship topics
    vices = ['sex', 'money', 'violence']
    # make a copy
    df3 = df.copy()
    # add a column to the dataframe where any topic that is a vices topic is gathered and all 
    # others are represented by 'other'
    df3['vice_topics'] = np.where(df3.topic_name.isin(vices), df3.topic_name, 'other')
    # drop anything that isn't a relationship topic
    df3 = df3.loc[df3['vice_topics'] != 'other']
    df3.groupby('decade').topic_name.value_counts(normalize = True).unstack().plot(kind = 'bar', width = 1, ec = 'black')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad=0.)
    plt.title('Vice Topics\' Prevalence Over the Decades')
    plt.xlabel('Decade of Song')
    plt.ylabel('Song Topic Count')
    plt.show()
    return

def vice_swarm(df):  
    df4 = df.copy()
    # train3 = train3.sample(3_000)
    df4['vices'] = np.where(df4['topic_name'].isin(['sex', 'money', 'violence']), df4['topic_name'], 
                                                    None)
    ax = sns.swarmplot(data = df4, x = 'vices', y = 'date')
    ax.set(title = 'Vice Topics Have Increased Significantly Beginning In The 90\'s')
    plt.ylabel('Decades')
    plt.xlabel('Top 3 \'Vice\' Topics')
    return


######################################## Sentiment Visuals ########################################




