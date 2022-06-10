# Typical Everydayers...
import pandas as pd 
import numpy as np

# The Viz Folks...
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap

# Modeling help...
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# nltk: natural language toolkit -> tokenization, stopwords
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as stemmer

# Quieeet!!! Y'all can't stop me now...
import warnings
warnings.filterwarnings('ignore')

# Let me see it AAAALLLL!!!
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

# set default style for charts
plt.rc('figure', figsize=(13, 7))


######################################## Topics Visuals ########################################

# Most popular topics...
def topic_popularity(df):
    colors =(
    '#ec1c34', #(red)
    '#fc9d1c', #(orange)
    '#fbdb08', #(yellow)
    '#2dace4', #(blue)
    '#69b138', #(green)
    '#1f1e1b' #(black)
    )
    df.topic_name.value_counts().plot(kind = 'bar', color = colors)
    plt.title('Billboard Hot 100 Topic Popularity 1958-Present', fontsize = 20)
    plt.xlabel('Topic Descriptors', fontsize = 18)
    plt.xticks(rotation = 35, ha = 'right', fontsize = 14)
    plt.ylabel('Song Topic Count', fontsize = 18)
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
    df2 = df.copy()
    # add a column to the dataframe where any topic that is a relationship topic is gathered and all 
    # others are represented by 'other'
    df2['relationship_topics'] = np.where(df2.topic_name.isin(relationships), df2.topic_name, 'other')
    # drop anything that isn't a relationship topic
    df2 = df2.loc[df2['relationship_topics'] != 'other']
    df2.groupby('decade').relationship_topics.value_counts(normalize = True).unstack().plot(kind = 'bar', width = 1, ec = 'black')
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
    ax.set(title = '\'Breakup\' and \'Love\' Songs Have A Consistent Presence Over The Decades\nWhile It Appears \'Affection\' And \'Sex\' Show A Trade-off')
    plt.ylabel('Decades')
    plt.xlabel('Relationship Topics')
    return

# Billboard Colors
def relationship_line(df):
    # create a variable that stores a list relationship topics
    relationships = ['affection','breakups','love', 
                     'sex', 'heartache', 'jealousy']
    my_cmap = ListedColormap([
    '#fc9d1c', #(orange)    
    '#1f1e1b', #(black)
    '#2dace4', #(blue)
    '#fbdb08', #(yellow)        
    '#69b138', #(green)
    '#ec1c34', #(red)
    ])
    # make a copy
    df2 = df.copy()
    df2 = df2.set_index('date')
    # add a column to the dataframe where any topic that is a relationship topic is gathered and all 
    # others are represented by 'other'
    df2['relationship_topics'] = np.where(df2.topic_name.isin(relationships), df2.topic_name, 'other')
    # drop anything that isn't a relationship topic
    df2 = df2.loc[df2['relationship_topics'] != 'other']
    ax = df2.groupby('relationship_topics').resample('Y').size().unstack(0).rolling(5).mean()\
                                      .apply(lambda row: row / row.sum(),axis=1).plot(kind = 'line', linewidth = 3, cmap = my_cmap)
    # move the legend outside
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad=0., prop={'size': 15})
    plt.xlim(pd.to_datetime('1960'), pd.to_datetime('2021'))
#     plt.ylim()
    plt.title('Prevalence of Relationship Topics in Lyrics', fontsize = 20)
    plt.xlabel('Year', fontsize = 18)
    plt.xticks(rotation = 25, fontsize = 14)
    plt.ylabel('Percentage of Songs', fontsize = 18)
    plt.yticks(fontsize = 14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    plt.show()
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
    palette = [
    '#ec1c34', #(red)
    '#fc9d1c', #(orange)
#   '#2dace4', #(blue)
#   '#fbdb08', #(yellow)
#   '#69b138' #(green)
              ]
    df6 = df.copy()
    # train3 = train3.sample(3_000)
    df6['affection_v_sex'] = np.where(df6['topic_name'].isin(['affection','sex']), df6['topic_name'], 
                                                    None)
    ax = sns.swarmplot(data = df6, x = 'affection_v_sex', y = 'date', palette = palette)
    plt.title('\'Affection\' Has Been Replaced By More Explicit \'Sex\' Lyrics', fontsize = 20)
#     plt.title(fontsize = 20)
    plt.ylabel('Date', fontsize = 18)
    plt.yticks(fontsize = 14)
    plt.xlabel('Topic', fontsize = 18)
    plt.xticks(fontsize = 14)
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
    plt.xticks(rotation = 25)
    plt.ylabel('Song Topic Count')
    plt.show()
    return

def vice_swarm(df):
    palette = [
    '#ec1c34', #(red)
#   '#fc9d1c', #(orange)
    '#2dace4', #(blue)
#   '#fbdb08', #(yellow)
    '#69b138' #(green)
    ]
    df4 = df.copy()
    # train3 = train3.sample(3_000)
    df4['vices'] = np.where(df4['topic_name'].isin(['sex', 'money', 'violence']), df4['topic_name'], 
                                                    None)
    ax = sns.swarmplot(data = df4, x = 'vices', y = 'date', palette = palette)
    plt.title('Vice Topics Have Increased Significantly Beginning In The 90\'s', fontsize = 20)
    plt.ylabel('Decades',fontsize = 18)
    plt.yticks(fontsize = 14)
    plt.xlabel('Top 3 \'Vice\' Topics',fontsize = 18)
    plt.xticks(fontsize = 14)
    return


######################################## Sentiment Visuals ########################################




