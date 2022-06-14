# Typical Everydayers...
import pandas as pd 
import numpy as np

# The Viz Folks...
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# Quieeet!!! Y'all can't stop me now...
import warnings
warnings.filterwarnings('ignore')

# Let me see it AAAALLLL!!!
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)


plt.rcParams['figure.figsize'] = (15, 5)


def sentiment_lineplot(df):
    mpl.style.use('seaborn')
    plt.figure(figsize=(12,9))
    df.groupby('decade').mean().sentiment.plot(marker='.',
                                               markersize=18,
                                               color='#69B138', #(green)
                                               linewidth=4
                                              )
    plt.title('Average Sentiment\nby Decade', fontsize=20)
    plt.xlabel(None)
    plt.xticks(fontsize=18)
    plt.ylabel('Sentiment Score', fontsize=18)
    plt.yticks(fontsize=14)
    plt.show()
    return

######################################## Topics Visuals ########################################

# Most popular topics...
def topic_popularity(df):
    plt.figure(figsize=(12, 9))
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
    plt.show()
    return

def show_topic_counts():
    pd.DataFrame(df.topic_name.value_counts())
    return

def all_topics_prevalence(df):
    plt.figure(figsize=(12, 9))
    ax = sns.countplot(data = df, x = 'decade', hue = 'topic_name', ec = 'black')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad=0.)
    plt.title('Topics\' Prevalence Over the Decades')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
    plt.xlabel('Decade of Song')
    plt.ylabel('Song Count')
    plt.figure(figsize=(12, 9))
    plt.show()
    return

def relationship_bar(df):
    plt.figure(figsize=(12, 9))
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

def relationship_swarm(df):
    plt.figure(figsize=(12, 9))    
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
    plt.figure(figsize=(13, 10))
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
                                      .apply(lambda row: row / row.sum(), axis=1).plot(kind = 'line', linewidth = 3, cmap = my_cmap)
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
    plt.figure(figsize=(12, 9))
    df5 = df.copy()
    # train3 = train3.sample(3_000)
    df5['love_v_sex'] = np.where(df5['topic_name'].isin(['affection','love', 'sex']), df5['topic_name'], 
                                                    None)
    ax = sns.swarmplot(data = df5, x = 'love_v_sex', y = 'date')
    ax.set(title = 'While Love Has Remained A Constant Topic, Sex Has Replaced Affection In Lyrics')
    plt.ylabel('Decades')
    plt.xlabel('Topics of Intimacy')
    plt.figure(figsize=(12, 9))
    return

def touch_swarm(df):
    plt.figure(figsize=(12, 9))
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
    plt.figure(figsize=(12, 9))
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
    plt.figure(figsize=(12, 9))
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


def sentiment_histplot(df):
    plt.figure(figsize=(12,2))
    sns.histplot(df.sentiment, 
                 bins=20, 
                 color='#ec1c34', #(red)
                )
    plt.title('Overall Sentiment Distribution', fontsize=18)
    plt.xticks(fontsize=13)
    plt.show()
    
def sentiment_stacked_bar(df):
    plt.style.use('seaborn')

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([
                            '#ec1c34', #(red)
                            '#fbdb08', #(yellow)
                            '#69b138' #(green)
                           ])

    (
        df2.groupby('decade')
         .sentiment_category_2
         .value_counts(normalize = True)
         .unstack()
         .plot(kind='bar',
               stacked=True,
               cmap=cmap,
               ec='black',
               figsize=(7, 8),
               )
    )
    plt.gca().yaxis.set_major_formatter('{:.0%}'.format)
    plt.title('Distribution of Sentiment\nAcross Decades', fontsize=16)
    plt.ylabel('Portion of All Songs', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xlabel(None)
    plt.xticks(rotation=0, fontsize=14)
    plt.legend(
                #bbox_to_anchor=(1,1), 
                fontsize=12
                )
    plt.ylim(0, 1.19)
    plt.show()
    return

def historical_lineplot(df):
    df = df.set_index('date')    
    df3 = df['sentiment'].resample('Y').mean().dropna()
    df3.index.freq = None

    mpl.style.use('seaborn')
    plt.figure(figsize=(12,9))

    df3.plot(label='Annual Average', 
             color='#2dace4', #(blue)
             linewidth='4'
            )

    plt.title('Lyric Sentiment vs. Historical Events', fontsize=20)
    plt.ylabel('Sentiment Score', fontsize=18)
    plt.yticks(fontsize=14)
    plt.xlabel(None)
    plt.xticks(fontsize=14)
    plt.xlim(pd.to_datetime('1960'), pd.to_datetime('2020'))
    plt.ylim(.11, .79)

    arrowprops = {
                  'arrowstyle': '->',
                  'linewidth': .8,
                  'facecolor': 'black',
                  #'relpos': (0,1)
                 }



    # 1964: Increased Presence in Vietnam
    plt.annotate('1964\nIncreased\nU.S. Presence\nin Vietnam', 
                 xy=(
                     pd.to_datetime('1964'), 
                     df3[df3.index == pd.to_datetime('1964-12-31')].values[0]
                     ), 
                 xytext=(
                         pd.to_datetime('1964'), 
                         df3[df3.index == pd.to_datetime('1964-12-31')].values[0] - .15
                         ),
                 ha='center',
                 arrowprops=arrowprops
                )

    # 1968: Martin Luther King Assassinated
    plt.annotate('1968\nMartin Luther King Jr.\nAssassinated', 
                 xy=(
                     pd.to_datetime('1968'), 
                     df3[df3.index == pd.to_datetime('1968-12-31')].values[0]
                     ), 
                 xytext=(
                         pd.to_datetime('1968'), 
                         df3[df3.index == pd.to_datetime('1968-12-31')].values[0] + .05
                         ),
                 ha='center',
                 arrowprops=arrowprops
                )

    # 1973: US. Withdrawal from Vietnam
    plt.annotate('1973\nU.S. Withdrawal\nfrom Vietnam', 
                 xy=(
                     pd.to_datetime('1973'), 
                     df3[df3.index == pd.to_datetime('1973-12-31')].values[0]
                     ), 
                 xytext=(
                         pd.to_datetime('1973'), 
                         df3[df3.index == pd.to_datetime('1973-12-31')].values[0] - .2
                         ),
                 ha='center',
                 arrowprops=arrowprops
                )


    # 1986 Challenger Explosion
    plt.annotate('1986\nChallenger\nExplosion', 
                 xy=(
                     pd.to_datetime('1986'), 
                     df3[df3.index == pd.to_datetime('1986-12-31')].values[0]
                     ), 
                 xytext=(
                         pd.to_datetime('1986'), 
                         df3[df3.index == pd.to_datetime('1986-12-31')].values[0] - .2
                         ),
                 ha='center',
                 arrowprops=arrowprops
                )

    # 1989 Fall of the Berlin Wall
    plt.annotate('1989\nFall of the\nBerlin Wall', 
                 xy=(
                     pd.to_datetime('1989'), 
                     df3[df3.index == pd.to_datetime('1989-12-31')].values[0]
                     ), 
                 xytext=(
                         pd.to_datetime('1989'), 
                         df3[df3.index == pd.to_datetime('1989-12-31')].values[0] - .2
                         ),
                 ha='center',
                 arrowprops=arrowprops
                )

    # 2001: September 11th Attacks
    plt.annotate('2001\nSeptember 11th\nAttacks', 
                 xy=(
                     pd.to_datetime('2001'), 
                     df3[df3.index == pd.to_datetime('2001-12-31')].values[0]
                     ), 
                 xytext=(
                         pd.to_datetime('2001'), 
                         df3[df3.index == pd.to_datetime('2001-12-31')].values[0] - .15
                         ),
                 ha='center',
                 arrowprops=arrowprops
                )

    # 2008: Global Financial Crisis
    plt.annotate('2008\nGlobal\nFinancial\nCrisis', 
                 xy=(
                     pd.to_datetime('2008'), 
                     df3[df3.index == pd.to_datetime('2008-12-31')].values[0]
                     ), 
                 xytext=(
                         pd.to_datetime('2008'), 
                         df3[df3.index == pd.to_datetime('2008-12-31')].values[0] + .1
                         ),
                 ha='center',
                 arrowprops=arrowprops
                )

    # 2011: Spotify Launch and the Rise of Streaming
    plt.annotate('2011\nSpotify Music\nStreaming Launch', 
                 xy=(
                     pd.to_datetime('2011'), 
                     df3[df3.index == pd.to_datetime('2011-12-31')].values[0]
                     ), 
                 xytext=(
                         pd.to_datetime('2011'), 
                         df3[df3.index == pd.to_datetime('2011-12-31')].values[0] - .2
                         ),
                 ha='center',
                 arrowprops=arrowprops
                )

    plt.legend(fontsize=12, loc='lower left')
    plt.show()
    return


######################################## Love/Like Visuals ########################################


def love_vs_like_lineplot(df):
    df = df.set_index('date')
    df['love_rate'] = df.lyrics.str.count('love') / (df.word_count / 100)
    df['like_rate'] = df.lyrics.str.count('like') / (df.word_count / 100)
    df2 = df[['love_rate', 'like_rate']].resample('Y').mean().dropna().rolling(5).mean()

    plt.figure(figsize=(12,9))
    sns.lineplot(data=df2, 
                 x='date', 
                 y='love_rate',
                 color='#2dace4', #(blue)
                 linewidth=4)
    sns.lineplot(data=df2, 
                 x='date', 
                 y='like_rate',
                 color='#69b138', #(green)
                 linewidth=4)
    plt.title('Love vs. Like Rate', fontsize=20)
    plt.xlabel('Year', fontsize=18)
    plt.xticks(fontsize=14)
    plt.ylabel('Average Times\nPer 100 Words', fontsize=18)
    plt.yticks(fontsize=14)
    plt.legend(['Love', 'Like'], fontsize=14, loc='center right')

    plt.annotate('(Rolling 5-Year Average)', 
                 xy=(pd.to_datetime('2009'), .7), 
                 xytext=(pd.to_datetime('2009'), .7),
                 fontsize=14
                )

    plt.show()
    return


######################################## Unique Words Visuals ########################################


def unique_words_lineplot(df):
    df = df.set_index('date')
    df2 = df[['unique_words_count', 'word_count']].resample('Y').mean().dropna().rolling(5).mean()

    plt.figure(figsize=(12,9))
    sns.lineplot(data=df2,
                 x='date',
                 y='unique_words_count',
                 color='#EC1C34', #(red)
                 linewidth=4)

    plt.title('Unique Words\nPer Song Increases Over Time', fontsize=20)
    plt.legend(['Rolling 5-year Average'], fontsize=14, loc='lower right')
    plt.xlabel('Year', fontsize=18)
    plt.xticks(fontsize=14)
    plt.ylabel('Unique Words', fontsize=18)
    plt.yticks(fontsize=14)
    plt.show()
    return





