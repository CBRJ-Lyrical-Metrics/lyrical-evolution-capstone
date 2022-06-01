# Unicode, Regex, json for text digestion
import unicodedata
import re
import json

# nltk: natural language toolkit -> tokenization, stopwords
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import nltk.sentiment

# Sentiment Analysis
sia = nltk.sentiment.SentimentIntensityAnalyzer()

# Modeling help...
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Pandas dataframe manipulation
import pandas as pd

# Time formatting
from time import strftime

# Quieeet!!! Y'all can't stop me now...
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import contractions
from sklearn.model_selection import train_test_split

################### BASIC CLEAN ###################


def basic_clean(string):
    """
    This function takes in a string and
    returns the string normalized.
    """
    string = (
        unicodedata.normalize("NFKD", string)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    # string = re.sub(r'[^\w\s]', '', string).lower() changed to below by chris
    string = re.sub(r"[^a-zA-Z\s]", "", string).lower()
    return string


################### TOKENIZE ###################


def tokenize(string):
    """
    This function takes in a string and
    returns a tokenized string.
    """
    # Create tokenizer.
    tokenizer = nltk.tokenize.ToktokTokenizer()

    # Use tokenizer
    string = tokenizer.tokenize(string, return_str=True)
    return string


################### FUNCTIONS ###################


def remove_stopwords(string, extra_words=[], exclude_words=[]):
    """
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    """
    # Create stopword_list.
    stopword_list = stopwords.words("english")

    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)

    # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))

    # Split words in string.
    words = string.split()

    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]

    # Join words in the list back into strings and assign to a variable.
    string_without_stopwords = " ".join(filtered_words)
    return string_without_stopwords


################### STEM ###################


def stem(string):
    """
    This function takes in a string and
    returns a string with words stemmed.a
    """
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()

    # Use the stemmer to stem each word in the list of words we created by using split.
    stems = [ps.stem(word) for word in string.split()]

    # Join our lists of words into a string again and assign to a variable.
    string = " ".join(stems)
    return string


################### LEMMATIZE ###################


def lemmatize(string):
    """
    This function takes in string for and
    returns a string with words lemmatized.
    """
    # Create the lemmatizer.
    wnl = nltk.stem.WordNetLemmatizer()

    # Use the lemmatizer on each word in the list of words we created by using split.
    lemmas = [wnl.lemmatize(word) for word in string.split()]

    # Join our list of words into a string again and assign to a variable.
    string = " ".join(lemmas)
    return string


################### CLEAN DATAFRAME ###################

def clean_df(df, extra_words=[], exclude_words=[]):
    """
    remove nulls,
    date to datetime,
    make decade column,
    remove incomplete decades, 
    remove "Lyrics" intro ('{song name} lyrics'),
    remove 'Embed' from tail of lyrics,
    remove tags [{chorus, etc}] and ({hook, etc}),
    expand contractions,
    run basic clean, all lower, letters only, 
    remove stopwords,
    lemma
    """
    # drop nulls
    df.dropna(inplace=True)
    # convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    # create decade column
    df["decade"] = df.date.dt.year - (df.date.dt.year % 10)
    # remove "Lyrics" intro ('{song name} lyrics')
    df["lyrics"] = df.lyrics.apply(lambda x: x.split("Lyrics")[1])
    # remove 'Embed' from tail of lyrics
    df["lyrics"] = df.lyrics.apply(lambda x: x.rsplit("Embed")[0])
    # remove everything contained in []
    df.lyrics = df.lyrics.apply(lambda x: re.sub(r"\[.*?\]", "", x))
    # remove everything contained in ()
    df.lyrics = df.lyrics.apply(lambda x: re.sub(r"\(.*?\)", "", x))
    # expand contractions
    df["lyrics"] = df.lyrics.apply(contractions.fix)
    # clean df
    df["lyrics"] = df.lyrics.apply(basic_clean)
    # remove stopwords
    df["lyrics"] = df.lyrics.apply(remove_stopwords)
    # lemmatize
    df["lyrics"] = df.lyrics.apply(lemmatize)
    # create columns with character and word counts
    df = df.assign(
        character_count=df.lyrics.str.len(),
        word_count=df.lyrics.str.split().apply(len),
    )
    df["sentiment"] = df.lyrics.apply(
        lambda msg: sia.polarity_scores(msg)["compound"]
    )
    return df


############################### Adding Topics ###############################
def get_topics(df):
    np.random.seed(42)  
    # Create an instance
    cv = CountVectorizer(max_df = .95, min_df = 2, stop_words = 'english')
    
    # Fit and transform the lemmatized lyrics data
    cv_fit = cv.fit_transform(df.lyrics)

    # Create the instance for LDA
    lda = LatentDirichletAllocation(n_components = 20, random_state = 42)
    
    # Fit the vectorizer with the LDA
    lda.fit(cv_fit)
    
    # Pull feature names out and define as feature
    feature = cv.get_feature_names()
    
    # Final df transforming cv_fit
    df_final = lda.transform(cv_fit)
    
#     # Make copy to save original df 
#     df_new = copy.deepcopy(df)
    
    prob = df_final[0][df_final[0].argmax()].round(2)
    
    # Assign the opics tp the dataframe
    df['topic'] = df_final.argmax(axis = 1)
    
    # Creating a dictionary with key as topic numbers and value as topic names
    topic_label = {0:'jealousy', 1:'affection', 2:'breakups', 3:'dance', 4:'holiday', 5:'nature', 
                   6:'spanish', 7:'transcendental', 8:'lost', 9:'violence', 10:'youth', 11:'love', 12:'heartache', 
                   13:'money', 14:'affection', 15:'sex', 16:'dance', 17:'good vibes', 18:'americana', 19:'breakup'}
    
    # Mapping the dictionary with the dataframe to get the labels column.
    df['topic_name'] = df['topic'].map(topic_label)
    # Drop unnecessary column 'topic'
    df = df.drop(columns = ['topic']) 
    return df

###############################  ###############################

def model_clean(df):
    """
    remove nulls,
    date to datetime,
    make decade column,
    remove incomplete decades, 
    remove "Lyrics" intro ('{song name} lyrics'),
    remove 'Embed' from tail of lyrics,
    remove tags [{chorus, etc}] and ({hook, etc}),
    expand contractions,
    run basic clean, all lower, letters only, 
    remove stopwords,
    lemma
    """
    # drop nulls
    df.dropna(inplace=True)
    # convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    # create decade column
    df["decade"] = df.date.dt.year - (df.date.dt.year % 10)
    # remove incomplete decades (1950, 2020)
    df = df[(df.decade != 1950) & (df.decade != 2020)]
    # remove "Lyrics" intro ('{song name} lyrics')
    df["lyrics"] = df.lyrics.apply(lambda x: x.split("Lyrics")[1])
    # remove 'Embed' from tail of lyrics
    df["lyrics"] = df.lyrics.apply(lambda x: x.rsplit("Embed")[0])
    # remove everything contained in []
    df.lyrics = df.lyrics.apply(lambda x: re.sub(r"\[.*?\]", "", x))
    # remove everything contained in ()
    df.lyrics = df.lyrics.apply(lambda x: re.sub(r"\(.*?\)", "", x))
    # expand contractions
    df["lyrics"] = df.lyrics.apply(contractions.fix)
    # clean df
    df["lyrics"] = df.lyrics.apply(basic_clean)
    # remove stopwords
    df["lyrics"] = df.lyrics.apply(remove_stopwords)
    # lemmatize
    df["lyrics"] = df.lyrics.apply(lemmatize)
    
    return df


def split_data_xy(X, y):
    """
    **Used for tfidf vectorizer model**
    This function takes in X and y variables as strings, then splits and returns the data as 
    X_train, X_validate, X_test, y_train, y_validate, and y_test sets using random state 42.
    """
    # split the data set with stratifiy if True
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.3, random_state=42
    )
    return (X_train, X_validate, X_test, y_train, y_validate, y_test)


def split_data(df):

    """
    This function takes in a dataframe, then splits and returns the data as train, validate, and test sets 
    using random state 42.
    """
    # split data into 2 groups, train_validate and test, assigning test as 20% of the dataset
    train_validate, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["decade"]
    )
    # split train_validate into 2 groups with
    train, validate = train_test_split(
        train_validate,
        test_size=0.3,
        random_state=42,
        stratify=train_validate["decade"],
    )
    return train, validate, test
