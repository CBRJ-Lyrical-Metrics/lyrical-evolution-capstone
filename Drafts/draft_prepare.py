# Unicode, Regex, json for text digestion
import unicodedata
import re
import json

# nltk: natural language toolkit -> tokenization, stopwords
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import nltk.sentiment

sia = nltk.sentiment.SentimentIntensityAnalyzer()

# Pandas dataframe manipulation
import pandas as pd

# Time formatting
from time import strftime

# Quieeet!!! Y'all can't stop me now...
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import contractions

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
    # drops nulls
    df.dropna(inplace=True)
    # add clean column that applies basic clean function
    df["clean"] = df.lyrics.apply(basic_clean).apply(remove_stopwords)
    # tokenize df applied after running tokenize function
    tokenized_df = df.clean.apply(tokenize)
    # stemmed column created from stem function
    df["stemmed"] = tokenized_df.apply(stem)
    # lemmatized column created from lemmatize function
    df["lemmatized"] = tokenized_df.apply(lemmatize)
    # create columns with character and word counts
    df = df.assign(
        character_count=df.stemmed.str.len(),
        word_count=df.stemmed.str.split().apply(len),
    )
    df["sentiment"] = df.lemmatized.apply(
        lambda msg: sia.polarity_scores(msg)["compound"]
    )
    return df


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
