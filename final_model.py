import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import final_prepare as prepare
from wordcloud import WordCloud
from matplotlib import style
import nltk

style.use("ggplot")
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
from collections import Counter


def baseline_accuracy(df, mode):
    """
    Calculate baseline accuracy
    """
    df["mode"] = mode
    baseline_accuracy = accuracy_score(df["actual"], df["mode"])
    return baseline_accuracy


# get the data
df = prepare.get_data()

# remove incomplete decades (1950, 2020)
df = df[(df.decade != 1950) & (df.decade != 2020)]

# make vectorizer
tfidf = TfidfVectorizer()

# fit the vectorizer to the data and make df
X = tfidf.fit_transform(df["lyrics"])
y = df["decade"]

# split data into train and test
X_train, X_validate, X_test, y_train, y_validate, y_test = prepare.split_data_xy(X, y)

# this allows the baseline accuracy to be calculated
train = pd.DataFrame(dict(actual=y_train))
validate = pd.DataFrame(dict(actual=y_validate))
test = pd.DataFrame(dict(actual=y_test))


# get mode to use as baseline
mode = df.decade.mode().values[0]
# get baseline_accuracy
train_baseline = baseline_accuracy(train, mode)
validate_baseline = baseline_accuracy(validate, mode)
test_baseline = baseline_accuracy(test, mode)

###############################################################################
# The following models run on tf-idf data
###############################################################################


def run_logistic_reg_models(df):
    """
    Run logistic models on data varying solver and C value
    """
    # make vectorizer
    tfidf = TfidfVectorizer()
    # fit the vectorizer to the data and make df
    print("applying tfidf to data", end="\r")
    X = tfidf.fit_transform(df["lyrics"])
    y = df["decade"]

    # split data into train and test
    print("splitting data", end="\r")
    X_train, X_validate, X_test, y_train, y_validate, y_test = prepare.split_data_xy(
        X, y
    )
    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))
    test = pd.DataFrame(dict(actual=y_test))
    # get mode to use as baseline
    mode = df.decade.mode().values[0]
    # get baseline_accuracy
    train_baseline = baseline_accuracy(train, mode)
    validate_baseline = baseline_accuracy(validate, mode)
    test_baseline = baseline_accuracy(test, mode)
    # make a df for results
    results = pd.DataFrame()
    # make baseline model
    print("running baseline", end="\r")

    baseline_model = pd.Series(
        {
            "model_number": "baseline",
            "model_type": "baseline",
            "train_accuracy": train_baseline,
            "validate_accuracy": validate_baseline,
            "test_accuracy": test_baseline,
            "better_than_baseline": False,
            "baseline_accuracy": validate_baseline,
        }
    )
    # add baseline model to results df
    results = pd.concat([results, baseline_model], axis=0)
    # make more models varying solver
    model_number = 106  # start at 106 because other models are run first
    c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for solver in ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]:
        print(
            f'Using solver: {solver} out of "liblinear", "lbfgs", "newton-cg", "sag", "saga"',
            end="\r",
        )
        for c in c_values:
            # make the model
            lm = LogisticRegression(C=c, solver=solver).fit(X_train, y_train)
            # run model on data splits
            train["predicted"] = lm.predict(X_train)
            validate["predicted"] = lm.predict(X_validate)
            test["predicted"] = lm.predict(X_test)
            # make results series to add to results df
            stats = pd.Series(
                {
                    "model_number": model_number,
                    "model_type": "LogisticRegression",
                    "solver": solver,
                    "C": c,
                    "train_accuracy": accuracy_score(y_train, train["predicted"]),
                    "validate_accuracy": accuracy_score(
                        y_validate, validate["predicted"]
                    ),
                    "test_accuracy": accuracy_score(y_test, test["predicted"]),
                    "baseline_accuracy": validate_baseline,
                    "better_than_baseline": accuracy_score(
                        y_validate, validate["predicted"]
                    )
                    > validate_baseline,
                }
            )
            # add to results df
            results = pd.concat([results, stats], axis=1)
            model_number += 1
    print("******* Returning Results ***********", end="\r")

    return results.T.reset_index(drop=True)


def run_decision_tree_models(df):
    """
    Run models with decision tree classifier with varying max_depth
    """
    # # get raw data
    # df = pd.read_csv("songs_0526.csv", index_col=0)
    # df = prepare.model_clean(df)
    # make vectorizer
    tfidf = TfidfVectorizer()
    # fit the vectorizer to the data and make df
    X = tfidf.fit_transform(df["lyrics"])
    y = df["decade"]

    # split data into train and test
    X_train, X_validate, X_test, y_train, y_validate, y_test = prepare.split_data_xy(
        X, y
    )
    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))
    test = pd.DataFrame(dict(actual=y_test))
    # get mode to use as baseline
    mode = df.decade.mode().values[0]
    # get baseline_accuracy
    train_baseline = baseline_accuracy(train, mode)
    validate_baseline = baseline_accuracy(validate, mode)
    test_baseline = baseline_accuracy(test, mode)
    # make a df for results
    results = pd.DataFrame()
    # make baseline model
    baseline_model = pd.Series(
        {
            "model_number": "baseline",
            "model_type": "baseline",
            "train_accuracy": train_baseline,
            "validate_accuracy": validate_baseline,
            "test_accuracy": test_baseline,
            "better_than_baseline": False,
        }
    )
    # add baseline model to results df
    results = pd.concat([results, baseline_model], axis=0)
    # make more models varying solver
    model_number = 1
    max_depths = [1, 2, 3, 4, 5, 10, 100]
    for max_depth in max_depths:
        # make the model
        dtc = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(
            X_train, y_train
        )
        # run model on data splits
        train["predicted"] = dtc.predict(X_train)
        validate["predicted"] = dtc.predict(X_validate)
        test["predicted"] = dtc.predict(X_test)
        # make results series to add to results df
        stats = pd.Series(
            {
                "model_number": model_number,
                "model_type": "DecisionTreeClassifier",
                "max_depth": max_depth,
                "train_accuracy": accuracy_score(y_train, train["predicted"]),
                "validate_accuracy": accuracy_score(y_validate, validate["predicted"]),
                "test_accuracy": accuracy_score(y_test, test["predicted"]),
                "baseline_accuracy": validate_baseline,
                "better_than_baseline": accuracy_score(
                    y_validate, validate["predicted"]
                )
                > validate_baseline,
            }
        )
        # add to results df
        results = pd.concat([results, stats], axis=1)
        model_number += 1

    print("******* Returning Results ***********", end="\r")
    return results.T.reset_index(drop=True)


def run_random_forest_models(df):
    """
    Run models with decision tree classifier varying depth, min leaf size, criterion
    """
    # # get raw data
    # df = pd.read_csv("songs_0526.csv", index_col=0)
    # df = prepare.model_clean(df)
    # make vectorizer
    tfidf = TfidfVectorizer()
    # fit the vectorizer to the data and make df
    print("Fitting vectorizer to data...", end="\r")
    X = tfidf.fit_transform(df["lyrics"])
    y = df["decade"]

    # split data into train and test
    print("Splitting data into train and test...", end="\r")
    X_train, X_validate, X_test, y_train, y_validate, y_test = prepare.split_data_xy(
        X, y
    )
    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))
    test = pd.DataFrame(dict(actual=y_test))
    # get mode to use as baseline
    mode = df.decade.mode().values[0]
    # get baseline_accuracy
    train_baseline = baseline_accuracy(train, mode)
    validate_baseline = baseline_accuracy(validate, mode)
    test_baseline = baseline_accuracy(test, mode)
    # make a df for results
    results = pd.DataFrame()
    # make baseline model
    print("Making baseline model...", end="\r")
    baseline_model = pd.Series(
        {
            "model_number": "baseline",
            "model_type": "baseline",
            "train_accuracy": train_baseline,
            "validate_accuracy": validate_baseline,
            "test_accuracy": test_baseline,
            "better_than_baseline": False,
        }
    )
    # add baseline model to results df
    results = pd.concat([results, baseline_model], axis=0)
    # make more models varying solver
    model_number = 8  # start at 8 because of decision tree run first
    max_depths = [1, 2, 3, 4, 5, 10, 100]
    min_sample_leafs = [1, 2, 3, 4, 5, 10, 100]
    criterion = ["gini", "entropy"]
    for max_depth in max_depths:
        print(
            f"Making models with max_depth = {max_depth}... out of '1, 2, 3, 4, 5, 10, 100'",
            end="\r",
        )
        for min_samples_leaf in min_sample_leafs:
            for crit in criterion:
                # make the model
                rf = RandomForestClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    criterion=crit,
                ).fit(X_train, y_train)
                # run model on data splits
                train["predicted"] = rf.predict(X_train)
                validate["predicted"] = rf.predict(X_validate)
                test["predicted"] = rf.predict(X_test)
                # make results series to add to results df
                stats = pd.Series(
                    {
                        "model_number": model_number,
                        "model_type": "RandomForest",
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "criterion": crit,
                        "train_accuracy": accuracy_score(y_train, train["predicted"]),
                        "validate_accuracy": accuracy_score(
                            y_validate, validate["predicted"]
                        ),
                        "test_accuracy": accuracy_score(y_test, test["predicted"]),
                        "baseline_accuracy": validate_baseline,
                        "better_than_baseline": accuracy_score(
                            y_validate, validate["predicted"]
                        )
                        > validate_baseline,
                    }
                )
                # add to results df
                results = pd.concat([results, stats], axis=1)
                model_number += 1

    print("******* Returning Results ***********", end="\r")
    return results.T.reset_index(drop=True)


###############################################################################
#                                                                             #
#               Feature engineer to do alternative modeling                   #
# This is done in decades df where each row is a different decade         #
# Then a rows are added back to the df                                        #
#                                                                             #
###############################################################################


def get_common_unique_words(text, threshold=5):
    """
    Get common unique words in dataframe, aka words that occur in multiple readme's
    a word must appear in at least threshold readmes to be considered a common word
    """

    words = text.split()
    counter = Counter(words)
    common_unique_words = [
        word for word, count in counter.items() if count >= threshold
    ]
    new_string = " ".join(common_unique_words)
    return new_string


# split data into train validate and test
train, validate, test = prepare.split_data(df)

# combine all strings in lyrics where decade is the same
decades = train.groupby("decade")["lyrics"].apply(lambda x: " ".join(x)).reset_index()
decades.rename(columns={"lyrics": "all_words"}, inplace=True)
decades["unique_words"] = (
    train.groupby("decade")["unique_words"].apply(lambda x: " ".join(x)).values
)
decades["common_unique_words"] = decades.unique_words.apply(get_common_unique_words)
decades["n_words"] = decades["all_words"].apply(lambda x: len(x.split()))
decades["unique_word_count"] = decades["all_words"].apply(lambda x: len(set(x.split())))
decades["mean_word_count"] = (
    train.groupby("decade")["word_count"].mean().values.round(1)
)
decades["median_word_count"] = (
    train.groupby("decade")["word_count"].median().values.round(1)
)
decades["most_common_word"] = decades["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x)
)
decades["2nd_most_common_word"] = decades["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x, 2)
)
decades["3rd_most_common_word"] = decades["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x, 3)
)
decades["4th_most_common_word"] = decades["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x, 4)
)
decades["5th_most_common_word"] = decades["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x, 5)
)


def get_common_bigrams(text, threshold=5):
    """
    This function takes in a text and returns a list of the top 5 bigrams that are common to the text.
    """
    # get all bigrams in text
    bigrams = pd.Series(nltk.ngrams(text.split(), 2)).value_counts()
    # filter out bigrams that are less than threshold
    bigrams = bigrams[bigrams > threshold]
    # return all that occur more than 5 times
    return bigrams.index.values


decades["common_bigrams"] = decades.all_words.apply(lambda x: get_common_bigrams(x))


# add bigrams to decade dataframe
decades["bigrams"] = decades.all_words.apply(
    lambda x: pd.Series(nltk.ngrams(x.split(), 2)).values
)

# makes sets of all common bigrams in all decades so no bigrams are repeated
set_60s = set(decades[decades.decade == 1960].common_bigrams.values[0])
set_70s = set(decades[decades.decade == 1970].common_bigrams.values[0])
set_80s = set(decades[decades.decade == 1980].common_bigrams.values[0])
set_90s = set(decades[decades.decade == 1990].common_bigrams.values[0])
set_00s = set(decades[decades.decade == 2000].common_bigrams.values[0])
set_10s = set(decades[decades.decade == 2010].common_bigrams.values[0])
# remove words found in other decades
unique_to_60 = set_60s - set_70s - set_80s - set_90s - set_00s - set_10s
unique_to_70 = set_70s - set_60s - set_80s - set_90s - set_00s - set_10s
unique_to_80 = set_80s - set_60s - set_70s - set_90s - set_00s - set_10s
unique_to_90 = set_90s - set_60s - set_70s - set_80s - set_00s - set_10s
unique_to_00 = set_00s - set_60s - set_70s - set_80s - set_90s - set_10s
unique_to_10 = set_10s - set_60s - set_70s - set_80s - set_90s - set_00s
# make a series to add to the dataframe
unique_to_decade = [
    unique_to_60,
    unique_to_70,
    unique_to_80,
    unique_to_90,
    unique_to_00,
    unique_to_10,
]
decades["bigrams_unique_to_decade"] = list(unique_to_decade)


def bigram_count(bigrams, decade):
    """
    This function takes in a list of bigrams and returns the count of bigrams that are in the decades unique set.
    """
    if decade == "HTML":
        unique_set = unique_to_html
    elif decade == "Python":
        unique_set = unique_to_python
    elif decade == "R":
        unique_set = unique_to_r
    elif decade == "Other":
        unique_set = unique_to_other

    count = 0
    for bigram in bigrams:
        if bigram in unique_set:
            count += 1
    return count


def baseline_accuracy2(series, mode):
    """
    Calculate baseline accuracy
    """
    test = pd.DataFrame(series)
    test["mode"] = mode
    baseline_accuracy = accuracy_score(test["decade"], test["mode"])
    return baseline_accuracy


###############################################################################
#                                                                             #
#               run models on feature engineered columns                      #
#                                                                             #
###############################################################################


def run_models_on_feature_engineered_columns(df, scale=True, bigrams_only=False):
    """
    Run models on data varying solver and C value
    """
    # add bigram counts to main df
    df["bigrams"] = df.lyrics.apply(
        lambda x: pd.Series(nltk.ngrams(x.split(), 2)).values
    )
    df["python_bigrams"] = df.bigrams.apply(lambda x: bigram_count(x, "Python"))
    df["html_bigrams"] = df.bigrams.apply(lambda x: bigram_count(x, "HTML"))
    df["r_bigrams"] = df.bigrams.apply(lambda x: bigram_count(x, "R"))
    df["other_bigrams"] = df.bigrams.apply(lambda x: bigram_count(x, "Other"))

    # split data into train, validate, and test sets
    train, validate, test = prepare.split_data(df)
    y_train = train.decade
    y_validate = validate.decade
    y_test = test.decade
    # list all columns with dtype int or float
    numeric_cols = [
        col for col in train.columns if train[col].dtype in ["int64", "float64"]
    ]
    # keep only numeric columns
    X_train = train[numeric_cols]
    X_validate = validate[numeric_cols]
    X_test = test[numeric_cols]
    if bigrams_only:
        bigram_columns = [col for col in train.columns if "_bigrams" in col]
        X_train = train[bigram_columns]
        X_validate = validate[bigram_columns]
        X_test = test[bigram_columns]
    if scale:
        print("Scaling data")
        # scale data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_validate = scaler.transform(X_validate)
        X_test = scaler.transform(X_test)

    # get mode to use as baseline
    mode = df.decade.mode().values[0]
    # get baseline_accuracy
    train_baseline = baseline_accuracy2(y_train, mode)
    validate_baseline = baseline_accuracy2(y_validate, mode)
    test_baseline = baseline_accuracy2(y_test, mode)
    # make a df for results
    results = pd.DataFrame()
    # make baseline model
    baseline_model = pd.Series(
        {
            "model_number": "baseline",
            "model_type": "baseline",
            "train_accuracy": train_baseline,
            "validate_accuracy": validate_baseline,
            "test_accuracy": test_baseline,
            "better_than_baseline": False,
        }
    )
    # add baseline model to results df
    results = pd.concat([results, baseline_model], axis=0)
    # make more models varying solver
    model_number = results.shape[1]
    c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for solver in ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]:
        for c in c_values:
            # make the model
            lm = LogisticRegression(C=c, solver=solver).fit(X_train, y_train)
            # run model on data splits
            train["predicted"] = lm.predict(X_train)
            validate["predicted"] = lm.predict(X_validate)
            test["predicted"] = lm.predict(X_test)
            # make results series to add to results df
            stats = pd.Series(
                {
                    "model_number": model_number,
                    "model_type": "LogisticRegression",
                    "solver": solver,
                    "C": c,
                    "train_accuracy": accuracy_score(y_train, train["predicted"]),
                    "validate_accuracy": accuracy_score(
                        y_validate, validate["predicted"]
                    ),
                    "test_accuracy": accuracy_score(y_test, test["predicted"]),
                    "baseline_accuracy": validate_baseline,
                    "better_than_baseline": accuracy_score(
                        y_validate, validate["predicted"]
                    )
                    > validate_baseline,
                }
            )
            # add to results df
            results = pd.concat([results, stats], axis=1)
            model_number += 1

    return results.T.reset_index(drop=True)
