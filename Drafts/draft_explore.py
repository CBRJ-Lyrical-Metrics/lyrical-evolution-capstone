# Typical Everydayers...
import pandas as pd 
import numpy as np

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


def get_topics(df):
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
    topic_label = {0:'Love', 1:'Kind Goodbye', 2:'Appeasing', 3:'Club', 4:'Country Life', 5:'Resentful Goodbye', 
                   6:'Lost', 7:'Hard Times', 8:'Nature', 9:'Miracles', 10:'Money', 11:'Dance', 12:'Fun', 
                   13:'Dance', 14:'Weekend', 15:'Transcendental', 16:'Sex', 17:'Summer', 18:'Spanish', 19:'Affection'}
    
    # Mapping the dictionary with the dataframe to get the labels.
    df['topic_name'] = df['topic'].map(topic_label)
#     # Drop the unnecessary duplicate column
#     df = pd.concat([df, df_new['topic_name']], axis = 1)
    # Drop unnecessary column 'topic'
    df = df.drop(columns = ['topic'])
    return df