import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import nltk
from bs4 import BeautifulSoup
import unicodedata
import re
import string
import spacy

from nltk.tokenize import ToktokTokenizer





os.chdir("Dataset")

df_DBLP = pd.read_csv('ACM.csv', header=0, encoding="ISO-8859-1")
df_ACM = pd.read_csv('DBLP2.csv', header=0, encoding="ISO-8859-1")
df_perfect_Match = pd.read_csv('DBLP-ACM_perfectMapping.csv', header=0, encoding="ISO-8859-1")

tokenizer = ToktokTokenizer()
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
# custom: removing words from list
stopword_list.remove('not')
nlp = spacy.load('en_core_web_sm')


# Exploring the DataSet

def evaluating_dataset():
    df_ACM.describe()
    df_ACM.info()

    df_DBLP.describe()
    df_DBLP.info()

    df_DBLP.duplicated().sum()

    df_ACM.duplicated().sum()

    df_DBLP.columns.values.tolist()

    df_ACM.columns.values.tolist()

    unique_acm_id = df_ACM['id'].unique()
    unique_acm_title = df_ACM['title'].unique()
    unique_acm_authors = df_ACM['authors'].unique()
    unique_acm_venue = df_ACM['venue'].unique()
    unique_acm_year = df_ACM['year'].unique()

    print("Unique ACM ID's", unique_acm_id.size)
    print("Unique ACM Title's", unique_acm_title.size)
    print("Unique ACM Author's", unique_acm_authors.size)
    print("Unique ACM Venue's", unique_acm_venue.size, unique_acm_venue)
    print("Unique ACM Year's", unique_acm_year.size, unique_acm_year)

    unique_dblp_id = df_DBLP['id'].unique()
    unique_dblp_title = df_DBLP['title'].unique()
    unique_dblp_authors = df_DBLP['authors'].unique()
    unique_dblp_venue = df_DBLP['venue'].unique()
    unique_dblp_year = df_DBLP['year'].unique()

    print("Unique DBLP ID's", unique_dblp_id.size)
    print("Unique DBLP ID's", unique_dblp_title.size)
    print("Unique DBLP ID's", unique_dblp_authors.size)
    print("Unique DBLP ID's", unique_dblp_venue.size, unique_dblp_venue)
    print("Unique DBLP ID's", unique_dblp_year.size, unique_dblp_year)

    sns.countplot(df_ACM['venue'])
    sns.countplot(df_ACM['year'])
    sns.countplot(df_DBLP['venue'])
    sns.countplot(df_DBLP['year'])

    # Filter data

    df_ACM[df_ACM['year'] == 2002].head()

    # Boxplot

    df_ACM[['year']].boxplot()

    # Boxplot

    df_DBLP[['year']].boxplot()

    # Correlation ACM and DBLP

    df_ACM.corr()

    sns.pairplot(df_ACM)
    plt.show(sns)

    sns.pairplot(df_DBLP)


def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()


def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text


# function to remove special characters
def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', text)


def remove_numbers(text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
    return re.sub(pattern, '', text)


def remove_punctuation(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    return text


# function for stemming
def get_stem(text):
    stemmer = nltk.porter.PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


def get_lem(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# function to remove stopwords
def remove_stopwords(text):
    # convert sentence into token of words
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    # check in lowercase
    t = [token for token in tokens if token.lower() not in stopword_list]
    text = ' '.join(t)
    return text


# function to remove special characters
def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()


def preprocessing():
    # make lower case ACM

    df_ACM['title'] = df_ACM['title'].str.lower()
    df_ACM['authors'] = df_ACM['authors'].str.lower()
    df_ACM['venue'] = df_ACM['venue'].str.lower()

    # make lower case DBLP

    df_DBLP['title'] = df_DBLP['title'].str.lower()
    df_DBLP['authors'] = df_DBLP['authors'].str.lower()
    df_DBLP['venue'] = df_DBLP['venue'].str.lower()
    df_DBLP['id'] = df_DBLP['id'].str.lower()

    # Replace null values
    df_ACM.replace(np.nan, ' ', inplace=True)
    df_ACM.isnull().sum()

    for dlb in df_DBLP.index:
        df_DBLP['title'][dlb] = remove_html_tags(df_DBLP['title'][dlb])
        df_DBLP['title'][dlb] = remove_accented_chars(df_DBLP['title'][dlb])
        df_DBLP['title'][dlb] = remove_special_characters(df_DBLP['title'][dlb])
        df_DBLP['title'][dlb] = remove_numbers(df_DBLP['title'][dlb])
        df_DBLP['title'][dlb] = remove_punctuation(df_DBLP['title'][dlb])
        df_DBLP['title'][dlb] = get_stem(df_DBLP['title'][dlb])
        df_DBLP['title'][dlb] = remove_stopwords(df_DBLP['title'][dlb])
        df_DBLP['title'][dlb] = remove_extra_whitespace_tabs(df_DBLP['title'][dlb])

        df_DBLP['authors'][dlb] = remove_html_tags(df_DBLP['authors'][dlb])
        df_DBLP['authors'][dlb] = remove_accented_chars(df_DBLP['authors'][dlb])
        df_DBLP['authors'][dlb] = remove_special_characters(df_DBLP['authors'][dlb])
        df_DBLP['authors'][dlb] = remove_numbers(df_DBLP['authors'][dlb])
        df_DBLP['authors'][dlb] = remove_punctuation(df_DBLP['authors'][dlb])
        df_DBLP['authors'][dlb] = get_stem(df_DBLP['authors'][dlb])
        df_DBLP['authors'][dlb] = remove_stopwords(df_DBLP['authors'][dlb])
        df_DBLP['authors'][dlb] = remove_extra_whitespace_tabs(df_DBLP['authors'][dlb])

        df_DBLP['venue'][dlb] = remove_html_tags(df_DBLP['venue'][dlb])
        df_DBLP['venue'][dlb] = remove_accented_chars(df_DBLP['venue'][dlb])
        df_DBLP['venue'][dlb] = remove_special_characters(df_DBLP['venue'][dlb])
        df_DBLP['venue'][dlb] = remove_numbers(df_DBLP['venue'][dlb])
        df_DBLP['venue'][dlb] = remove_punctuation(df_DBLP['venue'][dlb])
        df_DBLP['venue'][dlb] = get_stem(df_DBLP['venue'][dlb])
        df_DBLP['venue'][dlb] = remove_stopwords(df_DBLP['venue'][dlb])
        df_DBLP['venue'][dlb] = remove_extra_whitespace_tabs(df_DBLP['venue'][dlb])

    for ind in df_ACM.index:
        df_ACM['title'][ind] = remove_html_tags(df_ACM['title'][ind])
        df_ACM['title'][ind] = remove_accented_chars(df_ACM['title'][ind])
        df_ACM['title'][ind] = remove_special_characters(df_ACM['title'][ind])
        df_ACM['title'][ind] = remove_numbers(df_ACM['title'][ind])
        df_ACM['title'][ind] = remove_punctuation(df_ACM['title'][ind])
        df_ACM['title'][ind] = get_stem(df_ACM['title'][ind])
        df_ACM['title'][ind] = remove_stopwords(df_ACM['title'][ind])
        df_ACM['title'][ind] = remove_extra_whitespace_tabs(df_ACM['title'][ind])

        df_ACM['authors'][ind] = remove_html_tags(df_ACM['authors'][ind])
        df_ACM['authors'][ind] = remove_accented_chars(df_ACM['authors'][ind])
        df_ACM['authors'][ind] = remove_special_characters(df_ACM['authors'][ind])
        df_ACM['authors'][ind] = remove_numbers(df_ACM['authors'][ind])
        df_ACM['authors'][ind] = remove_punctuation(df_ACM['authors'][ind])
        df_ACM['authors'][ind] = get_stem(df_ACM['authors'][ind])
        df_ACM['authors'][ind] = remove_stopwords(df_ACM['authors'][ind])
        df_ACM['authors'][ind] = remove_extra_whitespace_tabs(df_ACM['authors'][ind])

        df_ACM['venue'][ind] = remove_html_tags(df_ACM['venue'][ind])
        df_ACM['venue'][ind] = remove_accented_chars(df_ACM['venue'][ind])
        df_ACM['venue'][ind] = remove_special_characters(df_ACM['venue'][ind])
        df_ACM['venue'][ind] = remove_numbers(df_ACM['venue'][ind])
        df_ACM['venue'][ind] = remove_punctuation(df_ACM['venue'][ind])
        df_ACM['venue'][ind] = get_stem(df_ACM['venue'][ind])
        df_ACM['venue'][ind] = remove_stopwords(df_ACM['venue'][ind])
        df_ACM['venue'][ind] = remove_extra_whitespace_tabs(df_ACM['venue'][ind])


evaluating_dataset()