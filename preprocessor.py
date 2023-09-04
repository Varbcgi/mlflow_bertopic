from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd

def clean_data(df):
    # Counts of 'Comment' column
    counts = df['Comment'].value_counts().rename_axis('comments').reset_index(name='count')

    # Filter out rows with '-' in 'Comment' column
    df = df[df['Comment'] != '-']

    # Clean 'Comment' column
    clean_comments = df['Comment'].replace(r'[^a-zA-Z0-9äöüÄÖÜß]+$', ' ', regex=True)
    clean_comments = clean_comments.replace(r'CGI', '', regex=True)

    # Replace empty strings and 'n.a ' with NaN
    cleaner_comments = clean_comments.replace(['', 'n.a '], np.nan)

    # Get counts of cleaned 'Comment' column
    clean_comments_counts = cleaner_comments.value_counts().rename_axis('comments').reset_index(name='count').head(20)

    # Assign cleaned comments back to 'Comment' column
    df['Comment'] = cleaner_comments

    # Get counts of cleaned 'Comment' column
    cleaned_counts = df['Comment'].value_counts().rename_axis('comments').reset_index(name='count')

    # Drop rows with missing values
    clean_df = df.dropna(axis=0)

    return clean_df


def remove_stopwords(sentence):
    stop_en = set(stopwords.words('english'))
    stop_de = set(stopwords.words('german'))
    stopwords_combined = stop_en.union(stop_de)

    clean = []
    for word in sentence.split():
        if word not in stopwords_combined:
            clean.append(word)
    return ' '.join(clean)

def remove_stopwords_from_comments(df):
    df['Comments_no_stop'] = df['Comment'].apply(remove_stopwords)
    return df