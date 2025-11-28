# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:23:46 2025

@author: BHASWATITALUKDAR
"""

#Libraries
import pandas as pd
import numpy as np 
import io
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

expected_column = 'Review'  
sentiment_column = 'Sentiment' 
scores_column = 'Scores'  
document_id_column = 'Document ID'
positive_threshold = 0.6
negative_threshold = -0.3
 
#df=spark.read.format("csv").option("header","true").load("/mnt/forecastdata/Review Data/landmarkgroupsplashfashion.csv")
df = pd.read_csv(r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\scrapper_files\pantaloons_reviews.csv")

#df=df.toPandas()
# df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['at'] = pd.to_datetime(df['at'], errors='coerce')

# df=df[['review','date','rating']]
# df.rename(columns={'review': 'Review', 'date': 'Date','rating':'Rating'}, inplace=True)
df = df[['content', 'at', 'score']]
df.rename(columns={'content': 'Review', 'at': 'Date', 'score': 'Rating'}, inplace=True)

nltk.download('vader_lexicon')

sentiment_analyzer = SentimentIntensityAnalyzer()
 
df[document_id_column] = range(1, len(df) + 1)

def calculate_sentiment(review):
    if review is None:
        # Handling None values by returning a neutral compound score
        return 0.0
    scores = sentiment_analyzer.polarity_scores(review)
    compound_score = scores['compound']
    return compound_score
 
df['Scores'] = df['Review'].apply(calculate_sentiment)

def classify_sentiment_threshold(score):
    if score >= positive_threshold:
        return 'Positive'
    elif score <= negative_threshold:
        return 'Negative'
    else:
        return 'Neutral'
 
df['Sentiment'] = df['Scores'].apply(classify_sentiment_threshold)
b=df.columns.to_list()
mod = [i.rstrip() for i in b]
mod = [i.replace(' ','_') for i in mod ]
mod_dict= dict(zip(b,mod))
df.rename(columns= mod_dict,inplace=True)
df = df.dropna(subset=['Date','Review'])
df = df.drop_duplicates(subset=['Review'], keep='first')
df = df[~df['Review'].str.lower().isin(['good','Good'])]

df.to_csv('D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\scrapper_files\pantaloons_sentiment_output.csv', index=False)
