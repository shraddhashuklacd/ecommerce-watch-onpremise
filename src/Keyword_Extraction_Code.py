# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:21:24 2024

@author: BHASWATITALUKDAR
"""

import time
import pandas as pd
import spacy
from rake_nltk import Rake
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Start timing
start_time = time.time()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize RAKE and KeyBERT
rake = Rake()
kw_model = KeyBERT()

# Input and output file paths
input_file = r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\scrapper_files\pantaloons_reviews.csv"# Input file path
output_file = r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\scrapper_files\keywords_output.csv"  # Output file path

# Read the input CSV
data = pd.read_csv(input_file, encoding='ISO-8859-1', on_bad_lines='skip')  # Safe mode
data = data.fillna("")
data = data.astype(str)

# Ensure 'content' column exists
# if 'review' not in data.columns:
#     raise ValueError("The input file does not have a 'review' column.")
# Ensure 'content' column exists and rename it to 'review'
if 'content' not in data.columns:
    raise ValueError("The input file does not have a 'content' column.")
data = data.rename(columns={'content': 'review'})


# Assign unique docIDs starting from 1
data['docID'] = range(1, len(data) + 1)

# Prepare output structure
output_rows = []

# Define a function to calculate relevance/coherence score
def calculate_relevance(text, keywords):
    # Ensure text and keywords are non-empty
    if not text.strip() or not keywords:
        return {}

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit-transform the text and keywords
    try:
        vectors = vectorizer.fit_transform([text] + keywords)
        text_vector = vectors[0]  # The vector for the text
        keyword_vectors = vectors[1:]  # The vectors for the keywords
        # Calculate cosine similarity
        scores = cosine_similarity(text_vector, keyword_vectors).flatten()
        return dict(zip(keywords, scores))
    except ValueError as e:
        # Handle empty or invalid vectors
        print(f"ValueError in calculate_relevance: {e}")
        return {}

# Define a function to clean and filter keywords
def clean_keywords(keywords):
    filtered_keywords = []
    seen = set()
    for kw in keywords:
        kw = kw.strip().lower()
        kw = re.sub(r'[^a-zA-Z\s]', '', kw)  # Remove non-alphanumeric characters
        if kw not in seen and len(kw.split()) > 1:  # Remove duplicates and single words
            seen.add(kw)
            filtered_keywords.append(kw)
    return filtered_keywords

# Define a function to detect redundancy using POS and regex
def remove_redundancy(keywords):
    final_keywords = []
    for keyword in keywords:
        doc = nlp(keyword)
        # Include only noun phrases (e.g., NN, NNP)
        if any(token.pos_ in ['NOUN', 'PROPN'] for token in doc):
            final_keywords.append(keyword)
    return list(set(final_keywords))

# Utility function to flatten nested lists
def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursively flatten
        else:
            flat_list.append(item)
    return flat_list

# Define the improved RAKE extraction function
def extract_keyphrases_rake(text, num_phrases=5, min_length=2, max_length=5):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    key_phrases = rake.get_ranked_phrases_with_scores()  # Get all ranked phrases

    filtered_phrases = []
    for kp in key_phrases:
        Key_Phrase = kp[1]
        if isinstance(Key_Phrase, str) and min_length <= len(Key_Phrase.split()) <= max_length:
            filtered_phrases.append(Key_Phrase)
    return filtered_phrases  # Return a list of strings

# Define a function to extract keywords using spaCy
def extract_keywords_spacy(text):
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    cleaned_keywords = [clean_keywords(kp) for kp in keywords]
    return cleaned_keywords

# Define a function to extract keywords using KeyBERT
def extract_keywords_keybert(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 4), stop_words="english", top_n=5)
    return [clean_keywords(kw[0]) for kw in keywords]

# Process each review content
for _, row in data.iterrows():
    doc_id = row['docID']
    review = row['review']

    if pd.isnull(review):  # Skip if the review content is null
        continue

    # Extract keywords using spaCy
    spacy_keywords = extract_keywords_spacy(review)

    # Extract keywords using RAKE
    rake_keywords = extract_keyphrases_rake(review)

    # Extract keywords using KeyBERT
    keybert_keywords = extract_keywords_keybert(review)
    
    # Flatten and clean extracted keywords
    spacy_keywords = flatten_list(spacy_keywords)
    rake_keywords = flatten_list(rake_keywords)
    keybert_keywords = flatten_list(keybert_keywords)
    
    # Ensure all elements are strings
    spacy_keywords = [str(kw) for kw in spacy_keywords if kw]
    rake_keywords = [str(kw) for kw in rake_keywords if kw]
    keybert_keywords = [str(kw) for kw in keybert_keywords if kw]

    # Combine all keywords into a set
    all_keywords = set(spacy_keywords + rake_keywords + keybert_keywords)
    
    # Calculate relevance scores for cleaned keywords
    cleaned_all_keywords = clean_keywords(list(all_keywords))  # Clean all keywords
    relevance_scores = calculate_relevance(review, cleaned_all_keywords)
    if not relevance_scores:
        print(f"Relevance scores could not be calculated for docID: {doc_id}")
        continue  # Skip to the next review if scores can't be calculated

    # Weight and filter keywords
    weighted_keywords = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

    # Remove redundancy from cleaned keywords
    final_keywords = remove_redundancy(cleaned_all_keywords)

    # Add each keyword with score as a new row in the output
    for keyword in final_keywords:
        try:
            score = relevance_scores.get(keyword, 0)
            output_rows.append({
                'docID': doc_id,
                'review': review,
                'keyword': keyword,
                'score': score
            })
        except KeyError as e:
            print(f"KeyError for keyword: {e}")


# Create output DataFrame
output_df = pd.DataFrame(output_rows, columns=['docID', 'review', 'keyword', 'method'])

# Save to CSV
output_df.to_csv(output_file, index=False)

#print(f"Keyword extraction completed. Output saved to {output_file}.")

# End timing
end_time = time.time()
execution_time = end_time - start_time

print(f"Keyword extraction completed. Output saved to {output_file}.")
print(f"Total computation time: {execution_time:.2f} seconds.")