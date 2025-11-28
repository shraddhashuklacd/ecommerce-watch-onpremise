# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:01:34 2025
 
@author: BHASWATITALUKDAR
"""
 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk
import openai
 
# Download NLTK stopwords if not already present
nltk.download('stopwords')
 
# Load the input data
file_path = r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\scrapper_files\keywords_output.csv"  # input file path
data = pd.read_csv(file_path)
 
# Create a keywords corpus grouped by docID
data['keywords corpus'] = data.groupby('docID')['keyword'].transform(lambda x: ' '.join(x))
corpus_df = data[['docID', 'review', 'keywords corpus']].drop_duplicates()
 
# Preprocess the keywords corpus
stop_words = set(stopwords.words('english'))
corpus_df['keywords corpus'] = corpus_df['keywords corpus'].apply(
    lambda text: ' '.join([word for word in text.split() if word.lower() not in stop_words])
)
 
# Vectorize the text using CountVectorizer
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(corpus_df['keywords corpus'])
 
# Perform LDA for topic modeling
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 topics
lda_model.fit(doc_term_matrix)
 
# Map topics to keywords
topic_keywords = {}
for idx, topic in enumerate(lda_model.components_):
    topic_keywords[idx] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]  # Top 10 words per topic
 
# Assign topics to documents
def get_topic_name(text):
    doc_vector = vectorizer.transform([text])
    topic_distribution = lda_model.transform(doc_vector)
    topic_idx = topic_distribution.argmax()
    return f"Topic {topic_idx}: {', '.join(topic_keywords[topic_idx])}"
 
corpus_df['topic'] = corpus_df['keywords corpus'].apply(get_topic_name)
 
# Save the final output to a CSV file before AI-based topic naming
output_file = r"world1_topic_modeling_output.csv"
corpus_df.to_csv(output_file, index=False)
print(f"Intermediate output CSV saved at: {output_file}")
 
# --------------------------------------------------------------
# AI-based Topic Naming
# --------------------------------------------------------------
 
# OpenAI API key
# OPENAI_API_KEY = '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'
apiKey = "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"  # Replace with your OpenAI API key
client = openai.OpenAI(api_key=apiKey) 

openai.debug = True  # Enable debugging for OpenAI API
 
# Function to assign topic names using AI 
def assign_topic_names_ai(topic_keywords):
    topic_names = {}
    for topic_id, keywords in topic_keywords.items():
        prompt = (
f"The following keywords represent a topic: {', '.join(keywords)}.\n"
            "Generate a short, meaningful topic name (3–4 words max) that summarizes these keywords.\n"
            "Use the same concise style as the following examples:\n"
            "- Support/User Feedback\n"
            "- Login/Access Issue\n"
            "- Account Management\n"
            "- Delivery Experience\n\n"
            "Rules:\n"
            "1. Do not include quotation marks or punctuation around the output.\n"
            "2. Avoid generic words like 'topic', 'about', or 'related to'.\n"
            "3. Output only the topic name — nothing else."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that assigns topic names."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=20,
            temperature=0.7,
        )
        topic_name = response.choices[0].message.content
        topic_names[topic_id] = topic_name
    return topic_names
 
# Generate AI-based topic names

ai_topic_names = assign_topic_names_ai(topic_keywords)
 
# Map AI topic names to the dataframe
def map_ai_topic_name(text):
    doc_vector = vectorizer.transform([text])
    topic_distribution = lda_model.transform(doc_vector)
    topic_idx = topic_distribution.argmax()
    return ai_topic_names[topic_idx]
 
# Add AI-based topic names to the dataframe
corpus_df['AI Topic Name'] = corpus_df['keywords corpus'].apply(map_ai_topic_name)
 
# Save the final output with AI-based topic names
final_output_file = r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\scrapper_files\topic_output.csv"
corpus_df.to_csv(final_output_file, index=False)
 

print(f"Final output CSV with AI topic names saved at: {final_output_file}")
