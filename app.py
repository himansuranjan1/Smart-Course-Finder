import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import requests
from bs4 import BeautifulSoup

# Scrape the course data from Analytics Vidhya as done previously
url = "https://courses.analyticsvidhya.com/pages/all-free-courses"
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    base_url = "https://courses.analyticsvidhya.com"
    courses = []

    for course in soup.select('.course-card'):
        title_element = course.select_one('.course-card__body h3')
        title = title_element.get_text(strip=True) if title_element else 'No title found'
        lessons_element = course.select_one('.course-card__lesson-count strong')
        lessons = int(re.search(r'\d+', lessons_element.get_text()).group()) if lessons_element else 0
        rating_element = course.select_one('.review__stars-count')
        rating = int(re.search(r'\d+', rating_element.get_text()).group()) if rating_element else 0
        link_element = course.get('href')
        link = base_url + link_element if link_element else 'No link found'

        courses.append({
            'title': title,
            'link': link,
            'lessons': lessons,
            'rating': rating
        })

    df_courses = pd.DataFrame(courses)
    df_courses['combined_text'] = df_courses['title'] + " | Lessons: " + df_courses['lessons'].astype(str) + " | Rating: " + df_courses['rating'].astype(str)

# Load the model and embed course descriptions
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(df_courses['combined_text'].tolist(), normalize_embeddings=True)

# Initialize FAISS index with cosine similarity
embedding_dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(corpus_embeddings.astype(np.float32))

# Streamlit interface
st.title('Smart Course Search Tool')
st.write("Search for Analytics Vidhya's free courses by title, description, rating, or lessons.")

query = st.text_input('Enter search query:', '')
filter_option = st.selectbox('Secondary Sort by:', ['Rating', 'Lessons'])

# Enhanced search function
def search_courses(query, filter_option, top_k=10):
    # Perform FAISS similarity search
    query_embedding = model.encode(query, normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    # Retrieve top results based on similarity
    relevant_courses = df_courses.iloc[indices[0]].copy()

    # Filter further by checking if query is directly in the title for strict relevance
    strict_relevance = relevant_courses[relevant_courses['title'].str.contains(query, case=False, na=False)]
    if not strict_relevance.empty:
        relevant_courses = strict_relevance

    # Apply secondary sorting based on user's selection
    if filter_option == "Rating":
        relevant_courses = relevant_courses.sort_values(by='rating', ascending=False)
    elif filter_option == "Lessons":
        relevant_courses = relevant_courses.sort_values(by='lessons', ascending=False)

    return relevant_courses[['title', 'link', 'lessons', 'rating']]

# Display results
# okay
if query:
    results = search_courses(query, filter_option)
    st.write(f"Showing top results based on your query: '{query}'")
    for idx, row in results.iterrows():
        st.subheader(row['title'])
        st.write(f"**Rating**: {row['rating']} | **Lessons**: {row['lessons']}")
        st.write(f"[View Course]({row['link']})")
else:
    st.write("Please enter a search query to get started.")
