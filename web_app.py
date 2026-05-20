import streamlit as st
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from langdetect import detect
import altair as alt
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Load models and vectorizer
with open('./Trained_Model/lrs_cv.pkl', 'rb') as file:
    logistic_regression_stem = pickle.load(file)
with open('./Vectorized_Matrix/tfidf_vectorizer_stem.pkl', 'rb') as vec_file:
    vectorization_stem = pickle.load(vec_file)

# Helper functions
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def classify_news(news):
    news_df = pd.DataFrame({"text": [news]})
    news_df["text"] = news_df["text"].apply(wordopt_stem)
    text_vectorized = vectorization_stem.transform(news_df["text"])
    return {
        "Logistic Regression": logistic_regression_stem.predict_proba(text_vectorized)[0][1] * 100
    }

def wordopt_stem(text):
    text = re.sub(r'http\S+|https?://\S+|www.\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt_stem)
    new_xv_test = vectorization_stem.transform(new_def_test["text"])
    pred_lr = logistic_regression_stem.predict(new_xv_test)
    return {
        "Logistic Regression": "True News" if pred_lr[0] == 1 else "Fake News",
    }

def display_stacked_bar_chart(results):
    data = pd.DataFrame({
        'Model': list(results.keys()) * 2,
        'Probability Type': ['True News'] * len(results) + ['Fake News'] * len(results),
        'Percentage': list(results.values()) + [100 - p for p in results.values()]
    })

    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Percentage:Q', stack='normalize', title='Percentage'),
        y=alt.Y('Model:N', title='Prediction Model'),
        color=alt.Color('Probability Type:N', scale=alt.Scale(domain=['True News', 'Fake News'], range=['green', 'red'])),
        tooltip=['Model:N', 'Probability Type:N', 'Percentage:Q']
    ).properties(
        title="**** Fake News Detection Probability by Model ****",
        width=600,
        height=300
    ).configure_title(
        fontSize=20,  # Adjust this value to make the title larger
        font='Arial',  # Optional: set the font type
        anchor='start'  # Optional: align the title to the start (left)
    )

    st.altair_chart(chart, use_container_width=True)

def fetch_article_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            return "\n".join([para.get_text() for para in paragraphs])
        else:
            return "Failed to retrieve content. Please try a different URL."
    except Exception as e:
        return f"Error: {e}"
    
def extract_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        links = [link.replace('/url?q=', '').split('&sa=')[0] for link in links[16:26]]
        return links
    return []

def extract_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return "\n".join([para.get_text() for para in paragraphs])
    return ""

def extract_domain(link):
    return urlparse(link).netloc

# Set the page configuration
st.set_page_config(page_title="Fake News Detection Web Application", page_icon="📰", layout="wide")

# Main title and description
st.title("Fake News Detection Web Application")
st.subheader("Analyze News for Potential Misinformation")
st.write("Enter a long passage, article URL, or keyword to detect potential fake news.\n")
st.write("")

# Tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📄 Long Passage", "🔗 URL Link", "🔍 Keyword"])

# Long Passage Detection
with tab1:
    st.header("Detect from Long Passage")
    user_input = st.text_area("Enter the news passage:", height=300)
    if st.button("Analyze Passage", key="passage"):
        if user_input:
            if len(user_input) > 300 and detect(user_input) == "en":
                st.subheader("Detection Results")

                # Model predictions and probabilities
                results_labels = manual_testing(user_input)
                results_probs = classify_news(user_input)

                for model, label in results_labels.items():
                    st.write(f"**{model} Prediction**: {label}")
                    st.write("")
                    st.write("")

                # Display stacked bar chart
                display_stacked_bar_chart(results_probs)

                # Display analyzed passage
                st.subheader("Analyzed Passage")
                with st.expander("Read the Full Analyzed Passage"):
                    st.text(user_input)
            else:
                st.error("Error: The text must be in English with more than 300 characters.") 
        else:
            st.warning("Please enter a passage to analyze.")

# URL Link Detection
with tab2:
    st.header("Detect from URL Link")
    user_url = st.text_input("Enter the article URL:")
    if st.button("Analyze URL", key="url"):
        if user_url:
            text = fetch_article_text(user_url)
            if text and len(text) > 300:
                try:
                    if detect(text) == "en":
                        st.write("**Detection Results:**")
                        results_labels = manual_testing(text)
                        results = classify_news(text)

                        for model, label in results_labels.items():
                            st.write(f"**{model} Prediction**: {label}")
                            st.write("")
                            st.write("")

                        display_stacked_bar_chart(results)
                        
                        with st.expander("Read the Full Article Text"):
                            st.text(text)  # Display the article text in an expandable section
                    else:
                        st.write("Error: This article does not appear to be in English.")
                except:
                    st.write("Error: Unable to determine the language of the article.")
            else:
                st.write("Error: The article text must be in English and contain at least 300 characters.")
        else:
            st.warning("Please enter a URL to analyze.")

# Keyword Detection
with tab3:
    st.header("Detect from Keyword")
    user_input = st.text_input("Enter a keyword or phrase:")
    if st.button("Analyze Keyword", key="keyword"):
        user_input.replace(" ", "+")
        url = f"https://www.google.com/search?q={user_input}&tbm=nws"
        extracted_texts = []

        if user_input:
            links = extract_url(url)
            count = 1

            for i, link in enumerate(links[:10]):        
                # Extract article text and display domain
                try:
                    text = extract_text(link)
                except:
                    break

                # Display detection results for each article
                if len(text) > 300 and detect(text) == "en":
                    st.subheader(f"**Article {count}**")
                    domain_name = extract_domain(link)
                    st.markdown(f"URL: [{domain_name}]({link})", unsafe_allow_html=True)
                    results = classify_news(text)
                    results_labels = manual_testing(text)
                    for model, label in results_labels.items():
                        st.write(f"Detection Results: **{model} Prediction**: {label}")
                    st.write("")
                    display_stacked_bar_chart(results)
                    
                    # Expandable section for full article
                    with st.expander("Show Full Article"):
                        st.text(text)
                else:
                    continue

                count += 1

                st.write("")
                st.write("")

                # Add a horizontal line
                st.markdown("---")
        else:
            st.warning("Please enter a keyword to analyze.")