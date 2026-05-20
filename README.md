
---

# 📰 Fake News Detection Web Application
*Last Updated: 20 May 2026*

A machine learning-powered web application built with Streamlit that analyzes news content to detect potential misinformation. By leveraging Natural Language Processing (NLP) and a trained Logistic Regression model, this tool classifies text as either "True News" or "Fake News" and provides a visual breakdown of the prediction probabilities.

.
## 🚀 Key Features
The application offers three distinct ways to analyze news for misinformation (only available for English):
* **Long Passage Detection:** Paste any news article snippet or long-form text (minimum 300 characters) directly into the app for instant analysis.
* **URL Link Detection:** Paste a link to a news article. The app automatically scrapes the webpage, extracts the main paragraph text, and analyzes the content.
* **Keyword Search Detection:** Enter a topic or keyword (e.g., "election results"). The app queries Google News, extracts the top articles, and individually evaluates each source, providing a breakdown of the domains and their respective truth probabilities.

.
## 🛠️ Technology Stack
* **Frontend / Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF Vectorization)
* **Natural Language Processing (NLP):** NLTK (Tokenization, Porter Stemming, Stopwords removal)
* **Web Scraping:** Requests, BeautifulSoup4
* **Data Visualization:** Altair, Pandas
* **Language Detection:** `langdetect` (Ensures inputs are evaluated in English)

.
## ⚙️ How It Works
1. **Text Preprocessing:** User input (or scraped text) is cleaned using regular expressions to remove URLs, HTML tags, punctuation, numbers, and newline characters.
2. **Tokenization & Stemming:** The text is tokenized into individual words, stripped of English stop words, and reduced to root forms using the `PorterStemmer`.
3. **Vectorization:** The cleaned text is transformed into a mathematical matrix using a pre-trained TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
4. **Prediction:** A pre-trained Logistic Regression model evaluates the matrix to predict the classification and calculate the exact probability of the text being real or fake.
5. **Visualization:** Results are displayed via a normalized stacked bar chart to easily interpret the model's confidence.

.
## 💻 Local Installation & Setup
**1. Clone the repository:**
```bash
git clone https://github.com/JaydenChuaComputer/fakenews-detection.git
cd your-repo-name
```
**2. Create a virtual environment and activate it (Optional but recommended):**
```bash
python -m venv venv
./venv/Scripts/activate  
```
**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```
**4. Run the Streamlit app:**
```bash
streamlit run web_app.py
```

.
## 📂 Project Structure
```text
├── Trained_Model/
│   └── lrs_cv.pkl                  # Pre-trained Logistic Regression Model
├── Vectorized_Matrix/
│   └── tfidf_vectorizer_stem.pkl   # Pre-trained TF-IDF Vectorizer
├── web_app.py                      # Main Streamlit application script
└── requirements.txt                # Python dependencies
```

---

*Note: This project was developed as part of a Final Year Project (FYP).*
