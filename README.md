
---

# 📰 Fake News Detection Web Application
*Last Updated: 20 May 2026* 

A machine learning-powered web application built with Streamlit that analyzes news content to detect potential misinformation. By leveraging Natural Language Processing (NLP) and a trained Logistic Regression model, this tool classifies text as either "True News" or "Fake News" and provides a visual breakdown of the prediction probabilities.

<br>

## 🌟 Test Now
No setup required! You can view the live versions of the app right now:
* 🌐 **View on Web:** [Fake News Detection](https://fakenews-detection-1bmw.onrender.com/) <br>
*Note: The app is hosted on a free server, so it might take some times to load and analyse text.*

<br>

## 🚀 Key Features
The application offers three distinct ways to analyze news for misinformation (only available in English):
* **Long Passage Detection:** Paste any news article snippet or long-form text (minimum 300 characters) directly into the app for instant analysis.
* **URL Link Detection:** Paste a link to a news article. The app automatically scrapes the webpage, extracts the main paragraph text, and analyzes the content.
* **Keyword Search Detection:** Enter a topic or keyword (e.g., "election results"). The app queries Google News, extracts the top articles, and individually evaluates each source, providing a breakdown of the domains and their respective truth probabilities.

---

## 📚 Dataset & Sources
To ensure the model was trained on a highly diverse and robust vocabulary, the final dataset was constructed by merging multiple distinct open-source datasets. This combined dataset includes political news, global events, COVID-19 misinformation, and satirical articles.
The data was sourced and aggregated from the following repositories:
* [Elvin Data (Kaggle)](https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp) 
* [Jruvika Data (Kaggle)](https://www.kaggle.com/datasets/jruvika/fake-news-detection) 
* [Saurabh Data (Kaggle)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) 
* [Challenge Data (Kaggle)](https://www.kaggle.com/competitions/fake-news/data)
* [IEEE Data (IEEE Dataport)](https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset) 
* [Ibrahim Data (Kaggle)](https://www.kaggle.com/datasets/ibrahimkaratas/fakenews/data) 
* [Jair Data (Kaggle)](https://www.kaggle.com/datasets/liberoliber/onion-notonion-datasets/data) 
* [Rahul Data (Kaggle)](https://www.kaggle.com/datasets/iamrahulthorat/fakenews-csv) 

---

## 📊 Project Workflow
The development of this application followed a standard data science lifecycle, documented in the included Jupyter Notebooks:

<br>

### 1. Data Understanding (`data_understanding.ipynb`)
This phase focuses on Exploratory Data Analysis (EDA) to comprehend the dataset before training. Key steps include:
* Analyzing the class distribution between "Fake" and "True" news to check for data imbalances.
* Visualizing text characteristics (e.g., word counts, character lengths).
* Identifying missing values, duplicates, and noisy data that require cleaning.
* Generating word clouds and frequency distributions to identify common vocabulary used in both classes.

<br>

### 2. Data Modelling (`data_modelling.ipynb`)
This phase transforms the raw text into a mathematical format that machines can learn from. Key steps include:
* **Text Preprocessing:** Removing URLs, HTML tags, punctuation, and digits using Regular Expressions.
* **NLP Pipeline:** Removing English stopwords and applying Porter Stemming to reduce words to their root forms.
* **Vectorization & Training:** Converting text into numerical features and training various machine learning classification algorithms to identify the most accurate predictor. Models and algorithms trained includes of Logistic Regression, Random Forest, Decision Tree, Convolutional Neural Networks (CNNs), and Recurrent Neural Networks (RNNs). Each models is trained with both the stemmed data and lemmatized data. 

---

## 🧠 Model Selection & Performance
After training and evaluating multiple machine learning models in the Data Modelling phase, **Logistic Regression** paired with a **Stemmed TF-IDF Vectorizer** was selected as the final production model.

<br>

### Why Logistic Regression (`lrs_cv.pkl`)?
While complex models like Random Forest or Deep Learning are powerful, Logistic Regression was chosen because it excels at binary text classification (True vs. Fake). It is highly computationally efficient, prevents overfitting on high-dimensional text data, and most importantly, it outputs exact **probability scores** (e.g., 85% True / 15% Fake), which is essential for the stacked bar charts in the web app.

<br>

### Why TF-IDF with Stemming (`tfidf_vectorizer_stem.pkl`)?
Instead of a basic word-count (CountVectorizer), **TF-IDF** (Term Frequency-Inverse Document Frequency) was used. It penalizes highly frequent, uninformative words across the dataset while giving weight to unique, defining keywords. Pairing this with **Stemming** reduces the overall vocabulary size (e.g., treating "running," "runs," and "ran" as the same root word), making the model faster, leaner, and more generalized to new articles.

<br>

### 📈 Final Model Performance
The selected Logistic Regression model with data used with TF-IDF and stemmed achieved the following performance metrics on the testing dataset:
| Metric | Score | Description |
| --- | --- | --- |
| **Accuracy** | `[89.2%]` | The overall percentage of correctly classified news articles. |
| **Precision** | `[89%]` | The proportion of predicted "Fake" articles that were actually fake. |
| **Recall** | `[89%]` | The proportion of actual "Fake" articles the model successfully caught. |
| **F1-Score** | `[89%]` | The harmonic mean of Precision and Recall, proving the model is well-balanced. |

---

## 🛠️ Technology Stack
* **Frontend / Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF Vectorization)
* **Natural Language Processing (NLP):** NLTK (Tokenization, Porter Stemming, Stopwords removal)
* **Web Scraping:** Requests, BeautifulSoup4
* **Data Visualization:** Altair, Pandas
* **Data Analysis:** Jupyter Notebook, Pandas, Matplotlib/Seaborn
* **Language Detection:** `langdetect` (Ensures inputs are evaluated in English)

---

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

---

## 📂 Project Structure
```text
├── data_understanding.ipynb        # Exploratory Data Analysis (EDA) notebook
├── data_modelling.ipynb            # Text preprocessing and model training notebook
├── Trained_Model/
│   └── lrl.pkl                     # Exported Logistic Regression Model with lemmatisation data
│   └── lrl_cv.pkl                  # Exported Logistic Regression Model with lemmatisation data and hyperparameter tuning
│   └── lrs.pkl                     # Exported Logistic Regression Model with stemming data
│   └── lrs_cv.pkl                  # Exported Logistic Regression Model with stemming data and hyperparameter tuning
│   └── dtl.pkl                     # Exported Decision Tree Model with lemmatisation data
│   └── dtl_cv.pkl                  # Exported Decision Tree Model with lemmatisation data and hyperparameter tuning
│   └── dts.pkl                     # Exported Decision Tree Model with stemming data
│   └── dts_cv.pkl                  # Exported Decision Tree Model with stemming data and hyperparameter tuning
│   └── rfl.pkl                     # Exported Random Forest Model with lemmatisation data
│   └── rfl_cv.pkl                  # Exported Random Forest Model with lemmatisation data and hyperparameter tuning
│   └── rfs.pkl                     # Exported Random Forest Model with stemming data
│   └── rfs_cv.pkl                  # Exported Random Forest Model with stemming data and hyperparameter tuning
│   └── rnn_l.h5                    # Exported RNN Model with lemmatisation data
│   └── rnn_s.h5                    # Exported RNN Model with stemming data
│   └── rnnl_tuned.h5               # Exported RNN Model with lemmatisation data and hyperparameter tuning 
│   └── rnns_tuned.h5               # Exported RNN Model with stemming data and hyperparameter tuning
│   └── cnnl.h5                     # Exported CNN Model with lemmatisation data
│   └── cnns.h5                     # Exported CNN Model with stemming data
│   └── cnnl_tuned.h5               # Exported CNN Model with lemmatisation data and hyperparameter tuning 
│   └── cnns_tuned.h5               # Exported CNN Model with stemming data and hyperparameter tuning
├── Vectorized_Matrix/
│   └── tfidf_vectorizer_stem.pkl   # Exported TF-IDF Vectorizer with stemming
│   └── tfidf_vectorizer_lemma.pkl  # Exported TF-IDF Vectorizer with lemmatisation
├── web_app.py                      # Main Streamlit application script
└── requirements.txt                # Python dependencies
```

---

*Note: This project was developed as part of a Final Year Project (FYP).*
