# Sentiment Analysis on Twitter Dataset

## Overview
This project classifies tweets into positive, negative sentiments using NLP techniques.

## Dataset
The dataset used in this project is the **Sentiment140** dataset, which can be found on Kaggle:  
[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Steps
1. **Data Preprocessing:**
   - Clean tweets (remove special characters, URLs, etc.).
   - Tokenize and remove stopwords.
   - Apply lemmatization.

2. **Feature Extraction:**
   - Techniques: Bag-of-Words, TF-IDF, Word2Vec, Word embeddings.

3. **Model Training:**
   - Models: Logistic Regression, SVM, or deep learning models (e.g., LSTM, Bidirectional LSTM).  

4. **Evaluation:**
   - Metrics: Accuracy, Precision, Recall, F1-score.


## Requirements
- Python 3.x
- Libraries: Pandas, NumPy, Scikit-learn, NLTK, TensorFlow, etc.

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Hirenkhokhar/Sentiment_Analysis_on_twitter_dataset.git
   cd twitter-sentiment-analysis
   ```
2. Preprocess the dataset:
   ```bash
   python preprocess.py
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results
Example metrics: Accuracy = 92%.

