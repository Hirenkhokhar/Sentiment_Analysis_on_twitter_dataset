Sentiment Analysis on Twitter Dataset

Abstract
This project leverages natural language processing (NLP) and machine learning techniques to perform sentiment analysis on a Twitter dataset comprising 1,600,000 annotated tweets. Through comprehensive data preprocessing steps such as stop words removal, negation handling, lower casing, and replacing URLs, usernames, emojis, and contractions, the data was prepared for analysis. Various machine learning models, including Logistic Regression, Naive Bayes, Decision Tree Classifier, SVC, Random Forest, Gradient Boost, and XGBoost, were trained, with Logistic Regression achieving the highest accuracy of 82%. Additionally, deep learning models like SimpleRNN, LSTM, GRU, and Bidirectional LSTM were explored, with Bidirectional LSTM reaching an accuracy of 84%. The project effectively demonstrated the potential of sentiment analysis in extracting meaningful insights from Twitter data, despite challenges such as handling mixed emotions, word ambiguity, and noisy data. These findings provide a robust framework for future sentiment analysis tasks, offering actionable insights for businesses and researchers to gauge public sentiment in real time.

Introduction
Twitter sentiment analysis is a pivotal application of natural language processing (NLP) and machine learning aimed at understanding public opinion and emotions conveyed through tweets. With the explosive growth of social media, particularly Twitter, analyzing sentiments expressed by users has become crucial for businesses, organizations, and researchers alike. This project focuses on leveraging a dataset of 1,600,000 annotated tweets, categorized into positive and negative sentiments, to develop robust sentiment analysis models. By employing comprehensive data preprocessing techniques such as stop words removal, negation handling, and emoji replacement, the dataset was cleaned and prepared for analysis. Through exploratory data analysis (EDA), including the generation of word clouds to visualize frequent terms, the project gained insights into the most prevalent positive and negative sentiments expressed on Twitter. Subsequently, both traditional machine learning models (e.g., Logistic Regression, Naive Bayes, Decision Tree Classifier) and advanced deep learning architectures (e.g., Bidirectional LSTM) were trained and evaluated for sentiment classification, achieving high accuracies. Despite inherent challenges like handling mixed emotions and noisy data, the project exemplifies the efficacy of sentiment analysis in deriving actionable insights from Twitter data, offering a robust framework for future applications in sentiment analysis across diverse domains.




Methodology


About Dataset and Data Collection

This is the sentiment Twitter dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 1 = positive) and they can be used to detect sentiment .In Dataset Total 6 columns Id, date,flag, user, text(tweets) which are inputdata and Target(sentiment) is predict data.  

Data Preprocessing

1.Drop unwanted Features : Drop Id, date,flag, user column take only two column    text(tweets) and Target(sentiment).
2.Stop Words Removal: Removing common words (e.g., "and", "the") that do not contribute to sentiment.
3.Nagation Handling  : text like should’t convert in to should not.
4.Lower Casing: Each text is converted to lowercase.
5.Replacing URLs: Links starting with 'http' or 'https' or 'www' are replaced by '<url>'.
6.Replacing Usernames: Replace @Usernames with word '<user>'. [eg: '@Kaggle' to '<user>'].
7.Replacing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. [eg: 'Heyyyy' to 'Heyy']
8.Replacing Emojis: Replace emojis by using a regex expression. [eg: ':)' to '<smile>']
9.Replacing Contractions: Replacing contractions with their meanings. [eg: "can't" to 'can not']
10.Removing Non-Alphabets: Replacing characters except Digits, Alphabets and pre-defined Symbols with a space.

EDA ( Exploratory Data Analysis)

Generate a word cloud to visualize the most frequent words in the entire dataset. This provides a general overview of common terms. Good, Love, Thank, Better are Most
Frequent positive word and Not, Miss, Bad, Sad are most Frequent negative word.

Model Building

1.Word Embeding : Word embedding is a technique used to represent words in a continuous vector space where words with similar meanings have similar representations. For vectorization use ML techniques TF-IDF (Term Frequency - Inverse Document Frequency)  , word2vec (Skip-gram and CBOW).

2.Machine Learning Model : Use Machine Learning Model for train the Data and Classify sentiment is positive or negative .Machine learning Classification Algorithms
Like Logistic Regression, naive bayes, decision tree classifier, SVC (Support Vector Classifier) and In Ensemble Learning Bagging algorithms Random forest and 
Boosting algorithms like gradient boost, xgboost.for better Accuracy use gridsearchcv for hyperparameter tuning.

Algoritms	Accuracy
Naive Bayes	82 %
KNeighborsClassifier	78 %
Logistic Regression	76 %
Support Vector Machine	79 %
Random Forest	77 %
Gradient Boost	76  % 
XG Boost	75 %
Using Machine learning Technique I Got  82 % Acuracy .This Machine Learning model Is Logistic Regression.


3.Deep Learning Model : Use Deep Learning for solving sentiment classify problem.
After Preprocesing of the Data and Vectorization of the Data use pad sequences
For do same length of input sequences.then add embeding Layer in Sequential model.then add  Neural network Architecture Like SimpleRnn ,LSTM (Long Term Short Memory),GRU(Gated Recurrent Unit), Bidirectional LSTM . 

In Deep learning Got 84 % Accuarcy using Bidirectional LSTM.

Model Evaluation and Results

Using Machine learning Technique I Got  82 % Acuracy .This Machine Learning model Is Logistic Regression.
In Deep learning Got 84 % Accuarcy using Bidirectional LSTM.


Challenges and Limitations
Mixed Emotions : A single text can contain mixed emotions, which can be difficult for models to capture accurately. For example, "I love the new design, but the performance is disappointing."
Ambiguity: Words with multiple meanings (polysemy) can confuse sentiment analysis models. For example, "cool" can mean both "cold" and "awesome."
Noisy Data : Social media and other user-generated content can be noisy, containing slang, abbreviations, and spelling errors, which can affect the performance of sentiment analysis models.
Computational Resources: Advanced sentiment analysis, especially using deep learning models, requires significant computational resources for training and inference.


Conclusion
Overall, the project successfully demonstrated the potential of sentiment analysis in extracting meaningful insights from Twitter data. The high accuracy achieved by the models, especially the Bidirectional LSTM, underscores the value of using advanced deep learning techniques for sentiment classification. Despite the challenges and limitations, the project provides a robust framework for future sentiment analysis tasks and offers actionable insights for businesses and researchers to gauge public sentiment in real time. The methods and models developed in this project can be further refined and adapted to various domains to enhance sentiment analysis capabilities.
