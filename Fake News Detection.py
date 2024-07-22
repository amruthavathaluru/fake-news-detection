import pandas as pd
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import metrics
from matplotlib import pyplot as plt
import itertools

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the Dataset
train_df = pd.read_csv(r'C:\\Users\\Amrutha\\Downloads\\train.csv')

# Verify column names and content
print(train_df.columns)
print(train_df.head())

# Data Cleaning (if necessary, but here we assume only 'Statement' and 'Label' are present)
columns_to_drop = []  # No columns to drop if only 'Statement' and 'Label' are present

# Check if 'Statement' column exists
if 'Statement' not in train_df.columns:
    raise KeyError("'Statement' column not found in the dataset. Check your CSV file.")

# Exploratory Data Analysis
def data_quality_check():
    print("Checking data qualities...")
    train_df.info()
    print("Check finished.")

data_quality_check()

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def preprocess_text(text):
    corpus = []
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Ensure text is string type
    text = text.lower()
    words = nltk.word_tokenize(text)
    for word in words:
        if word not in stpwrds:
            corpus.append(lemmatizer.lemmatize(word))
    return ' '.join(corpus)

# Apply text preprocessing to 'Statement' column
train_df['Statement'] = train_df['Statement'].apply(preprocess_text)

# Feature Extraction
tfidf_v = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf_v.fit_transform(train_df['Statement'])

# Splitting Data
label_train = train_df.Label
X_train, X_test, Y_train, Y_test = train_test_split(tfidf_train, label_train, test_size=0.2, random_state=42)

# Model Training and Evaluation
classifier = PassiveAggressiveClassifier()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
score = metrics.accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {round(score*100,2)}%')

cm = metrics.confusion_matrix(Y_test, Y_pred)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])

# Saving and Loading the Model
with open('./model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('./model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Prediction Function
def fake_news_det(news):
    news = preprocess_text(news)
    input_data = [news]
    vectorized_input_data = tfidf_v.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    if prediction[0] == 0:
        print("Prediction of the News : Looking Fake⚠ News📰 ")
    else:
        print("Prediction of the News : Looking Real News📰 ")

# Example usage of prediction function
fake_news_det("This is a test news article.")
