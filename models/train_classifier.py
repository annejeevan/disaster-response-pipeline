# import libraries
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict, Counter
#natural language processing
import nltk
nltk.download(['stopwords','punkt', 'wordnet'])
from nltk.corpus import brown
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
stop_words = set(stopwords.words('english'))
#machine learning
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import chi2, SelectKBest
import re
import string
import math
import pickle
import warnings
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Retrieve the data from the database

    Parameters:
    database_filepath : Database Filename where data is stored

    Returns:
    X : Features
    y : Labels
    category_names : Data Viz
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disasters', con=engine)
    X = df['message']
    y = df.iloc[:,4:]
    categories = y.columns.tolist()
    return X, y, categories

def find_url(string):
    # findall() has been used
    # with valid conditions for urls in string
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return url


def tokenize(text):
    """
    Convertes text to words
    """
    urls = find_url(text)
    for url in  urls:
        text = text.replace(url, "URL")
    #case folding - normalizing
    text = text.lower()
    # remove punctuations
    words = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text))
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # to reduce our words to a single word, better than stemming
    words = [WordNetLemmatizer().lemmatize(word) for word in words]
    return words


def build_model():
    """
    Selects the best model with optimal parameters
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('svd', TruncatedSVD()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
        ])

    parameters = {'clf__estimator__n_estimators': [25, 50],
                  'clf__estimator__max_depth': [10, 25]}

    cv_rf = GridSearchCV(pipeline, parameters)
    return cv_rf

def clf_report_mult(y_test, y_pred, category_names):
    """
    Classification report
    """
    clf_results = pd.DataFrame(columns=['category', 'precision', 'recall', 'f1_score'])
    for i in range(0, len(category_names)):
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test[category_names[i]], y_pred[:,i], average='weighted')
        clf_results = clf_results.append({'category': category_names[i], 'precision': precision, 'recall': recall, 'f1_score': f1_score}, ignore_index=True)
    print(clf_results)

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluating the model based on the predictions

    Returns:
    Output of the performance metrics
    """
    y_pred = model.predict(X_test)
    clf_report_mult(Y_test, y_pred, category_names=category_names)
    clf_report = clf_report_mult(Y_test, y_pred, category_names=category_names)

def save_model(model, model_filepath):
    """
    Serialize the model created

    Parameters:
    model: Best model created
    model_filepath: Path to store the pickle file

    Stores the model in a pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        Y = Y.apply(pd.to_numeric)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
