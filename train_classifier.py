"""
Trains a multi-label classifier for text messages using a ML Pipeline

Loads data from database, cleans and tokenizes the text data, builds the ML
pipeline, trains, tests and evaluates the model and exports it as a pickle file.

Arguments:
    database_filepath:    path to SQLite destination database (e.g. disaster_response_db.db)
    model_filepath:       path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)

Usage:
    python train_classifier.py <database_filepath> <model_filepath>

Execution Example:
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl

"""

import nltk

import sys
import os
import re
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """ Function to load dataset from an sqlite database.

    Arguments: 
        database_filepath: name of database
    
    Returns: 
        X: features dataframe
        Y: labels dataframe
        category_names: list of categories' names
    """  
    
    engine = create_engine('sqlite:///'+ database_filepath)
    path, file = os.path.split(database_filepath)
    # print(path, file)
    table_name = file.replace(".db","") + "Table"
  
    sql_query = f"SELECT * FROM {table_name}"
    # print(sql_query)
    df = pd.read_sql(sql_query, engine)
    
    X = df['message']

    # categories are all the columns after the 4th column
    Y = df.iloc[:,4:]

    category_names = Y.columns #or without values
    return X, Y, category_names


def tokenize(text):
    """ Function to tokenize text.

    Replaces URLs with a placeholder, splits text into words, applies 
    lemmatization and case normalization

    Arguments: 
        text: text to be cleaned
    
    Returns: 
        clean_tokens: list of clean tokens from text
    """  

    # Replace URLs with urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Split text into words
    tokens = word_tokenize(text)

    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        #  Lemmatization and case normalization
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """ Function to build the ML pipeline.

    Arguments: 
        None
    
    Returns: 
        model: Scikit Pipeline or GridSearchCV object
    """ 


    # Create custom transformer

    class StartingVerbExtractor(BaseEstimator, TransformerMixin):

        def starting_verb(self, text):
            sentence_list = nltk.sent_tokenize(text)
            for sentence in sentence_list:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            return False

        def fit(self, x, y=None):
            return self

        def transform(self, X):
            X_tagged = pd.Series(X).apply(self.starting_verb)
            return pd.DataFrame(X_tagged)

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(LinearSVC()))
    ])

    model = pipeline

    # Use grid search to find better parameters #

    check pipeline parameters
    pipeline.get_params()

    parameters = {
        'clf__estimator__loss': ('hinge', 'squared_hinge'),
        'clf__estimator__C': (0.01, 0.5, 1.0)
    } 


    cv = GridSearchCV(estimator=pipeline, n_jobs = -1, param_grid=parameters)
    
    model = cv

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ Function to evaluate the ML model.
    It outputs the overall accuracy score, as well as the f1 score, precision 
    and recall for each output category of the dataset

    Arguments: 
        model:  Scikit Pipeline or GridSearchCV object
        X_test: features of test dataset
        Y_test: labels of test dataset
        category_names: list of categories' names
    
    Returns: 
        None
    """  

    Y_pred = model.predict(X_test)

    # overall accuracy
    accuracy = (Y_pred == Y_test).mean().mean()
    print('Accuracy {0:.2f}% \n'.format(accuracy*100))

    # print(classification_report(Y_test.values, Y_pred, category_names))

    # if we use grid search
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """
    Function to save the trained model as a Pickle file that can be loaded 
    later.
    
    Arguments:
        model: Scikit Pipeline or GridSearchCV object
        model_filepath: destination path to save .pkl file
    
    Returns: 
        None
    """  

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
        
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