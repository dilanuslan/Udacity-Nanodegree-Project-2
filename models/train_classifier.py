#import necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Loads data from the database

    Parameters:
        database_filepath: file path of the SQLite database

    Returns:
        X: Features
        Y: Target
        columns: category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('project_2', con=engine) 
    X = df['message']
    Y = df.iloc[:,4:]
    columns = df.columns
    return X,Y,columns


def tokenize(text):
    """
    Tokenize and lemmatize the given text

    Parameters:
        text: Text to tokenize and lemmatize
        
    Returns:
        clean_tokens: tokens created after the operations
    """
    #tokenize text
    tokens = word_tokenize(text)
    
    #initiate the lemmatizer
    lemmatizer = WordNetLemmatizer()

    #iterate through each token
    clean_tokens = []
    for tok in tokens:
        #lemmatize, normalize, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Creates a pipeline and implements Grid Search to choose the best parameters

    Returns:
        cv: multi output classifier that was built with GridSearch
    """
    #build the pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))       
    ])

    #choose the parameter to use in the grid search
    parameters = {'clf__estimator__n_estimators':[10,50]}

    #apply grid search to pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of the model and prints the precision, recall, and f1score for each category. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each category
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Exports the final model into a pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Performs the operations defined in this script
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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
