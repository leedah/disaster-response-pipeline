import sys
import pandas as pd
from sqlalchemy import create_engine

"""
Data Preprocessing using an ETL Piepleine.

Loads, cleans and saves data into an sqlite database.

Arguments:
    messages_filepath:   path to the csv file containing messages (e.g. disaster_messages.csv)
    categories_filepath: path to the csv file containing categories (e.g. disaster_categories.csv)
    database_filename:   path to SQLite destination database (e.g. disaster_response_db.db)

Usage:
    python process_data.py <messages_filepath> <categories_filepath> <database_filename>

Sample Execution:
    python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db

"""



def load_data(messages_filepath, categories_filepath):
    """ Function to load messages and categories data

    Arguments: 
        messages_filepath : path to messages csv file
        categories_filepath : path to categories csv file
    
    Returns: 
        df: dataframe with merged datasets
    """ 

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'], how='outer')

    return df



def clean_data(df):
    """ Function to clean the dataframe.

    Splits categories into separate category columns, converts category values
    to just numbers 0 or 1, replace categories column in df with new category 
    columns and removes duplicates.

    Arguments: 
        df : dataframe with raw data
    
    Returns: 
        df: dataframe with cleaned data
    """ 

    categories = df['categories'].str.split(";", expand=True)
    # categories.head()

    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # use first row removing the last two characters for list of column names
    category_colnames = [x[:-2] for x in row]
    # print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)

    # check number of duplicates
    # print(df.duplicated().sum())
    
    # drop duplicates
    df = df.drop_duplicates()

    # check number of duplicates
    # print(df.duplicated().sum())

    return df



def save_data(df, database_filename):
    """ Function to save dataset into an sqlite database.

    Arguments: 
        df : dataframe with raw data
        database_filename: name of database
    
    Returns: 
        None
    """  

    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "Table"
    df.to_sql('table_name', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()