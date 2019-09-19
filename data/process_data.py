import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge data from different files

    Parameters:
    messages_filepath (file - string): Disaster messages
    categories_filepath (file - string): Category of the disaster messages

    Returns:
    dataframe : Merged dataframe of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Clean the categories data, as there are multiple values in a column and merge with the original dataframe without categories

    Parameters:
    df : Pandas Dataframe

    Returns:
    df : Transformed Dataframe
    """
    categories = df.categories.str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[1,:]
    #extract a list of column names
    category_colnames = row.apply(lambda x: x.split('-')[0]).values.tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)

    # convert column from string to numeric
    categories[column] = categories[column].astype('int')
    #dropping the original categories column
    df.drop(['categories'], axis = 1, inplace=True)
    #joining both the dataframes
    df = pd.merge(df, categories, left_index=True, right_index=True)
    return df


def save_data(df, database_filename):
    """
    Store the data in a database

    Parameters:
    df : Pandas Dataframe
    database_filename : Database Filename

    Stores the data in database_filename
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disasters', engine, index=False, if_exists='replace')


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
