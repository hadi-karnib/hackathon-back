import numpy as np
import pandas as pd
import warnings
from dvc import api
import re
warnings.filterwarnings("ignore", category=FutureWarning)
import subprocess

def execute(command):
    process = subprocess.run(command, shell=True, text=True, capture_output=True)
    print("Command:", command)
    print("Return code:", process.returncode)
    print("Output:", process.stdout)
    print("Error:", process.stderr)

def add_file(file):
    execute(f"git rm -r --cached \"{file}\"")
    execute(f"git commit -m 'stop tracking {file}'")
    execute(f"dvc add {file}")
    execute(f"git add {file}.dvc .gitignore")
    execute(f"git commit -m 'Add {file} to DVC'")
    execute(f"dvc push")

def add_version(file):
    execute(f"dvc add {file}")
    execute(f"git add {file}.dvc")
    execute(f"git commit -m 'Add {file} version to DVC'")
    execute(f"dvc push")

def convert_size(size):
    if isinstance(size, str):
        if 'k' in size:
            return float(size.replace('k', "")) * 1024
        elif 'M' in size:
            return float(size.replace('M', "")) * 1024 * 1024
        elif 'Varies with device' in size:
            return np.nan
    return size


def clean_installs(df):
    df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', "") if '+' in str(x) else x)
    df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', "") if ',' in str(x) else x)
    df['Installs'] = df['Installs'].apply(lambda x: int(x))
    return df

def bin_categories(df):
    bins = [-1, 0, 10, 1000, 10000, 100000, 1000000, 10000000, 10000000000]
    labels=['no', 'Very low', 'Low', 'Moderate', 'More than moderate', 'High', 'Very High', 'Top Notch']
    df['Installs_category'] = pd.cut(df['Installs'], bins=bins, labels=labels)
    return df


def clean_price(df):
    df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if '$' in str(x) else x)
    df['Price'] = df['Price'].apply(lambda x: float(x))
    return df


def specific_clean(df):
    df.drop_duplicates(inplace = True)
    df = df.drop(df[df['Category'] == '1.9'].index)
    df.loc[9148, 'Type'] = 'Free'
    df = df.drop(df[df['Android Ver'].isnull()].index)
    return df

def clean_data(file_path):
    # load csv into df
    df = pd.read_csv(file_path)

    # clean specific values for this table
    df = specific_clean(df)
    df.rename(columns={
    'Last Updated': 'Last_Updated',
    'Android Ver': 'Android_Ver',
    'Content Rating' : 'Content_Rating'
    }, inplace=True)
    # drop name duplicates and keep the record with highest nb of reviews
    df = df.loc[df.groupby('App')['Reviews'].idxmax()]
    
    # clean size
    df['Size'] = df['Size'].apply(convert_size)
    df['Size'] = df['Size'].apply(lambda x: (x/(1024*1024)))
    df.rename(columns={'Size': "Size_in_MB"}, inplace=True)
    df['Size_in_MB'] = df['Size_in_MB'].round(2)

    # clean installs
    df = clean_installs(df)
    
    # binning categories
    df = bin_categories(df)
    
    # clean price
    df = clean_price(df)
    
    # fill null ratings
    mean_by_cat = df.groupby('Category')['Rating'].transform('mean')
    df['Rating'] = df['Rating'].fillna(mean_by_cat)
    df['Rating'] = df['Rating'].round(2)
    
    # drop some values
    df.dropna(subset = ['Size_in_MB', 'Current Ver'], inplace=True)
    #df = df[df['App'] != "哈哈姆特不EY"]
    # concert datatypes
    df['Reviews'] = df['Reviews'].astype('int')
    df['Last_Updated'] = pd.to_datetime(df['Last_Updated'])

    return df



def clean_reviews (file_path) :
# Read the CSV file into a DataFrame
    df = pd.read_csv('googleplaystore_user_reviews.csv')

    # Drop nulls in the required columns
    df.dropna(subset = ['Sentiment_Polarity', 'Sentiment_Subjectivity', 'App'], inplace=True)

    # group by `App` and find mean of `Sentiment_Polarity` and `Sentiment_Subjectivity`
    df_result=df.groupby('App')[['Sentiment_Polarity', 'Sentiment_Subjectivity']].mean().reset_index()
    df_result['Mean_App_Sentiment'] = 0.3* df_result['Sentiment_Polarity'] + 0.2 * df_result['Sentiment_Subjectivity']
    return df_result

def combine_tables():
    df1 = clean_data('googleplaystore.csv')
    df2 = clean_reviews('googleplaystore_user_reviews.csv')
    df_merged = pd.merge(df1, df2, on='App', how='left')
    
    return df_merged


import random

def split_dataframe_into_three(df):


    # Shuffle the DataFrame's index randomly
    shuffled_index = df.index.to_list()
    random.shuffle(shuffled_index)

    # Calculate the size of each part
    part_size = len(df) // 3

    # Split the DataFrame into three parts based on the shuffled index
    part1 = df.loc[shuffled_index[:part_size]].copy()
    part2 = df.loc[shuffled_index[part_size:2 * part_size]].copy()
    part3 = df.loc[shuffled_index[2 * part_size:]].copy()

    return part1, part2, part3


df = combine_tables()
final_df = df[['App','Category','Installs_category',"Rating",'Mean_App_Sentiment','Reviews','Size_in_MB',"Type","Price","Content_Rating","Genres","Last_Updated","Android_Ver"]]
final_df.to_csv('finaldataset.csv', index = False)

df1, df2, df3 = split_dataframe_into_three(final_df)
df2= pd.concat([df1,df2])

df1.to_csv('first_data.csv' , index =False)
df2.to_csv('second_data.csv' , index =False)