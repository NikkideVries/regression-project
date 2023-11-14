# imports: 
import pandas as pd
import numpy as np

#
import env
import os

from sklearn.model_selection import train_test_split




#--------------------Data Aquisition------------------------#

# get conncection url:
def get_db_url(db, user= env.user, host=env.host, password=env.password):
    """
    This function will:
    - take credentials from env.py file
    - make a connection to the SQL database with given credentials
    - return url connection
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# make the zillow data based on the function
# make a function based on this data:
def new_zillow_data():
    '''
    This function will:
    - read a set sql query
    - return a dataframe based on the given query
    '''

    zillow_query = '''
    SELECT
        transactiondate,
        bathroomcnt,
        bedroomcnt,
        calculatedfinishedsquarefeet,
        fips,
        taxvaluedollarcnt,
        propertylandusetypeid,
        propertylandusedesc,
        yearbuilt
    FROM properties_2017
        JOIN predictions_2017 USING(parcelid)
        JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE propertylandusedesc='Single Family Residential' AND transactiondate LIKE '2017%%'
        '''
        
    # read in the dataframe from codeup
    df = pd.read_sql(zillow_query, get_db_url('zillow'))
    
    return df

# make the zillow data into a csv: 
def get_zillow_data():
    '''
    This functino will check for a zillow.csv,
    If it exists it will pull data from said file.
    '''
    
    if os.path.isfile('zillow.csv'):
        #if csv file exists read in data from csv file:
        df = pd.read_csv('zillow.csv', index_col = 0)
        
    else:
        
        #read the fresh data form db into a dataframe
        df = new_zillow_data()
        
        #cache data:
        df.to_csv('zillow.csv')
    
    return df




#------------------Data Peperation---------------#

# clean the data: 
def clean_zillow(df):
    '''
    This function:
    - This function will drop nulls
    - This function, will drop unescessary columns
    - Rename the existing columns
    '''
    
    # drop nulls
    df = df.dropna()
    
    col_list = ['transactiondate','propertylandusetypeid','propertylandusedesc','fips']
    
    #drop column names in col_list
    df = df.drop(columns = col_list)
    
    #rename the columns
    df = df.rename(columns ={
    'bathroomcnt': 'bathrooms',
    'bedroomcnt': 'bedrooms',
    'calculatedfinishedsquarefeet': 'sqft',
    'taxvaluedollarcnt': 'tax_value'
})  
    return df



# feature engineering
def create_columns(df):
    '''
    This function will complete some feature engineering and produce two columns:
    house age = how old the house is
    bb_roomcnt = bedrooms with bathrroms 
    '''
    
    df['house_age'] = 2017 - df['yearbuilt']
    
    df['bb_roomcnt'] = df['bathrooms'] + df['bedrooms']
    
    
    return df



# split function

def split_zillow_data(df):
    '''
    This funciton will split the data frame into train, validate, and test
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test



# remove outliers:
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:
        
        # For each column, it calculates the first quartile (q1) and 
        #third quartile (q3) using the .quantile() method, where q1 
        #corresponds to the 25th percentile and q3 corresponds to the 75th percentile.
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#what could be outliers:
def outlier(df, feature, m):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound


# prep zillow: 
def prep_zillow():
    '''
    This funciton will complete all the function listed above to make a clean perpared dataset.
    get zillow, clean zillow, create columns, remove outliers, split data
    '''
    
    # get the data frame:
    df = get_zillow_data()
    
    #clean the data:
    df = clean_zillow(df)
    
    #create columns:
    df = create_columns(df)
    
    # remove outliers:
    all_columns = ['bathrooms','bedrooms','sqft','tax_value','yearbuilt','house_age','bb_roomcnt']
    df = remove_outliers(df, 3, all_columns)
    
    train, validate, test = split_zillow_data(df)
    
    return train, validate, test



