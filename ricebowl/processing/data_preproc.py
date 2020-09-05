import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

np.random.seed(7)


# Overview:
# This is the code for RiceBowl Preprocessing.
# There are various data pre-processing
# and cleaning functionalities.

# General function to read a csv file
def read_csv(path):
    data = pd.read_csv(path)
    return data


# General function to read an excel file
def read_excel(path, sheet_name):
    xl = pd.ExcelFile(path)
    df = pd.read_excel(xl, sheet_name)
    return df


# General function for formatting the column headers to lower case.
# All "spaces" and "-" are replaced by "_"
def reformat_col_headers(df):
    col_list = list(df.columns.values)
    processed_col_list = [re.sub(r'\s+', '_', item.strip().lower()) for item in
                          col_list]
    processed_col_list = [x.replace("-", "_") for x in processed_col_list]
    col_dict = dict(zip(col_list, processed_col_list))
    df.rename(columns=col_dict, inplace=True)
    return df


# General Function to convert string columns in date format to datetime
def str_to_datetime(df, **col_names):
    for i in col_names.values():
        df[i] = pd.to_datetime(df[i], dayfirst=True)
    return df


# General Function to convert timestamp columns to datetime
def timestamp_to_datetime(df, **col_names):
    for i in col_names.values():
        df[i] = pd.to_datetime(df[i], dayfirst=True, unit='s',
                               errors='coerce')
    return df


# General Function to convert datetime/str columns in datetime format to timestamp
def to_timestamp(df, **col_names):
    for col in col_names.values():
        df[col] = pd.to_datetime(df[col], utc=True, dayfirst=True)
        df[col] = df[col].apply(lambda x: x.timestamp())
    return df


# General Function to label encode the categorical columns
def label_encode(df, **col_names):
    le = LabelEncoder()
    for i in col_names.values():
        df[i] = le.fit_transform(df[i])
    return df, le


# General Function to one-hot encode the categorical columns
def one_hot_encode(df, **col_names):
    for col in col_names.values():
        encoded_column = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, encoded_column], axis=1).drop(col, axis=1)

    return df


# General Function to calculate the difference between 2 date columns.
def dates_diff(df, col1, col2, diff_type='days'):
    df = str_to_datetime(df, c1=col1, c2=col2)
    if diff_type == 'days':
        df['days'] = (np.absolute(df[col1] - df[col2])).dt.days
    elif diff_type == 'months':
        df['months'] = (np.absolute(df[col1] - df[col2])).dt.days // 30
    elif diff_type == 'years':
        df['years'] = (np.absolute(df[col1] - df[col2])).dt.days // 365
    elif diff_type == 'weeks':
        df['weeks'] = (np.absolute(df[col1] - df[col2])).dt.days // 7
    else:
        print(
            'Wrong difference string provided. Please provide "days","weeks","years" or "months".')
    return df


# General Function to remove duplicate rows.
def drop_duplicates(df):
    df = df.drop_duplicates(ignore_index=True)
    return df


# General Function to reset index
def reset_index(df, drop=True):
    df = df.reset_index(drop=drop)
    return df


# General Function to convert a column to a particular datatype
def to_dtype(df, dtype, **cols):
    if dtype == str:
        for col in cols.values():
            df[col] = df[col].fillna(' ')
    for col in cols.values():
        df[col] = df[col].astype(dtype)

    return df


# General Function to fill null values with mode
def fill_mode(df, **col_names):
    for i in col_names.values():
        df[i] = df[i].fillna(df[i].mode()[0])
    return df


# General Function to fill null values with mean
def fill_mean(df, **col_names):
    for i in col_names.values():
        df[i] = df[i].fillna(round(df[i].mean(), 3))
    return df


# General Function to melt data
# parameters:
# df: DataFrame to be melted
# cols_to_melt: cols that are needed as rows instead of columns in the form of a list
# new_col_name: name of the new column formed by melted values in the form of a string
# value_name: (optional, default: 'value') name of value column of melted values
def melt(df, cols_to_melt, new_col_name, value_name='value'):
    unchanged_cols = list(set(df.columns) - set(cols_to_melt))
    new_df = pd.melt(
        df,
        id_vars=unchanged_cols, value_vars=list(cols_to_melt), var_name=new_col_name, value_name=value_name)

    return new_df


# General Function to make existing data a list of split values
def split_columns(df, orig_col, separator):
    df[orig_col] = df[orig_col].apply(lambda x: str(x).split(separator))
    return df


# General Function to remove unwanted characters from data
def remove_unwanted_chars(df, **col_names):
    for i in col_names.values():
        df[i] = df[i].map(lambda x: str(x).lstrip('*+-~$€£inr¥₹')
                          .rstrip('inrsec*sdmywminmokKmMbB'))
    return df


# General Function to fill "million M", "billion B", "thousand k", "lakhs L", "crore cr"
def fill_num_abbreviations(df, **col_names):
    for i in col_names.values():
        df = to_dtype(df, str, c1=i)
        df[i] = df[i].apply(lambda x: x.lower())
        df[i] = df[i].apply(lambda x: x + '000000000' if 'b' in str(x) else x).apply(
            lambda x: x.replace('b', '') if 'b' in str(x) else x)
        df[i] = df[i].apply(lambda x: x + '000000' if 'm' in str(x) else x).apply(
            lambda x: x.replace('m', '') if 'm' in str(x) else x)
        df[i] = df[i].apply(lambda x: x + '000' if 'k' in str(x) else x).apply(
            lambda x: x.replace('k', '') if 'k' in str(x) else x)
        df[i] = df[i].apply(lambda x: x + '00000' if 'l' in str(x) else x).apply(
            lambda x: x.replace('l', '') if 'l' in str(x) else x)
        df[i] = df[i].apply(lambda x: x + '0000000' if 'cr' in str(x) else x).apply(
            lambda x: x.replace('cr', '') if 'cr' in str(x) else x)
    return df


# General Function to split data for modeling purpose
def split_data(data, label, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=test_size,
                                                        random_state=7,
                                                        shuffle=True)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test


# General Function to find correlation excluding all null values
def find_corr(df, method='pearson'):
    corr = df.corr(method=method)
    return corr


# General Function to find outliers in a random variable using zscore
def zscore_outliers(series):
    data = list(series)
    mean = np.mean(data)
    std = np.std(data)
    threshold = 2.5
    outlier = []
    for i in data:
        z = (i - mean) / std
        if z > threshold:
            outlier.append(i)
    return outlier


# General function to standardize the data using standard scaler.
def standarization(data, list_of_cols=['ALL']):
    scaling = StandardScaler()
    if list_of_cols == ['ALL']:
        cols = list(data.columns)
    else:
        cols = list_of_cols

    data = scaling.fit_transform(data[cols])
    df = pd.DataFrame(data, columns=cols)
    return df


# General function to normalize the data using min-max scaler.
def normalization(data, list_of_cols=['ALL']):
    scaling = MinMaxScaler()
    if list_of_cols == ['ALL']:
        cols = list(data.columns)
    else:
        cols = list_of_cols

    data = scaling.fit_transform(data[cols])
    df = pd.DataFrame(data, columns=cols)
    return df
