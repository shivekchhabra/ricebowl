Data-Preprocessing
==================
Documentation of ricebowl data preprocessing.


read_csv
^^^^^^^^
General function to read a csv file.

Parameters- Path of the csv file

Output- Dataframe

Usage::
    
    df=read_csv(path)


read_excel
^^^^^^^^^^
General function to read an excel file.

Parameters- Path of the excel file, Sheet name

Output- Dataframe

Usage::
    
    df=read_excel(path,sheet_name)


reformat_col_headers
^^^^^^^^^^^^^^^^^^^^
General function for formatting the column headers to lower case.
All "spaces" and "-" are replaced by "_"

Parameters- Dataframe

Output- Dataframe with formatted column headers

Usage::
    
    df=reformat_col_headers(df)


str_to_datetime
^^^^^^^^^^^^^^^
General function to convert string columns in date format to datetime.
Add the names of the columns which need to be converted to datetime from string type.

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with the columns updated to datetime

Usage::
    
    df=str_to_datetime(df, c1='col1', c2='col2' .... cn='col_n')


timestamp_to_datetime
^^^^^^^^^^^^^^^^^^^^^
General function to convert timestamp columns to datetime.
Add the names of the columns which need to be converted to datetime from timestamp.

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with the columns updated to datetime

Usage::
    
    df=timestamp_to_datetime(df, c1='col1', c2='col2' .... cn='col_n')


to_timestamp
^^^^^^^^^^^^
General function to convert datetime/str columns in datetime format to timestamp.
Add the names of the columns which need to be converted to timestamp.

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with the columns updated to timestamp

Usage::
    
    df=to_timestamp(df, c1='col1', c2='col2' .... cn='col_n')


label_encode
^^^^^^^^^^^^
General function to label encode the categorical columns.
Add the names of the columns which need to be encoded.

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with the columns updated with encoded labels, label encoder object

Usage::
    
    df=label_encode(df, c1='col1', c2='col2' .... cn='col_n')


one_hot_encode
^^^^^^^^^^^^^^
General function to one-hot encode the categorical columns.
Add the names of the columns which need to be encoded.

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with the columns updated with encoded labels

Usage::
    
    df=one_hot_encode(df, c1='col1', c2='col2' .... cn='col_n')


dates_diff
^^^^^^^^^^
General function to calculate the difference between 2 date columns.

Parameters- Dataframe, column 1, column 2, diff_type(Optional; Default='days'; Takes in 'days'/'weeks'/'months'/'years') 

Output- Dataframe with a new column according to difference. For example if diff_type='weeks' then the new column will be of the name 'weeks'

Error Print- If a wrong "diff_type" is provided, prints an error message.

Usage::
    
    df=dates_diff(df,col1,col2,diff_type='days')



drop_duplicates
^^^^^^^^^^^^^^^
General function to remove duplicate rows.

Parameters- Dataframe

Output- Dataframe without duplicate rows.

Usage::
    
    df=drop_duplicates(df)


reset_index
^^^^^^^^^^^
General function to reset the index of the dataframe.

Parameters- Dataframe, Drop(True/False)

Output- Dataframe with a new index

Usage::
    
    df=reset_index(df,drop=True)


to_dtype
^^^^^^^^
General function to convert a column to a particular datatype.

Parameters- Dataframe, Data type, kwargs[column names]

Output- Dataframe with updated columns

Usage::
    
    df=to_dtype(df, 'float', c1='col1', c2='col2'...., cn='col_n')



fill_mode
^^^^^^^^^
General function to fill null values with mode.

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with the columns updated. The null values in the columns will be filled with the mode of that column.

Usage::
    
    df=fill_mode(df, c1='col1', c2='col2' .... cn='col_n')


fill_mean
^^^^^^^^^
General function to fill null values with mean.

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with the columns updated. The null values in the columns will be filled with the mean of that column.

Usage::
    
    df=fill_mean(df, c1='col1', c2='col2' .... cn='col_n')


melt
^^^^
General function to melt data.

Parameters- Dataframe, Columns to melt(in the form of a list), New column name to be made after melting, Column name displaying values; Default- 'value'

Output- Dataframe with the columns updated. The data is melted.

Usage::
    
    df=melt(df, ['col1','col2'...'col_n'], 'new_col_name_xyz', value)


split_columns
^^^^^^^^^^^^^
General function to make existing data a list of split values.

Parameters- Dataframe, Original Column, Separator to split on

Output- Dataframe with columns seperated.
Example: if a column had dates like 2019-01-01 and we use this function with a separator '-', then the data will be modified to [2019,01,01] 

Usage::

    df=split_columns(df, 'column_name', separator='-')


remove_unwanted_chars
^^^^^^^^^^^^^^^^^^^^^
General function to remove unwanted characters from data.

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with unwanted characters removed. (like $,€,£,inr,¥,₹) 

Usage::

    df=remove_unwanted_chars(df, c1='col1', c2='col2' .... cn='col_n')

 
fill_num_abbreviations
^^^^^^^^^^^^^^^^^^^^^^
General function to fill "million M", "billion B", "thousand k", "lakhs L", "crore cr"

Parameters- Dataframe, kwargs[column names]

Output- Dataframe with filled abbreviations.
Example: 20k would be replaced by 20000

Usage::

    df=fill_num_abbreviations(df, c1='col1', c2='col2' .... cn='col_n')




