import re
import numpy as np
import pandas as pd

def rough_text_cleaner(feature: pd.Series) -> pd.Series:
    """
    This function performs a rough text cleaning operation on a pandas Series.
    It converts the Series to lowercase, removes leading and trailing whitespace,
    replaces special characters and spaces with underscores, and fills NaN values with empty strings.
    
    Parameters:
    feature (pd.Series): The input pandas Series to be cleaned.
    
    Returns:
    pd.Series: The cleaned pandas Series.
    """
    feature = feature.fillna('') 
    print(f" Initial number of unique : {feature.nunique()} ")
    feature = feature.str.lower().str.strip().apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]', '_', x))
    print(f" Post cleaning : {feature.nunique()} ")
    return feature

def display_feature_dist_by_ref_col(df: pd.DataFrame, feature:str, ref:str):
    # Ensure df has model_make
    df["model_make"] = df.model + "_" + df.make
    # Check if there are any nan
    check_list = list()
    for elem in df[df[feature].isna()][ref].unique():
        check_list.append(df[df[ref] ==  elem][feature].isnull().all())
    print(f"Number of {ref} unable to be inputted {check_list.count(True)}")
    return df.groupby(ref)[feature].unique()

def inpute_by_ref_col(df: pd.DataFrame, feature:str, ref:str, method:str):

    # Calculate the mean of column B grouped by column A
    values = df.groupby(ref)[feature].agg(
        [method]).rename(
            columns={method: feature})[feature]
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # If column B is NaN, fill it with the mean value of column B for the corresponding value in column A
        if pd.isna(row[feature]):
            df.at[index, feature] = values[row[ref]]
    
    return df