import re
import numpy as np
import pandas as pd

from typing import Any, Union

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

## General EDA Utils
def count_missing_rows_for_x_cols(df: pd.DataFrame, column_name: list) -> int:
    """
    Count the number of rows with missing values in the specified column.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to check for missing values.
    
    Returns:
    int: The number of rows with missing values in the specified column.
    """
    print(f"Individual :: {df[column_name].isnull().sum()}")
    missing_count = df[df[column_name].isnull().all(axis=1)].shape[0]
    return missing_count

def retun_numeric_cols(df:pd.DataFrame):
    return df.select_dtypes(include=['int64', 'float64']).columns

def isnan(value: Any) -> bool:
    """Returns True if value is NaN, otherwise False"""
    return value != value

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

def inpute_by_ref_col(df: pd.DataFrame, feature:str, ref:str, method:str="mean"):

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

def find_fuel_type(row:pd.Series, fuel_type:str):
    find_fuel_type = any(
                fuel_type in token.strip().lower() for token in re.compile(r"; |,|\*|\s|&|\|").split(row)
            )
    # Check if any exits 
    if find_fuel_type:
        return fuel_type
    else: 
        return None

def fuel_type_row_extractor(row:pd.Series)->str:
    # Check if the fuel type is present and not NaN
    if isinstance(row.fuel_type, str) and row.fuel_type:
        return row.fuel_type    
    else:
        for text in [row.get("description"), row.get("features"),row.get("category")]:
            if not isinstance(text, str):
                continue    
            # check petrol-electric
            if find_fuel_type(text,"petrol-electric"):
                print(f"row pe {text}")
                return "petrol-electric"
            # check diesel-electric
            if find_fuel_type(text,"diesel-electric"):
                print(f"row de {text}")
                return "diesel-electric"
            # check diesel
            if find_fuel_type(text,"diesel"):
                return "diesel"
            # check electric
            if find_fuel_type(text,"electric"):
                return "electric"
            # check petrol
            if find_fuel_type(text,"petrol"):
                return "petrol"

        return np.nan
    
def get_fuel_type(df:pd.DataFrame)-> pd.DataFrame:
    """
    Gets fuel type from feature/description

    """
    print(f"Original Imputation Nan {df.fuel_type.isna().sum()}")
    df.fuel_type = df.apply(fuel_type_row_extractor, axis=1)
    print(f"After Imputation Nan {df.fuel_type.isna().sum()}")
    return df

def encoding_vehicle_type_custom(type_of_vehicle: str):
    """
    Groups the different types of vehicles into a smaller number
    of categories to handle sparsity issues by assigning a number
    to each meso-group of vehicles. After this has been
    run, the column should be made categorical
    """
    VEHICLE_CATEGORIES = [
    {"sports car"},
    {"luxury sedan", "suv"},
    {"others", "mpv", "stationwagon", "mid-sized sedan"},
    ]
    if not type_of_vehicle or not isinstance(type_of_vehicle, str):
        type_of_vehicle = "others"

    for cat_num, cat in enumerate(VEHICLE_CATEGORIES, start=1):
        if type_of_vehicle in cat:
            return cat_num

    return 0

def generic_one_hotencoding(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column_name]))
    
    return pd.concat([df, encoded_df], axis=1)

def onehot_fit_transform(df:pd.DataFrame, column_name:str="type_of_vehicle"):
    """
    returns encoded vehicle type to the df and the scale
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[[column_name]])
    return onehot_type_fit(df, encoder,column_name), encoder

def onehot_type_fit(df:pd.DataFrame, encoder, column_name:str="type_of_vehicle"):
    df = df.copy().reset_index(drop=True)
    encoded_data = encoder.transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column_name]))
    
    return pd.concat([df, encoded_df], axis=1)

def generic_outlier(df:pd.DataFrame, column_name:str, min_val:Union[float,None], max_val:Union[float,None]):
    proc_df = df.copy()
    orig_na = proc_df[column_name].isna().sum()
    if min_val:
        proc_df[column_name] = proc_df[column_name].where(proc_df[column_name] > min_val, np.nan)
    if max_val:
        proc_df[column_name] = proc_df[column_name].where(proc_df[column_name] < max_val, np.nan)
    print(f" For {column_name} column :: Found {proc_df[column_name].isna().sum() - orig_na} outliers")
    return proc_df