import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from utils.sgcarmart_scraper import get_emission_data
import os
from tqdm import tqdm

### make model imputation ###
def extract_make(title, compiled_regex):
    """
    Extract the car make from the title using the compiled regex.
    :param title: str - The title from which to extract the make
    :param compiled_regex: regex pattern - Precompiled regex pattern
    :return: str - Extracted make or empty string if no match
    """
    title = title.lower()
    matches = compiled_regex.findall(title)
    return matches[0] if matches else None

def compile_make_pattern(make_list):
    """
    Compile a regex pattern from the list of makes.
    :param make_list: list - List of car makes
    :return: regex pattern - Compiled regex pattern to find car makes
    """
    # Combine patterns into a single regex
    combined_pattern = r'\b(' + '|'.join(make_list) + r')\b'
    # Precompile the regex
    return re.compile(combined_pattern)

def apply_make_extraction(df, compiled_regex):
    """
    Apply the extraction function to the 'title' column of the dataframe.
    :param df: DataFrame - The dataframe containing the 'title' column
    :param compiled_regex: regex pattern - Precompiled regex pattern
    :return: DataFrame - The dataframe with an 'extracted_make' column
    """
    df['extracted_make'] = df['title'].apply(lambda x: extract_make(x, compiled_regex))
    return df

### cylinder extraction ###
def extract_cylinder_from_features(feature):
    # Regex pattern to match engine capacity (floating-point number followed by 'l')
    pattern = r'\b(\d+)[ |-]?cylinder[s]?\b |\b[v|V][ ]?(\d+)\b'

    # Use re.search to extract the engine capacity
    match = re.search(pattern, feature)

    if match:
        cylinder_cnt = match.group(1) or match.group(2)
        cylinder_cnt = int(cylinder_cnt)
        if cylinder_cnt <= 12:
            return cylinder_cnt
        else:
            return None
    else:
        return None
    
def extract_cylinder_by_model_make(df):
    cylinder_count_dict = (
        df.groupby(['make', 'model'])['cylinder_cnt']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None) # agg by dataframe and take mode 
        .dropna()  # Exclude groups with no valid cylinder count
        .to_dict()
    )
    return cylinder_count_dict

def impute_row_by_make_model(row, dict_to_impute, col):
    # If col is missing, look up (make, model) in the dictionary
    if pd.isnull(row[col]):
        return dict_to_impute.get((row['make'], row['model']), row[col])
    return row[col]


class LTADataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, df_lta_car_data):
        self.df_lta_car_data = df_lta_car_data.copy()
    
    def fit(self, X, y=None):
        """
        No fitting necessary for this imputer.

        """
        return self
    
    def transform(self, X):
        X['omv'] = X.apply(self._impute_using_lta_data, axis=1)
        return X


    def _impute_using_lta_data(self,row):
        # print(row)
        if not np.isnan(row['omv']):
            return row['omv']
        else:
            make = row['make']
            model = row['model']
            year = row['reg_date_year']
            lookup_by_make_model_year = self.df_lta_car_data[(self.df_lta_car_data['make_clean'] == make) & (self.df_lta_car_data['model_split'].str.contains(model)) & (self.df_lta_car_data['year'] == year)]
            
            lookup_by_make_model = self.df_lta_car_data[(self.df_lta_car_data['make_clean'] == make) & (self.df_lta_car_data['model_split'].str.contains(model))]

            lookup_by_make = self.df_lta_car_data[(self.df_lta_car_data['make_clean'] == make)]
            
            if not lookup_by_make_model_year.empty:
                # print('lookup_by_make_model_year')
                return lookup_by_make_model_year['omv_clean'].mean()
            elif not lookup_by_make_model.empty:
                # print('lookup_by_make_model')
                return lookup_by_make_model['omv_clean'].mean()
            elif not lookup_by_make.empty:
                # print('lookup_by_make')
                return lookup_by_make['omv_clean'].mean()
            else: 
                return None

class MakeModelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, make_list):
        self.make_list = make_list

        self.combined_pattern = r'\b(' + '|'.join(map(re.escape, self.make_list)) + r')\b'
        self.compiled_regex = re.compile(self.combined_pattern, re.IGNORECASE)


    def fit(self, X, y=None):
        """
        Do nothing as regex is compiled during init
        """
        return self

    def transform(self, X):
        """
        Apply the extraction to the 'title' column.
        """
        X = X.copy()
        X['make'] = X['title'].apply(self._extract_make)
        return X

    def _extract_make(self, title):
        title = str(title).lower()
        matches = self.compiled_regex.findall(title)
        return matches[0] if matches else title.split()[0] # if no match, return first word of the title

class CylinderExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # No initialization parameters

    def fit(self, X, y=None):
        """
        No fitting necessary for this transformer.
        """
        return self

    def transform(self, X):
        """
        Apply the extraction to the 'features' column.
        """
        X = X.copy()
        X['cylinder_cnt'] = X['features'].apply(self._extract_cylinder_from_features)
        return X

    def _extract_cylinder_from_features(self, feature):
        if pd.isnull(feature):
            return None
        # Regex pattern to match cylinder counts
        pattern = r'\b(\d+)[ -]?cylinder[s]?\b|\b[Vv][ ]?(\d+)\b'
        match = re.search(pattern, str(feature))
        if match:
            cylinder_cnt = match.group(1) or match.group(2)
            try:
                cylinder_cnt = int(cylinder_cnt)
                if cylinder_cnt <= 12 and cylinder_cnt >= 1:
                    return cylinder_cnt
            except ValueError:
                return None
        return None
    
class CylinderImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cylinder_count_dict = None

    def fit(self, X, y=None):
        """
        Compute the mode of cylinder counts for each (make, model).
        """
        # Ensure 'make', 'model', and 'cylinder_cnt' columns exist
        if not all(col in X.columns for col in ['make', 'model', 'cylinder_cnt']):
            raise ValueError("Columns 'make', 'model', and 'cylinder_cnt' must be present in the data.")

        # Create a dictionary for imputation
        self.cylinder_count_dict = (
            X.groupby(['make', 'model'])['cylinder_cnt']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None) # agg by dataframe and take mode 
            .dropna()  # Exclude groups with no valid cylinder count
            .to_dict()
            )
        return self

    def transform(self, X):
        """
        Impute missing 'cylinder_cnt' values based on the mode from training data.
        """
        X = X.copy()
        X['cylinder_cnt'] = X.apply(self._impute_row_by_make_model, axis=1)
        return X

    def _impute_row_by_make_model(self, row):
        if pd.isnull(row['cylinder_cnt']):
            return self.cylinder_count_dict.get((row['make'], row['model']), row['cylinder_cnt'])
        return row['cylinder_cnt']
    
class CategoryParser(BaseEstimator, TransformerMixin):
    def __init__(self, column='category'):
        """
        Initializes the transformer.

        Parameters:
        - column (str): The name of the column to parse.
        """
        self.column = column
        self.mlb = MultiLabelBinarizer()
        self.categories_ = None  # To store the unique categories learned during fit

    def fit(self, X, y=None):
        """
        Fits the transformer by learning the unique categories.

        Parameters:
        - X (DataFrame): The input DataFrame.
        - y: Ignored.

        Returns:
        - self: Returns the instance itself.
        """
        # Check if the column exists in X
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")

        # Extract categories from the column
        category_lists = X[self.column].apply(self._parse_categories)

        # Fit the MultiLabelBinarizer to learn the categories
        self.mlb.fit(category_lists)
        self.categories_ = self.mlb.classes_
        return self

    def transform(self, X):
        """
        Transforms the input DataFrame by parsing the categories.

        Parameters:
        - X (DataFrame): The input DataFrame.

        Returns:
        - X_transformed (DataFrame): The transformed DataFrame with new category features.
        """
        # Check if the column exists in X
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")

        # Copy X to avoid modifying the original DataFrame
        X_transformed = X.copy()

        # Parse the categories
        category_lists = X_transformed[self.column].apply(self._parse_categories)

        # Transform the categories into a binary indicator format
        category_dummies = pd.DataFrame(
            self.mlb.transform(category_lists),
            columns=self.mlb.classes_,
            index=X_transformed.index
        )

        # Concatenate the new category features with the original DataFrame
        X_transformed = pd.concat([X_transformed, category_dummies], axis=1)

        return X_transformed

    def _parse_categories(self, category_string):
        """
        Parses a category string into a list of categories.

        Parameters:
        - category_string (str): The category string to parse.

        Returns:
        - list: A list of categories.
        """
        if pd.isnull(category_string):
            return []
        # Split the string by commas and strip whitespace
        categories = [cat.strip().lower() for cat in category_string.split(',')]
        return categories


class EmissionImputer():
    '''
    Data is extracted from SGCarMart
    '''
    def __init__(self, train_csv_dir=None, test_csv_dir=None):
        # checks if files were scrapped
        if os.path.exists(train_csv_dir):
            self.train_emission_df = pd.read_csv(train_csv_dir)
        else:
            self.train_emission_df = self._scrap_emission_data_from_sgcarmart(train_emission_df, train_csv_dir)
            
        # checks if files were scrapped
        if os.path.exists(test_csv_dir):
            self.test_emission_df = pd.read_csv(test_csv_dir)
        else:
            self.test_emission_df = self._scrap_emission_data_from_sgcarmart(test_emission_df, test_csv_dir)
        self.preprocess_extracted_data()

    def _scrap_emission_data_from_sgcarmart(self, df, csv_dir):
        df['scrapped_emission_data'] = None
        failed_idx = []
        # Iterate over each row with index
        for index, row in tqdm(df.iterrows()):
            # print(index, row)
            try:
                # Check if data is already scrapped to resume operation
                if pd.isna(row['scrapped_emission_data']) or row['scrapped_emission_data'] is None:
                    # Apply the get_emission_data function and store in the DataFrame
                    df.at[index, 'scrapped_emission_data'] = get_emission_data(row['listing_id'], row['title'])
            except Exception as e:
                print(e)
                failed_idx.append(index)
            # Save progress every few rows to a file 
            if index % 100 == 0:  
                df.to_csv("progress.csv", index=False)
        
        # Save final progress after the loop completes
        df.to_csv(csv_dir, index=False)
        return df

    def preprocess_extracted_data(self):
        def extract_emission(text):
            if not isinstance(text, str):
                return None
            # Regex pattern to extract the numeric value before "g/km"
            pattern = r"(\d+)\s*g/km"
        
            # Search for the pattern in the text
            match = re.search(pattern, text)
            
            # Return the numeric value if found, otherwise None
            if match:
                return int(match.group(1))  # Convert to integer if needed
            else:
                return None
        self.train_emission_df['emission_data'] = self.train_emission_df['scrapped_emission_data'].apply(extract_emission)
        self.test_emission_df['emission_data'] = self.test_emission_df['scrapped_emission_data'].apply(extract_emission)

    
    def impute_values(self, df, df_type):
        # Check for df_type validity
        if df_type not in ["train", "test"]:
            raise ValueError('Please pass df_type="train" or df_type="test" for imputation.')

        # Select DataFrame based on type
        emission_df = self.train_emission_df if df_type == "train" else self.test_emission_df
        
        # merge df with emission_df on listing id
        merged_df = pd.merge(df, emission_df[['listing_id', 'emission_data']], on='listing_id', how='left')

        return merged_df

        

############################## FORMAT ####################################
# cylinder_imputer = CylinderImputer()
# train_df = cylinder_imputer.fit_transform(train_df) # 1479 rows missing
# val_df = cylinder_imputer.transform(val_df) # 334 rows missing
# test_df = cylinder_imputer.transform(test_df)

# # Many rows are missing, impute using median values
# imputer = SimpleImputer(strategy='median')
# train_df['cylinder_cnt'] = imputer.fit_transform(train_df[['cylinder_cnt']])
# val_df['cylinder_cnt'] = imputer.transform(val_df[['cylinder_cnt']])
# test_df['cylinder_cnt'] = imputer.transform(test_df[['cylinder_cnt']])