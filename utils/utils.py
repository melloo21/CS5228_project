import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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
        self.compiled_regex = None

    def fit(self, X, y=None):
        """
        Compile the regex pattern from the make list.
        """
        combined_pattern = r'\b(' + '|'.join(map(re.escape, self.make_list)) + r')\b'
        self.compiled_regex = re.compile(combined_pattern, re.IGNORECASE)
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
        return matches[0] if matches else None
    


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
