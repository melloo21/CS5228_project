import re
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

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

def calc_vehicle_age(df):
    
    current_year = datetime.now().year
    df['car_age'] = current_year - df['manufactured']

    return df

def calculate_depreciation_rate_without_age(df):

    # Keep non null records
    dftemp = df.dropna(subset=['depreciation'])

    # Group by 'make' and 'model' to calculate the average depreciation for each group
    avg_depreciation_rate_wo_age = (
        dftemp.groupby(['extracted_make', 'model'])['depreciation']
        .mean()  # Calculate the mean depreciation for each group
        .to_dict()
    )
    
    return avg_depreciation_rate_wo_age

def calculate_depreciation_rate_with_age(df):

    # Keep non null records
    dftemp = df.dropna(subset=['depreciation'])

    # Group by 'make', 'model' and 'age' to calculate the average depreciation for each group
    avg_depreciation_rate_with_age = (
        dftemp.groupby(['extracted_make', 'model', 'car_age'])['depreciation']
        .mean()
        .to_dict()
    )
    
    return avg_depreciation_rate_with_age

def impute_depreciation(df, avg_depreciation_rate_without_age, avg_depreciation_rate_with_age):

    mean_depreciation = df['depreciation'].mean()
    for i, row in df.iterrows():
        # If depreciation is missing
        if pd.isnull(row['depreciation']):
            make_model = (row['extracted_make'], row['model'])
            make_model_age = (row['extracted_make'], row['model'], row['car_age'])

            # Null records with matching make-model-age group
            if make_model_age in avg_depreciation_rate_with_age:
                df.at[i, 'depreciation'] = avg_depreciation_rate_with_age[make_model_age]
            # Null records with matching make-model group but different vehicle age
            elif make_model in avg_depreciation_rate_without_age:
                df.at[i, 'depreciation'] = avg_depreciation_rate_without_age[make_model]
            # Null records with no matching make-model
            else:
                df.at[i, 'depreciation'] = mean_depreciation
    return df


class DepreciationImputer(BaseEstimator, TransformerMixin):
    def init(self):
        self.avg_depreciation_rate_wo_age = {}
        self.avg_depreciation_rate_with_age = {}
        self.mean_depreciation = None

    def fit(self, X, y = None):
        """
        Group by make, model and age and calculate mean depreciation for each group
        """
        self.avg_depreciation_rate_with_age = (
        X.groupby(['make', 'model', 'car_age'])['depreciation']
        .mean()
        .to_dict()
    )
        self.avg_depreciation_rate_wo_age = (
        X.groupby(['make', 'model'])['depreciation']
        .mean()
        .to_dict()
    )
        return self
    
    def transform(self, X):
        """
        Impute using mean values of same make, model, age group
        """
        self.mean_depreciation = X['depreciation'].mean()
        for i, row in X.iterrows():
        # If depreciation is missing
            if pd.isnull(row['depreciation']):
                make_model = (row['make'], row['model'])
                make_model_age = (row['make'], row['model'], row['car_age'])

                # Null records with matching make-model-age group
                if make_model_age in self.avg_depreciation_rate_with_age:
                    X.at[i, 'depreciation'] = self.avg_depreciation_rate_with_age[make_model_age]
                # Null records with matching make-model group but different vehicle age
                elif make_model in self.avg_depreciation_rate_wo_age:
                    X.at[i, 'depreciation'] = self.avg_depreciation_rate_wo_age[make_model]
                # Null records with no matching make-model
                else:
                    X.at[i, 'depreciation'] = self.mean_depreciation

        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
