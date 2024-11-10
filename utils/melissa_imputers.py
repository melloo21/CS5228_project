import os
import pickle
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer

class ModelMakeImputer(BaseEstimator, TransformerMixin):

    def __init__(self, column_a, column_b, savepath=None, impute_type="median"):
        # model make column
        self.column_a = column_a
        # to be imputed column
        self.column_b = column_b
        self.impute_type = impute_type
        self.values = None

    def fit(self, df):
        """
        Learn the imputation values for column_b grouped by column_a.
        """
        # Model and Make 
        # Calculate the specified impute_type (mean, median, etc.) of column_b grouped by column_a
        self.values = df.groupby(self.column_a)[self.column_b].agg(self.impute_type)

        return self

    def transform(self, df):
        """
        Transform the DataFrame by filling NaN values in column_b using learned imputation values.
        """
        if self.values is None:
            raise ValueError("The imputer has not been fitted yet. Please call fit() first.")

        # Fill NaN values in column_b using the imputation values learned from fit()
        df[self.column_b] = df[self.column_b].fillna(df[self.column_a].map(self.values))
        
        return df

    def fit_transform(self, df):
        """
        Fit and transform the DataFrame in a single step.
        """
        self.fit(df)
        return self.transform(df)

class GenericKNNImputer(BaseEstimator, TransformerMixin):

    def __init__(self, feature:list, neighbours:int, imputed_feature:list):
        self.feature = feature
        self.neighbours = neighbours
        self.imputed_feature = imputed_feature
    
    def fit(self, df:pd.DataFrame):
        
        impute_df = df[self.feature]
        # Initialize KNNImputer
        imputer = KNNImputer(n_neighbors=self.neighbours)
        # Fit imputer 
        self.imputer = imputer.fit(impute_df)

    def transform(self, df:pd.DataFrame):

        impute_df = df[self.feature]
        # Perform imputation
        imputed_array = self.imputer.transform(impute_df)     
        # Create a DataFrame from the imputed array
        imputed_df = pd.DataFrame(imputed_array, columns=self.feature)

        return imputed_df[self.imputed_feature]

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

class GenericSGCarMartImputer:

    def __init__(self, train_pickle_dir=None, test_pickle_dir=None):

        if os.path.exists(train_pickle_dir):
            self.train_scraped = pd.read_pickle(train_pickle_dir)
        else:
            raise NotImplementedError

        if os.path.exists(test_pickle_dir):
            self.test_scraped = pd.read_pickle(test_pickle_dir)
        else:
            raise NotImplementedError
        
        # [TODO] Tidy up map class
        self.map_class = {
            'power': 'Power',
            'fuel_type' : 'Fuel Type',
            'engine_cap': 'Engine Capacity',
            'curb_weight': 'Kerb Weight'
        }

    def impute_if_missing(self, row, variable, ref_df):

        if pd.isnull(row[variable]):
            if ref_df[ref_df.listing_id==row["listing_id"]][self.map_class[variable]].empty:
                return np.NaN
            else:
                return ref_df[ref_df.listing_id==row["listing_id"]][self.map_class[variable]].values[0]
        else:
            return row[variable]

    def impute_val(self, df, variable:str, df_type:str):
        assert df_type in ["train", "test"] , ValueError('Please pass df_type="train" or df_type="test" for imputation.')
        
        ref_df = self.train_scraped if df_type == "train" else self.test_scraped
        
        # Merge df with variable_df on listing id
        return df.apply(self.impute_if_missing,axis=1, variable=variable, ref_df=ref_df)