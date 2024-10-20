import pickle
import pandas as pd

class ModelMakeImputer:
    def __init__(self, column_a, column_b, savepath, impute_type="median"):
        self.column_a = column_a
        self.column_b = column_b
        self.impute_type = impute_type
        self.values = None

    def fit(self, df):
        """
        Learn the imputation values for column_b grouped by column_a.
        """
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

    def save_scale(self):
        """Saving the values as a pickle"""

        if (self.values is None) and (self.path is None):
            raise ValueError("The imputer has not been fitted yet. Please call fit() first.")

        else: 
            return