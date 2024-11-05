import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin



class DepreciationImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
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

        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def cap_coe_outliers(X):
    """
    Cap coe values to the highest recorded value for given category
    """
    coe_cap_A = 106000  # Max COE for Category A (cars with engine capacity ≤ 1600 cc)
    coe_cap_B = 150001  # Max COE for Category B (cars with engine capacity > 1600 cc)
    coe_cap_C = 91101

    conditions = [
        (X['engine_cap'] <= 1600) & (X['type_of_vehicle'].isin(['hatchback', 'suv', 'mid-sized sedan', 'luxury sedan', 'mpv', 'sports car', 'stationwagon', 'other'])),
        (X['engine_cap'] > 1600) & (X['type_of_vehicle'].isin(['hatchback', 'suv', 'mid-sized sedan', 'luxury sedan', 'mpv', 'sports car', 'stationwagon', 'other'])),
        X['type_of_vehicle'].isin(['van', 'truck','bus/mini bus'])
    ]

    capping = [
        np.minimum(X['coe'], coe_cap_A),
        np.minimum(X['coe'], coe_cap_B),
        np.minimum(X['coe'], coe_cap_C)
    ]

    X['coe'] = np.select(conditions, capping, default=X['coe'])

    return X

def calc_vehicle_age(df):
    
    current_year = datetime.now().year
    df['car_age'] = current_year - df['manufactured']

    return df