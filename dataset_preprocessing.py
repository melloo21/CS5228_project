import os
import time
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import boxcox, skew
import matplotlib.pyplot as plt

from IPython.display import display, HTML
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder , PowerTransformer, MultiLabelBinarizer , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

## Local Utils File
from utils.eda import *
from utils.road_tax import *
from utils.kanhon_utils import *
from utils.melissa_imputers import *
from utils.melissa_utils import *
from utils.Bhushan_utils import *
from utils.lta_omv_scraper import *
from utils.models import Regression
from utils.regression_evaluation import *

import argparse




def outlier_processing(orig_df):
    print(f' Initial Nan {orig_df[["curb_weight","power","engine_cap"]].isna().sum()}')
    orig_df = generic_outlier(df=orig_df,column_name='curb_weight',min_val=440, max_val=46000)
    orig_df = generic_outlier(df=orig_df,column_name='power',min_val=1, max_val=None)
    orig_df = generic_outlier(df=orig_df,column_name='engine_cap',min_val=1, max_val=None)
    print(f' Initial Nan {orig_df[["curb_weight","power","engine_cap"]].isna().sum()}')
    return orig_df

def coe_outlier_processing(train_df, val_df, test_df):
        # Cap Outliers - Dependency on 'engine_cap' and 'type_of_vehicle'
    train_df = cap_coe_outliers(train_df)
    val_df = cap_coe_outliers(val_df)
    test_df = cap_coe_outliers(test_df)
    return train_df, val_df, test_df

def make_model_imputer(train_df, val_df, test_df):
    make_df = pd.read_csv(r"./dataset/make.csv")
    make_ls = [make.lower() for make in make_df['Make List'].unique()]

    make_model_imputer = MakeModelImputer(make_ls)

    train_df = make_model_imputer.transform(train_df)
    val_df = make_model_imputer.transform(val_df) 
    test_df = make_model_imputer.transform(test_df) 

    # Generating model make imputer
    train_df["model_make"] = train_df.model + "_" + train_df.make
    val_df["model_make"] = val_df.model + "_" + val_df.make
    test_df["model_make"] = test_df.model + "_" + test_df.make

    return train_df, val_df, test_df

def coe_age_imputer(train_df, val_df, test_df):
    ## New method
    coeAge_impute = CoeAgeImputer()

    train_df = coeAge_impute.fit_transform(df=train_df)
    val_df = coeAge_impute.fit_transform(val_df)
    test_df = coeAge_impute.fit_transform(test_df)
    display(f' After imputation : {train_df["coe_age_left"].isna().sum()}')
    return train_df, val_df, test_df

def agerangeprocessor(train_df, val_df, test_df):
    ## New method
    ageRangeProc_impute = AgeRangeproc_dfer()

    train_df = ageRangeProc_impute.fit_transform(df=train_df)
    val_df = ageRangeProc_impute.fit_transform(val_df)
    test_df = ageRangeProc_impute.fit_transform(test_df)
    display(f' After imputation train_df : {train_df["age_range"].isna().sum()}')
    display(f' After imputation val_df : {val_df["age_range"].isna().sum()}')
    display(f' After imputation test_df : {test_df["age_range"].isna().sum()}')
    return train_df, val_df, test_df


def car_age_imputer(train_df, val_df, test_df):
    depreciation_imputer = DepreciationImputer()
    train_df = depreciation_imputer.calc_vehicle_age(train_df) # 0 empty records - due to 'manufactured' having 7 empty records
    val_df = depreciation_imputer.calc_vehicle_age(val_df) # 0 empty records
    test_df = depreciation_imputer.calc_vehicle_age(test_df) # 0 empty records

    display(f' After imputation train_df : {train_df["car_age"].isna().sum()}')
    display(f' After imputation val_df : {val_df["car_age"].isna().sum()}')
    display(f' After imputation test_df : {test_df["car_age"].isna().sum()}')
    return train_df, val_df, test_df

def manufactured_date_imputer(train_df, val_df, test_df):
    # Impute manufactured date
    train_df = impute_manufactured_date(train_df) 
    val_df = impute_manufactured_date(val_df) 
    test_df = impute_manufactured_date(test_df)
    return train_df, val_df, test_df

def vehicle_cond_encoder(train_df, val_df, test_df):
    vehicle_encoder = VehicleCondensedEncoder()
    train_df = vehicle_encoder.fit_transform(train_df) # 1537 rows missing
    val_df = vehicle_encoder.transform(val_df) # 379 rows missing
    test_df = vehicle_encoder.transform(test_df) # 789 rows missing
    return train_df, val_df, test_df

def vehicle_one_hot_encoder(train_df, val_df, test_df):
# ## Processing 
    train_df , vehicle_type_encoder = onehot_fit_transform(train_df, "type_of_vehicle")
    val_df = onehot_type_fit(val_df, vehicle_type_encoder)
    test_df = onehot_type_fit(test_df, vehicle_type_encoder)
    return train_df, val_df, test_df


def curb_weight_imputer(train_df, val_df, test_df, simple_impute=False):

    train_pkl_dir_path = "./dataset/proc_scraped_new_car_info.pkl"
    test_pkl_dir_path = "./dataset/test_proc_scraped_new_car_info.pkl"
    variable = "curb_weight"

    sgcarmart_imputer = GenericSGCarMartImputer(
        train_pickle_dir=train_pkl_dir_path, test_pickle_dir=test_pkl_dir_path)

    display(f' train_df Before imputation {variable} : {train_df[variable].isna().sum()}')
    display(f' val_df Before imputation : {val_df[variable].isna().sum()}')
    display(f' test_df Before imputation : {test_df[variable].isna().sum()}')

    train_df.loc[:,variable] = sgcarmart_imputer.impute_val(df=train_df,variable=variable,df_type="train")
    val_df.loc[:,variable]  = sgcarmart_imputer.impute_val(df=val_df,variable=variable,df_type="train")
    test_df.loc[:,variable]  = sgcarmart_imputer.impute_val(df=test_df,variable=variable,df_type="test")

    display(f' train_df After imputation : {train_df[variable].isna().sum()}')
    display(f' val_df After imputation : {val_df[variable].isna().sum()}')
    display(f' test_df After imputation : {test_df[variable].isna().sum()}')

    curb_weight_impute = ModelMakeImputer(column_a="model_make", column_b="curb_weight")
    display(f' train_df Before imputation {variable} : {train_df[variable].isna().sum()}')
    display(f' val_df Before imputation : {val_df[variable].isna().sum()}')
    display(f' test_df Before imputation : {test_df[variable].isna().sum()}')

    train_df = curb_weight_impute.fit_transform(train_df) # 82 rows missing
    val_df = curb_weight_impute.transform(val_df) # 20 rows missing
    test_df = curb_weight_impute.transform(test_df) # 40 rows missing

    display(f' train_df After imputation : {train_df[variable].isna().sum()}')
    display(f' val_df After imputation : {val_df[variable].isna().sum()}')
    display(f' test_df After imputation : {test_df[variable].isna().sum()}')

    if simple_impute:
        imputer = SimpleImputer(strategy='median')
        train_df['curb_weight'] = imputer.fit_transform(train_df[['curb_weight']])
        val_df['curb_weight'] = imputer.transform(val_df[['curb_weight']])
        test_df['curb_weight'] = imputer.transform(test_df[['curb_weight']])
    return train_df, val_df, test_df

def power_imputer(train_df, val_df, test_df, simple_impute=False):
    train_pkl_dir_path = "./dataset/proc_scraped_new_car_info.pkl"
    test_pkl_dir_path = "./dataset/test_proc_scraped_new_car_info.pkl"
    variable = "power"

    sgcarmart_imputer = GenericSGCarMartImputer(
        train_pickle_dir=train_pkl_dir_path, test_pickle_dir=test_pkl_dir_path)

    display(f' train_df Before imputation {variable}: {train_df[variable].isna().sum()}')
    display(f' val_df Before imputation : {val_df[variable].isna().sum()}')
    display(f' test_df Before imputation : {test_df[variable].isna().sum()}')

    train_df.loc[:,variable] = sgcarmart_imputer.impute_val(df=train_df,variable=variable,df_type="train")
    val_df.loc[:,variable]  = sgcarmart_imputer.impute_val(df=val_df,variable=variable,df_type="train")
    test_df.loc[:,variable]  = sgcarmart_imputer.impute_val(df=test_df,variable=variable,df_type="test")

    display(f' train_df After imputation : {train_df[variable].isna().sum()}')
    display(f' val_df After imputation : {val_df[variable].isna().sum()}')
    display(f' test_df After imputation : {test_df[variable].isna().sum()}')

    power_impute = ModelMakeImputer(column_a="model_make", column_b="power")

    display(f' train_df Before imputation {variable}: {train_df[variable].isna().sum()}')
    display(f' val_df Before imputation : {val_df[variable].isna().sum()}')
    display(f' test_df Before imputation : {test_df[variable].isna().sum()}')
    train_df = power_impute.fit_transform(train_df) # 1537 rows missing
    val_df = power_impute.transform(val_df) # 379 rows missing
    test_df = power_impute.transform(test_df) # 789 rows missing
    display(f' train_df After imputation : {train_df[variable].isna().sum()}')
    display(f' val_df After imputation : {val_df[variable].isna().sum()}')
    display(f' test_df After imputation : {test_df[variable].isna().sum()}')

    if simple_impute:
        imputer = SimpleImputer(strategy='median')
        train_df['power'] = imputer.fit_transform(train_df[['power']])
        val_df['power'] = imputer.transform(val_df[['power']])
        test_df['power'] = imputer.transform(test_df[['power']])
    return train_df, val_df, test_df


def engine_cap_imputer(train_df, val_df, test_df, simple_impute=False):

    train_pkl_dir_path = "./dataset/proc_scraped_new_car_info.pkl"
    test_pkl_dir_path = "./dataset/test_proc_scraped_new_car_info.pkl"
    variable = "engine_cap"

    sgcarmart_imputer = GenericSGCarMartImputer(
    train_pickle_dir=train_pkl_dir_path, test_pickle_dir=test_pkl_dir_path)


    display(f' train_df Before imputation {variable} : {train_df[variable].isna().sum()}')
    display(f' val_df Before imputation : {val_df[variable].isna().sum()}')
    display(f' test_df Before imputation : {test_df[variable].isna().sum()}')

    train_df.loc[:,variable] = sgcarmart_imputer.impute_val(df=train_df,variable=variable,df_type="train")
    val_df.loc[:,variable]  = sgcarmart_imputer.impute_val(df=val_df,variable=variable,df_type="train")
    test_df.loc[:,variable]  = sgcarmart_imputer.impute_val(df=test_df,variable=variable,df_type="test")

    display(f' train_df After imputation : {train_df[variable].isna().sum()}')
    display(f' val_df After imputation : {val_df[variable].isna().sum()}')
    display(f' test_df After imputation : {test_df[variable].isna().sum()}')

    engine_impute = ModelMakeImputer(column_a="model_make", column_b="engine_cap")

    display(f' train_df Before imputation {variable} : {train_df[variable].isna().sum()}')
    display(f' val_df Before imputation : {val_df[variable].isna().sum()}')
    display(f' test_df Before imputation : {test_df[variable].isna().sum()}')

    train_df = engine_impute.fit_transform(train_df) # 384 rows missing
    val_df = engine_impute.transform(val_df) # 97 rows missing
    test_df = engine_impute.transform(test_df) # 186 rows missing

    display(f' train_df After imputation : {train_df[variable].isna().sum()}')
    display(f' val_df After imputation : {val_df[variable].isna().sum()}')
    display(f' test_df After imputation : {test_df[variable].isna().sum()}')

    if simple_impute:
        # Many rows are missing, impute using median values
        imputer = SimpleImputer(strategy='median')
        train_df['engine_cap'] = imputer.fit_transform(train_df[['engine_cap']])
        val_df['engine_cap'] = imputer.transform(val_df[['engine_cap']])
        test_df['engine_cap'] = imputer.transform(test_df[['engine_cap']])
    return train_df, val_df, test_df

def owner_cnt_imputer(train_df, val_df, test_df):
    ## New method
    display(f' Before imputation : {train_df["no_of_owners"].isna().sum()}')

    owner_impute = OwnerImputer()

    train_df = owner_impute.fit_transform(df=train_df)
    val_df = owner_impute.transform(val_df)
    test_df = owner_impute.transform(test_df)
    display(f' After imputation : {train_df["no_of_owners"].isna().sum()}')
    return train_df, val_df, test_df

def depreciation_imputer(train_df, val_df, test_df, simple_impute=False):
    # Imputation using make, model and car age - Dependency on 'car_age'
    display(f' Before imputation : {train_df["depreciation"].isna().sum()}')

    depreciation_imputer = DepreciationImputer()
    train_df = depreciation_imputer.fit_transform(train_df) # 96 empty records
    val_df = depreciation_imputer.transform(val_df) # 9 empty records
    test_df = depreciation_imputer.transform(test_df) # 9 empty records
    if simple_impute:
        # # Median imputation for remaining records
        imputer = SimpleImputer(strategy='median')
        train_df['depreciation'] = imputer.fit_transform(train_df[['depreciation']])
        val_df['depreciation'] = imputer.transform(val_df[['depreciation']])
        test_df['depreciation'] = imputer.transform(test_df[['depreciation']])
        display(f' After imputation : {train_df["depreciation"].isna().sum()}')
    return train_df, val_df, test_df


def transmission_imputer(train_df, val_df, test_df):
    # One-hot (Binary) encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    train_df['transmission'] = encoder.fit_transform(train_df[['transmission']])
    val_df['transmission'] = encoder.transform(val_df[['transmission']])
    test_df['transmission'] = encoder.transform(test_df[['transmission']])
    return train_df, val_df, test_df


def mileage_imputer(train_df, val_df, test_df):
    display(f' Before imputation : {train_df["mileage"].isna().sum()}')
    ## New method

    mileage_impute = mileageImputerByType()

    train_df = mileage_impute.fit_transform(df=train_df)
    val_df = mileage_impute.transform(val_df)
    test_df = mileage_impute.transform(test_df)

    display(f' After imputation : {train_df["mileage"].isna().sum()}')
    return train_df, val_df, test_df


def omv_imputer(train_df, val_df, test_df, simple_impute=False):
    csv_filename = r'./dataset/lta_omv_data.csv'

    if os.path.exists(csv_filename):
        df_lta_car_data = pd.read_csv(csv_filename)
    else:
        result_ls = get_lta_omv_data(2002, 2025)
        df_lta_car_data = preprocess_lta_omv_data(result_ls)
        df_lta_car_data.to_csv(csv_filename)

    lta_data_imputer = LTADataImputer(df_lta_car_data)

    train_df = lta_data_imputer.transform(train_df) # before transform: 49 rows missing, after transform: 24 rows missing
    val_df = lta_data_imputer.transform(val_df) # before transform: 15 rows missing, after transform: 9 rows missing
    test_df = lta_data_imputer.transform(test_df) 

    if simple_impute:
        imputer = SimpleImputer(strategy='median')
        train_df['omv'] = imputer.fit_transform(train_df[['omv']])
        val_df['omv'] = imputer.transform(val_df[['omv']])
        test_df['omv'] = imputer.transform(test_df[['omv']])
    return train_df, val_df, test_df

def arf_imputer(train_df, val_df, test_df, simple_impute=False, knn_impute=False):
    if simple_impute:
        imputer = SimpleImputer(strategy='median')
        train_df['arf'] = imputer.fit_transform(train_df[['arf']])
        val_df['arf'] = imputer.transform(val_df[['arf']])
        test_df['arf'] = imputer.transform(test_df[['arf']])
    elif knn_impute:
        # Select features relevant for imputation
        features = ['manufactured', 'reg_date_year', 'omv', 'arf', 'type_of_vehicle_bus/mini bus',
            'type_of_vehicle_hatchback', 'type_of_vehicle_luxury sedan',
            'type_of_vehicle_mid-sized sedan', 'type_of_vehicle_mpv',
            'type_of_vehicle_others', 'type_of_vehicle_sports car',
            'type_of_vehicle_stationwagon', 'type_of_vehicle_suv',
            'type_of_vehicle_truck', 'type_of_vehicle_van',]

        arf_knn_imputer = GenericKNNImputer(feature=features, neighbours=5, imputed_feature="arf")

        train_df['arf'] = arf_knn_imputer.fit_transform(train_df)
        val_df['arf'] = arf_knn_imputer.transform(val_df)
        test_df['arf'] = arf_knn_imputer.transform(test_df)

    return train_df, val_df, test_df


def fuel_type_imputer(train_df, val_df, test_df):
    train_df = get_fuel_type(train_df)
    val_df = get_fuel_type(val_df)
    test_df = get_fuel_type(test_df)

    train_pkl_dir_path = "./dataset/proc_scraped_new_car_info.pkl"
    test_pkl_dir_path = "./dataset/test_proc_scraped_new_car_info.pkl"
    variable = "fuel_type"
    sgcarmart_imputer = GenericSGCarMartImputer(
    train_pickle_dir=train_pkl_dir_path, test_pickle_dir=test_pkl_dir_path)

    display(f' train_df Before imputation {variable} : {train_df[variable].isna().sum()}')
    display(f' val_df Before imputation : {val_df[variable].isna().sum()}')
    display(f' test_df Before imputation : {test_df[variable].isna().sum()}')

    train_df.loc[:,variable] = sgcarmart_imputer.impute_val(df=train_df,variable=variable,df_type="train")
    val_df.loc[:,variable]  = sgcarmart_imputer.impute_val(df=val_df,variable=variable,df_type="train")
    test_df.loc[:,variable]  = sgcarmart_imputer.impute_val(df=test_df,variable=variable,df_type="test")

    display(f' train_df After imputation : {train_df[variable].isna().sum()}')
    display(f' val_df After imputation : {val_df[variable].isna().sum()}')
    display(f' test_df After imputation : {test_df[variable].isna().sum()}')

    train_df , fuel_type_encoder = onehot_fit_transform(train_df, column_name=variable)
    val_df = onehot_type_fit(val_df, fuel_type_encoder, column_name=variable)
    test_df = onehot_type_fit(test_df, fuel_type_encoder, column_name=variable)
    return train_df, val_df, test_df


def cylinder_imputer(train_df, val_df, test_df, simple_impute=False):
    cylinder_count_extractor = CylinderExtractor()

    train_df = cylinder_count_extractor.transform(train_df)
    val_df = cylinder_count_extractor.transform(val_df) 
    test_df = cylinder_count_extractor.transform(test_df) 

    cylinder_imputer = CylinderImputer()
    train_df = cylinder_imputer.fit_transform(train_df) # 1479 rows missing
    val_df = cylinder_imputer.transform(val_df) # 334 rows missing
    test_df = cylinder_imputer.transform(test_df)
    if simple_impute:
        imputer = SimpleImputer(strategy='median')
        train_df['cylinder_cnt'] = imputer.fit_transform(train_df[['cylinder_cnt']])
        val_df['cylinder_cnt'] = imputer.transform(val_df[['cylinder_cnt']])
        test_df['cylinder_cnt'] = imputer.transform(test_df[['cylinder_cnt']])
    return train_df, val_df, test_df


def category_parser(train_df, val_df, test_df):
    category_parser = CategoryParser()
    train_df = category_parser.fit_transform(train_df) 
    val_df = category_parser.transform(val_df) 
    test_df = category_parser.transform(test_df) 
    return train_df, val_df, test_df

def co2_emission_imputer(train_df, val_df, test_df, simple_impute=False):
    train_csv_dir_path = "./dataset/train_data_scrapped_co2_emission.csv"
    test_csv_dir_path = "./dataset/test_data_scrapped_co2_emission.csv"
    emission_imputer = EmissionImputer(train_csv_dir=train_csv_dir_path, test_csv_dir=test_csv_dir_path)

    train_df = emission_imputer.impute_values(df=train_df,df_type="train")
    val_df = emission_imputer.impute_values(df=val_df,df_type="train")
    test_df = emission_imputer.impute_values(df=test_df,df_type="test")
    if simple_impute:
    # # Many rows are missing, impute using median values
        imputer = SimpleImputer(strategy='median')
        train_df['emission_data'] = imputer.fit_transform(train_df[['emission_data']])
        val_df['emission_data'] = imputer.transform(val_df[['emission_data']])
        test_df['emission_data'] = imputer.transform(test_df[['emission_data']])

    return train_df, val_df, test_df

def numeric_imputer(train_df, val_df, test_df, impute_type = "KNN", impute_neighbours=5):
    impute_strategy="median", 
     # mean, median, most_frequent, constant, Callable 
    
    random_state = 0
    impute_max_iter= 10

    impute_choice = {
        "simple" : SimpleImputer(strategy=impute_strategy),
        "KNN" : KNNImputer(n_neighbors=impute_neighbours)
    }

    imputer = impute_choice[impute_type]
    cols = retun_numeric_cols(train_df.drop(columns=["indicative_price","price"]))

    train_df[cols] = imputer.fit_transform(train_df[cols])
    val_df[cols] = imputer.transform(val_df[cols])
    test_df[cols] = imputer.transform(test_df[cols])
    return train_df, val_df, test_df

def scalar_transform(train_df, val_df, test_df, features=None):
    if not features:
        features = ['curb_weight', 'power', 'cylinder_cnt', 'omv', 'arf', 'emission_data' ,\
    'engine_cap', 'depreciation', 'mileage', 'coe', 'car_age', 'manufactured', 'road_tax', 'dereg_value']
    scaler = MinMaxScaler()
    # Fit and transform the numerical columns
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])    
    test_df[features] = scaler.transform(test_df[features])    
    return train_df, val_df, test_df

def feature_transform(train_df, val_df, test_df, features=None):
    if not features:
        features = ['curb_weight', 'power', 'cylinder_cnt', 'omv', 'arf', 'emission_data' ,\
    'engine_cap', 'depreciation', 'mileage', 'coe', 'car_age', 'manufactured', 'road_tax', 'dereg_value']
    for feature in features:
        ft = FeatureTransformer()
        ft.fit_transform(train_df[feature])
        train_df[feature] = ft.apply_best_transform(train_df[feature])
        val_df[feature] = ft.apply_best_transform(val_df[feature])
        test_df[feature] =ft.apply_best_transform(test_df[feature])
    return train_df, val_df, test_df


def save_dataset(train_df, val_df, test_df, name):
    ## Saving dataset
    train_df.to_csv(f"./processed_dataset/train_{name}.csv", index=False)
    val_df.to_csv(f"./processed_dataset/val_{name}.csv", index=False)
    test_df.to_csv(f"./processed_dataset/test_{name}.csv", index=False)
    return




def SimpleImputers_OutliersRemoved(orig_df, test_df, args_method):
    
    name = args_method
    orig_df = outlier_processing(orig_df)
    ## Split into train val split
    train_df, val_df = train_test_split(orig_df, test_size=0.2, random_state=42, shuffle=True)
    
    train_df, val_df, test_df = coe_outlier_processing(train_df, val_df, test_df)
    
    train_df, val_df, test_df = make_model_imputer(train_df, val_df, test_df)

    train_df, val_df, test_df = coe_age_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = agerangeprocessor(train_df, val_df, test_df)
    train_df, val_df, test_df = car_age_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = manufactured_date_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = vehicle_cond_encoder(train_df, val_df, test_df)
    # train_df, val_df, test_df = vehicle_one_hot_encoder(train_df, val_df, test_df)
    train_df, val_df, test_df = curb_weight_imputer(train_df, val_df, test_df, simple_impute=True)
    train_df, val_df, test_df = power_imputer(train_df, val_df, test_df, simple_impute=True)
    train_df, val_df, test_df = engine_cap_imputer(train_df, val_df, test_df, simple_impute=True)
    train_df, val_df, test_df = owner_cnt_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = depreciation_imputer(train_df, val_df, test_df, simple_impute=True)
    train_df, val_df, test_df = transmission_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = mileage_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = omv_imputer(train_df, val_df, test_df, simple_impute=True)
    train_df, val_df, test_df = fuel_type_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = cylinder_imputer(train_df, val_df, test_df, simple_impute=True)
    train_df, val_df, test_df = category_parser(train_df, val_df, test_df)
    train_df, val_df, test_df = co2_emission_imputer(train_df, val_df, test_df, simple_impute=True)
    train_df, val_df, test_df = numeric_imputer(train_df, val_df, test_df, impute_type = "KNN", impute_neighbours=5)
    features = ['curb_weight', 'power', 'cylinder_cnt', 'omv', 'emission_data' ,\
    'engine_cap', 'depreciation', 'mileage', 'coe', 'car_age', 'manufactured']
    train_df, val_df, test_df = feature_transform(train_df, val_df, test_df, features=features)
    save_dataset(train_df, val_df, test_df, name)

    print(train_df, val_df, test_df)

def NoSimpleImputers_OutliersRemoved(orig_df, test_df, args_method):
    
    name = args_method
    orig_df = outlier_processing(orig_df)
    ## Split into train val split
    train_df, val_df = train_test_split(orig_df, test_size=0.2, random_state=42, shuffle=True)
    
    train_df, val_df, test_df = coe_outlier_processing(train_df, val_df, test_df)
    
    train_df, val_df, test_df = make_model_imputer(train_df, val_df, test_df)

    train_df, val_df, test_df = coe_age_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = agerangeprocessor(train_df, val_df, test_df)
    train_df, val_df, test_df = car_age_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = manufactured_date_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = vehicle_cond_encoder(train_df, val_df, test_df)
    # train_df, val_df, test_df = vehicle_one_hot_encoder(train_df, val_df, test_df)
    train_df, val_df, test_df = curb_weight_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = power_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = engine_cap_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = owner_cnt_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = depreciation_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = transmission_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = mileage_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = omv_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = fuel_type_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = cylinder_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = category_parser(train_df, val_df, test_df)
    train_df, val_df, test_df = co2_emission_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = numeric_imputer(train_df, val_df, test_df, impute_type = "KNN", impute_neighbours=5)
    features = ['curb_weight', 'power', 'cylinder_cnt', 'omv', 'emission_data' ,\
    'engine_cap', 'depreciation', 'mileage', 'coe', 'car_age', 'manufactured']
    train_df, val_df, test_df = feature_transform(train_df, val_df, test_df, features=features)
    save_dataset(train_df, val_df, test_df, name)

    print(train_df, val_df, test_df)

def NoSimpleImputers_OutliersRemoved_NoFeatureTransform(orig_df, test_df, args_method):
    
    name = args_method
    orig_df = outlier_processing(orig_df)
    ## Split into train val split
    train_df, val_df = train_test_split(orig_df, test_size=0.2, random_state=42, shuffle=True)
    
    train_df, val_df, test_df = coe_outlier_processing(train_df, val_df, test_df)
    
    train_df, val_df, test_df = make_model_imputer(train_df, val_df, test_df)

    train_df, val_df, test_df = coe_age_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = agerangeprocessor(train_df, val_df, test_df)
    train_df, val_df, test_df = car_age_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = manufactured_date_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = vehicle_cond_encoder(train_df, val_df, test_df)
    # train_df, val_df, test_df = vehicle_one_hot_encoder(train_df, val_df, test_df)
    train_df, val_df, test_df = curb_weight_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = power_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = engine_cap_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = owner_cnt_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = depreciation_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = transmission_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = mileage_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = omv_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = fuel_type_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = cylinder_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = category_parser(train_df, val_df, test_df)
    train_df, val_df, test_df = co2_emission_imputer(train_df, val_df, test_df, simple_impute=False)
    train_df, val_df, test_df = numeric_imputer(train_df, val_df, test_df, impute_type = "KNN", impute_neighbours=5)
    # features = ['curb_weight', 'power', 'cylinder_cnt', 'omv', 'emission_data' ,\
    # 'engine_cap', 'depreciation', 'mileage', 'coe', 'car_age', 'manufactured']
    # train_df, val_df, test_df = feature_transform(train_df, val_df, test_df, features=features)
    save_dataset(train_df, val_df, test_df, name)

    print(train_df, val_df, test_df)

def SimpleImputers_OutliersRemoved_NoVehCond(orig_df, test_df, args_method):
    simple_impute = True
    name = args_method
    orig_df = outlier_processing(orig_df)
    ## Split into train val split
    train_df, val_df = train_test_split(orig_df, test_size=0.2, random_state=42, shuffle=True)
    
    train_df, val_df, test_df = coe_outlier_processing(train_df, val_df, test_df)
    
    train_df, val_df, test_df = make_model_imputer(train_df, val_df, test_df)

    train_df, val_df, test_df = coe_age_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = agerangeprocessor(train_df, val_df, test_df)
    train_df, val_df, test_df = car_age_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = manufactured_date_imputer(train_df, val_df, test_df)
    # train_df, val_df, test_df = vehicle_cond_encoder(train_df, val_df, test_df)
    train_df, val_df, test_df = vehicle_one_hot_encoder(train_df, val_df, test_df)
    train_df, val_df, test_df = curb_weight_imputer(train_df, val_df, test_df, simple_impute=simple_impute)
    train_df, val_df, test_df = power_imputer(train_df, val_df, test_df, simple_impute=simple_impute)
    train_df, val_df, test_df = engine_cap_imputer(train_df, val_df, test_df, simple_impute=simple_impute)
    train_df, val_df, test_df = owner_cnt_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = depreciation_imputer(train_df, val_df, test_df, simple_impute=simple_impute)
    train_df, val_df, test_df = transmission_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = mileage_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = omv_imputer(train_df, val_df, test_df, simple_impute=simple_impute)
    train_df, val_df, test_df = arf_imputer(train_df, val_df, test_df, knn_impute=True)
    train_df, val_df, test_df = fuel_type_imputer(train_df, val_df, test_df)
    train_df, val_df, test_df = cylinder_imputer(train_df, val_df, test_df, simple_impute=simple_impute)
    train_df, val_df, test_df = category_parser(train_df, val_df, test_df)
    train_df, val_df, test_df = co2_emission_imputer(train_df, val_df, test_df, simple_impute=simple_impute)
    train_df, val_df, test_df = numeric_imputer(train_df, val_df, test_df, impute_type = "KNN", impute_neighbours=5)
    features = ['curb_weight', 'power', 'cylinder_cnt', 'omv', 'arf', 'emission_data' ,\
    'engine_cap', 'depreciation', 'mileage', 'coe', 'car_age', 'manufactured', 'road_tax', 'dereg_value']
    train_df, val_df, test_df = scalar_transform(train_df, val_df, test_df, features=features)
    train_df, val_df, test_df = feature_transform(train_df, val_df, test_df, features=features)
    save_dataset(train_df, val_df, test_df, name)

    print(train_df, val_df, test_df)


def main():
    parser = argparse.ArgumentParser(description='Data Preprocessing Script with Multiple Methods')
    parser.add_argument('--method', type=str, choices=['SimpleImputers_OutliersRemoved', 'NoSimpleImputers_OutliersRemoved', \
                                                       'NoSimpleImputers_OutliersRemoved_NoFeatureTransform', 'SimpleImputers_OutliersRemoved_NoVehCond'], default='SimpleImputers_OutliersRemoved',
                        help='Choose preprocessing method: "SimpleImputers_OutliersRemoved" or "NoSimpleImputers_OutliersRemoved"')
    args = parser.parse_args()


    test_df = pd.read_csv(r"./dataset/test.csv")
    test_df['model'] = test_df['model'].apply(lambda x:x.replace('(', ''))
    test_df['reg_date_dt'] = test_df['reg_date'].apply(lambda x: datetime.strptime(x, "%d-%b-%Y"))
    test_df['reg_date_year'] = test_df['reg_date_dt'].apply(lambda x:x.year)
    test_df['reg_date_month'] = test_df['reg_date_dt'].apply(lambda x:x.month)

    orig_df = pd.read_csv(r"./dataset/train.csv")

    # clean model
    orig_df['model'] = orig_df['model'].apply(lambda x:x.replace('(', ''))
    orig_df['reg_date_dt'] = orig_df['reg_date'].apply(lambda x: datetime.strptime(x, "%d-%b-%Y"))
    orig_df['reg_date_year'] = orig_df['reg_date_dt'].apply(lambda x:x.year)
    orig_df['reg_date_month'] = orig_df['reg_date_dt'].apply(lambda x:x.month)

    if args.method == 'SimpleImputers_OutliersRemoved':
        print("Running preprocessing method SimpleImputers_OutliersRemoved")
        orig_df = SimpleImputers_OutliersRemoved(orig_df, test_df, args.method)

    elif args.method == 'NoSimpleImputers_OutliersRemoved':
        print("Running preprocessing NoSimpleImputers_OutliersRemoved")
        orig_df = NoSimpleImputers_OutliersRemoved(orig_df, test_df, args.method)

    elif args.method == 'SimpleImputers_OutliersRemoved_NoVehCond':
        print("Running preprocessing SimpleImputers_OutliersRemoved_NoVehCond")
        orig_df = SimpleImputers_OutliersRemoved_NoVehCond(orig_df, test_df, args.method)

    elif args.method == 'NoSimpleImputers_OutliersRemoved_NoFeatureTransform':
        print("Running preprocessing NoSimpleImputers_OutliersRemoved_NoFeatureTransform")
        orig_df = NoSimpleImputers_OutliersRemoved_NoFeatureTransform(orig_df, test_df, args.method)
    else:
        print("Invalid method selected. Please choose 'SimpleImputers_OutliersRemoved' or 'NoSimpleImputers_OutliersRemoved'.")
        return

if __name__ == "__main__":
    main()



