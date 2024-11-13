import pandas as pd
# import xgboost as xgb
# import lightgbm as lgb
from utils.constants import *
from sklearn.model_selection import KFold, StratifiedShuffleSplit,ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn import ensemble, svm, tree, linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler, PowerTransformer
from sklearn.metrics import root_mean_squared_error,root_mean_squared_log_error, r2_score, median_absolute_error,mean_absolute_percentage_error,mean_absolute_error, max_error
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import pickle

# REF: SCALERS -- https://medium.com/@daython3/scaling-your-data-using-scikit-learn-scalers-3d4b584107d7

## Flags
# raw_data=False
# impute_type = "KNN"
# impute_strategy = "median" # mean, median, most_frequent, constant, Callable 
# impute_neighbours = 5
# random_state = 0
# impute_max_iter= 10
# scale_flag = False
# scaler_type = "minmax"
# model_type = "xgb"
save_model = True
full_search = True
features = ['curb_weight', 'power', 'cylinder_cnt', 'omv', 'dereg_value', 'car_age', 'depreciation', 'arf','coe', 'road_tax',
       'engine_cap',  'mileage', 'no_of_owners', 
            # 'cond_vehicle_type_0',	'cond_vehicle_type_1',	'cond_vehicle_type_2',	
            # 'cond_vehicle_type_3',	'cond_vehicle_type_4',
            'type_of_vehicle_bus/mini bus', 'type_of_vehicle_hatchback',
       'type_of_vehicle_luxury sedan', 'type_of_vehicle_mid-sized sedan',
       'type_of_vehicle_mpv', 'type_of_vehicle_others',
       'type_of_vehicle_sports car', 'type_of_vehicle_stationwagon',
       'type_of_vehicle_suv', 'type_of_vehicle_truck',
        'coe car', 'parf car', 
       'rare & exotic', 'emission_data']
# CV_FOLDS = 5
# # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}  , epsilon = 0.1 ,C = 10
# svr_kernel = 'rbf'
name = 'SimpleImputers_OutliersRemoved_NoVehCond'
train_dir = rf'./processed_dataset/train_{name}.csv'
val_dir = rf'./processed_dataset/val_{name}.csv'

train_df = pd.read_csv(train_dir)
val_df = pd.read_csv(val_dir)


X_train = train_df[features].values
y_train = train_df['price'].values
X_val = val_df[features].values
y_val = val_df['price'].values

### Grid Search ###
param_grid = {
    'learning_rate': [ 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.1, 0.5, 1],  # L1 regularization
    'reg_lambda': [1, 1.5, 2, 3],    # L2 regularization
}

# Initialize XGBRegressor
xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

if full_search:
    random_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring=make_scorer(root_mean_squared_error, greater_is_better=False),  # Negative RMSE for maximizing,
        n_jobs=14,
        cv=5,
        verbose=1,
    )
else:
    # Set up the randomized search with RMSE as the scoring metric
    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_grid,
        n_iter=500,  # Number of parameter settings to sample
        scoring=make_scorer(root_mean_squared_error, greater_is_better=False),  # Negative RMSE for maximizing
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

# Fit the random search model
random_search.fit(X_train, y_train)

# Get the best model and parameters
best_xgb = random_search.best_estimator_
print("Best parameters found: ", random_search.best_params_)

########################################################################
# Fit the model

# [TODO] We can do this on all as well
# scores = cross_val_score(best_xgb, X_train, y_train, cv=CV_FOLDS, scoring="neg_root_mean_squared_error").mean()

# Evaluate on the hold-out set
# Predict on the training and holdout sets
y_train_pred = best_xgb.predict(X_train)
y_val_pred = best_xgb.predict(X_val)

######## Metrics that we are tracking ########
best_params = random_search.best_params_

train_rmse = root_mean_squared_error(y_train, y_train_pred)
holdout_rmse = root_mean_squared_error(y_val, y_val_pred)
holdout_r2 = r2_score(y_val, y_val_pred)
holdout_mae = mean_absolute_error(y_val, y_val_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
holdout_max_error = max_error(y_val, y_val_pred)
train_max_error= max_error(y_train, y_train_pred)
holdout_mape = mean_absolute_percentage_error(y_val, y_val_pred)

print("best_params: " , best_params)
# print("scores: %f" % scores)
print("train_rmse: %f" % train_rmse)
print("train_mae: %f" % train_mae)
print("train_max_error: %f" % train_max_error)

print("holdout_rmse: %f" % holdout_rmse)
print("holdout_r2: %f" % holdout_r2)
print("holdout_mae: %f" % holdout_mae)
print("holdout_max_error: %f" % holdout_max_error)
print("holdout_mape: %f" % holdout_mape)

## saving model
if save_model:
    # pickle.dump(best_xgb, open(r"./model_assets/xgb.pkl", "wb"))
    pickle.dump(best_xgb, open(rf"C:/Users/kan_h/Desktop/Kan Hon/Admin/NUS MComp/AY2425_Sem1/CS5228/cs5228-project/CS5228_project/model_assets/xgb_{name}.pkl", "wb"))
