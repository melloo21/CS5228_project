import pandas as pd

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn import ensemble, svm, tree, linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer, IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler, PowerTransformer
from sklearn.metrics import neg_root_mean_squared_error, r2, neg_mean_absolute_error,max_error

# REF: SCALERS -- https://medium.com/@daython3/scaling-your-data-using-scikit-learn-scalers-3d4b584107d7

# Reading the dataset
local_path = "/Users/melloo21/Desktop/NUS Items/CS5228/Project/CS5228_project/"
train_name = "train_df_imputation_v1"
val_name = "val_df_imputation_v1"

train_df = pd.read_csv(f"{local_path}/processed_dataset/{train_name}")
val_df = pd.read_csv(f"{local_path}/processed_dataset/{val_name}")

## Flags
impute_type = "simple"
impute_strategy = "median" # mean, median, most_frequent, constant, Callable 
impute_neighbours = 5
randome_state = 0
impute_max_iter= 10
scaler_type = "minmax"
model_type = "decision_tree"
features = ['curb_weight', 'power', 'cylinder_cnt', 'omv', 'dereg_value', 'car_age', 'depreciation', 'arf','coe', 'road_tax',
       'engine_cap', 'depreciation', 'mileage', 'no_of_owners','type_of_vehicle_bus/mini bus', 'type_of_vehicle_hatchback',
       'type_of_vehicle_luxury sedan', 'type_of_vehicle_mid-sized sedan',
       'type_of_vehicle_mpv', 'type_of_vehicle_others',
       'type_of_vehicle_sports car', 'type_of_vehicle_stationwagon',
       'type_of_vehicle_suv', 'type_of_vehicle_truck', 'type_of_vehicle_van']

# Options (Can add more)
scaler_choice = {
    "minmax":  MinMaxScaler(),
    "robust": RobustScaler(),
    "standard": StandardScaler()
}

impute_choice = {
    "simple" : SimpleImputer(strategy=impute_strategy),
    "KNN" : KNNImputer(n_neighbors=impute_neighbours),
    "iterative": IterativeImputer(max_iter=impute_max_iter, random_state=randome_state)
}

model_choice  = {
            "decision_tree": tree.DecisionTreeRegressor(),
            "random_forest": ensemble.RandomForestRegressor(),
            "lr": linear_model.LinearRegression(),
            "knn": KNeighborsRegressor(),
            'gb': ensemble.GradientBoostingRegressor(),
            "svr": svm.SVR()
        }

# Get other scores
# [TODO] :: Should we add a missing indicator as well?
# from sklearn.impute import MissingIndicator
# indicator = MissingIndicator()
# missing_mask = indicator.fit_transform(X)

# Selection
imputer = impute_choice[impute_type]
scaler = scaler_choice[scaler_type]
model = model_choice[model_type]

# Average cv score -- limitation is that it does not give best model
# cv = cross_validate(model, X, y, cv=5, return_train_score=True)
# Prepare data

X_train[features] = imputer.fit_transform(X_train[features])
val_df[features] = imputer.transform(val_df[features])

# Fit and transform the numerical columns
X_train[features] = scaler.fit_transform(X_train[features])
val_df[features] = scaler.transform(val_df[features])

X_train = train_df[features]
y_train = train_df['price']

X_val = val_df[features]
y_val = val_df['price']

# Fit the model
model.fit(X_train,y_train)

# [TODO] We can do this on all as well
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")

# Evaluate on the hold-out set
y_pred = model.predict(X_val)

# Metrics that we are tracking
holdout_rmse = neg_root_mean_squared_error(y_val, y_pred)
holdout_r2 = r2_score(y_val, y_pred)
holdout_mae = mean_absolute_error(y_val, y_pred)
holdout_max_error = max_error(y_val, y_pred)
print("CV Score ", scores.mean())
print("\nHold-out set scores:")
print("RMSE:", holdout_rmse)
print("R^2:", holdout_r2)
print("MAE:", holdout_mae)
print("Max Error:", holdout_max_error)


