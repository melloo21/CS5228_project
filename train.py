import pandas as pd
from utils.constants import *
from sklearn.model_selection import KFold, StratifiedShuffleSplit,ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn import ensemble, svm, tree, linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler, PowerTransformer
from sklearn.metrics import root_mean_squared_error,root_mean_squared_log_error, r2_score, median_absolute_error,mean_absolute_percentage_error,mean_absolute_error, max_error

# REF: SCALERS -- https://medium.com/@daython3/scaling-your-data-using-scikit-learn-scalers-3d4b584107d7

## Flags
raw_data=True
impute_type = "KNN"
impute_strategy = "median" # mean, median, most_frequent, constant, Callable 
impute_neighbours = 30
random_state = 0
impute_max_iter= 10
scale_flag = True
scaler_type = "minmax"
model_type = "lr"
features = [ 'power', 'dereg_value', 'depreciation', 'arf','coe', 'mileage']
CV_FOLDS = 5
# {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}  , epsilon = 0.1 ,C = 10
svr_kernel = 'rbf'
# Fold types
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# Options (Can add more)
scaler_choice = {
    "minmax":  MinMaxScaler(),
    "robust": RobustScaler(),
    "standard": StandardScaler()
}

impute_choice = {
    "simple" : SimpleImputer(strategy=impute_strategy),
    "KNN" : KNNImputer(n_neighbors=impute_neighbours),
    "iterative": IterativeImputer(max_iter=impute_max_iter, random_state=random_state)
}

model_choice  = {
            "decision_tree": tree.DecisionTreeRegressor(),
            "random_forest": ensemble.RandomForestRegressor(),
            "lr": linear_model.LinearRegression(),
            "knn": KNeighborsRegressor(),
            'gb': ensemble.GradientBoostingRegressor(),
            "svr": svm.SVR(kernel=svr_kernel )
        }

# Do switching
if raw_data:
    test_size=0.2
    random_state=42
    shuffle=True
    orig_df = pd.read_csv(r"./dataset/train.csv")
    ## Split into train val split
    train_df, val_df = train_test_split(orig_df, test_size=test_size, random_state=random_state, shuffle=shuffle)

else:
    # Reading the dataset
    train_df = pd.read_csv(f"{local_path}/{folder}/{train_dataset}")
    val_df = pd.read_csv(f"{local_path}/{folder}/{val_dataset}")
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

train_df[features] = imputer.fit_transform(train_df[features])
val_df[features] = imputer.transform(val_df[features])

if scale_flag:
    # Fit and transform the numerical columns
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])

X_train = train_df[features]
y_train = train_df['price']

X_val = val_df[features]
y_val = val_df['price']

# Fit the model
model.fit(X_train,y_train)

# [TODO] We can do this on all as well
scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="neg_root_mean_squared_error").mean()

# Evaluate on the hold-out set
y_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

# Metrics that we are tracking
train_rmse = root_mean_squared_error(y_train, y_train_pred)
holdout_rmse = root_mean_squared_error(y_val, y_pred)
holdout_r2 = r2_score(y_val, y_pred)
holdout_mae = mean_absolute_error(y_val, y_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
holdout_max_error = max_error(y_val, y_pred)
train_max_error= max_error(y_train, y_train_pred)
holdout_mape = mean_absolute_percentage_error(y_val, y_pred)

print("scores: %f" % scores)
print("train_rmse: %f" % train_rmse)
print("train_mae: %f" % train_mae)
print("train_max_error: %f" % train_max_error)

print("holdout_rmse: %f" % holdout_rmse)
print("holdout_r2: %f" % holdout_r2)
print("holdout_mae: %f" % holdout_mae)
print("holdout_max_error: %f" % holdout_max_error)
print("holdout_mape: %f" % holdout_mape)



