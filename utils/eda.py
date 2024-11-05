import pandas as pd
import sklearn 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


train = "../cs-5228-2410-final-project/train.csv"

dateFormat = "%d-%b-%Y"

features = ["reg_date", "no_of_owners", "mileage", "lifespan", "manufactured"]

#Attempts to impute X months per owner
class OwnerImputer():
    mean = None
    median = None
    mode = None
    
    def __init__(self):
        pass
    
    ##Learns the parameters (like mean and median)
    def fit(self, input):
        #Here we calculate the number of owners based on the number of months that have passed since reg_date
        rows = input[input["no_of_owners"].notna()].index
        
        #Initialize to 0
        accumulated = []
        
        for r in rows:
            row = input.iloc[r]
            reg_date = row["reg_date"]
            owners = row["no_of_owners"]
            months = calculateDateDiff(reg_date)
            months_per_owner = months / owners
            accumulated.append(months_per_owner)
        
        self.mean = np.mean(accumulated)
        self.median = np.median(accumulated)
        mode_result = stats.mode(accumulated, axis = None, keepdims=False)
        self.mode = mode_result.mode
        
        
        print(f"[Number of months per owner]  ->  [mean = {self.mean}, median = {self.median}, mode = {self.mode}]")
    
    #Applies the mean as the imputed values for the na values
    def transform(self, input, strategy="mean"):
        if strategy != "mean" and strategy != "median" and strategy != "mode":
            raise ValueError("transform() - strategy only accepts mean, median and mode")
        
        rows = input[input["no_of_owners"].isna()].index
        for r in rows:
            #Get the number of regdate of the entry
            row = input.iloc[r]
            reg_date = row["reg_date"]
            months = calculateDateDiff(reg_date)
            
            if strategy == "mean":
                input.loc[r, "no_of_owners"] = months / self.mean
            elif strategy == "median":
                input.loc[r, "no_of_owners"] = months / self.median
            elif strategy == "mode":
                input.loc[r, "no_of_owners"] = months / self.mode
    
    def fit_transform(self, input, strategy="mean"):
        self.fit(input)
        self.transform(input, strategy)
        
class mileageImputer():
    mean = None
    median = None
    mode = None 
    
    def __init__(self):
        pass
    
    #Calculates the value of amount of mileage per month of use 
    def fit(self, input):
        rows = input[input["mileage"].notna()].index
        
        accumulated = []
        
        for r in rows:
            row = input.iloc[r]
            reg_date = row["reg_date"]
            mileage = row["mileage"]
            months = calculateDateDiff(reg_date)
            mileage_per_month = mileage / months
            accumulated.append(mileage_per_month)
        
        self.mean = np.mean(accumulated)
        self.median = np.median(accumulated)
        mode_result = stats.mode(accumulated, axis=None, keepdims=False)
        self.mode = mode_result.mode
        
        print(f"[Miles per month]  ->  [mean = {self.mean}, median = {self.median}, mode = {self.mode}]")
    
    def transform(self, input, strategy="mean"):
        if strategy != "mean" and strategy != "median" and strategy != "mode":
            raise ValueError("transform() - strategy only accepts mean, median and mode")

        rows = input[input["mileage"].isna()].index

        for r in rows:
            row = input.iloc[r]
            reg_date = row["reg_date"]
            months = calculateDateDiff(reg_date)
            
            if strategy == "mean":
                input.loc[r, "mileage"] = months * self.mean
            elif strategy == "median":
                input.loc[r, "mileage"] = months * self.median
            elif strategy == "mode":
                input.loc[r, "mileage"] = months * self.mode
                
    def fit_transform(self, input, strategy="mean"):
        self.fit(input)
        self.transform(input, strategy)

def main():
    print(f"Reading csv...")
    input = pd.read_csv(train)
    
    #Retrieve only the features that we are interested in 
    input = input[features]

    #Split the input
    print("Splitting the input for train and val...")
    X_train, X_val = train_test_split(input, test_size=0.2, random_state=5228)
    
    #Reset the index and drops the original index
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    
    print(X_train.head())
    
    print(f"Performing imputation...")
    #imput = imputeNoOwners(input)
    imputer = OwnerImputer()
    imputer.fit(X_train)
    imputer.transform(X_train, strategy="mode")
    imputer.transform(X_val, strategy="mode")

    #imputeMileage(input)
    imputer = mileageImputer()
    imputer.fit(X_train)
    imputer.transform(X_train, strategy="mean")
    imputer.transform(X_val, strategy="mean")
        
    imputeLifespan(X_train)
    imputeLifespan(X_val)
        
    imputeManufactured(X_train)
    imputeManufactured(X_val)
    
    #Dates cannot be fitted into the regression model we need to extract the useful feature of the dates 
    #For our context, I'm just using the year for now 
    filterYearsRegDateLifespan(X_train)
    filterYearsRegDateLifespan(X_val)
    
    X_train.to_pickle("train.pkl")
    X_val.to_pickle("val.pkl")
    
    #Imputation End
    
    #Scale the data (Useful for KNN and Linear Regression models?)
    scaler = MinMaxScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    
    X_train_scaled.to_pickle("train_scaled.pkl")
    X_val_scaled.to_pickle("val_scaled.pkl")
    
def filterYearsRegDateLifespan(input):
    for idx, row in input.iterrows():
        #print(row)
        lifespan = row["lifespan"]
        #print(f"lifespan: {lifespan}")
        reg_date = row["reg_date"]
        
        #print(f"lifespan: {lifespan}, reg_date: {reg_date}")
        
        input.loc[idx, "lifespan"] = extractYear(lifespan)
        input.loc[idx, "reg_date"] = extractYear(reg_date)

def imputeManufactured(input):
    rows = input[input["manufactured"].isna()].index
    for r in rows:
        row = input.iloc[r]
        regDate = row["reg_date"][-4:]
        #print(f"reg date: {regDate}")
        input.loc[r, "manufactured"] = float(regDate)
        

def imputeLifespan(input):
    rows = input[input["lifespan"].isna()].index
    #print(f"len of lifespan na: {len(rows)}")
    for r in rows:
        row = input.iloc[r]
        regDate = row["reg_date"]
        lifespan = add10Years(regDate)
        input.loc[r, "lifespan"] = lifespan

def imputeMileage(input):
    rows = input[input["mileage"].isna()].index
    for r in rows:
        row = input.iloc[r]
        #print(f"row: {row}")
        regDate = row["reg_date"]
        months = calculateDateDiff(regDate)

        #impute mileage based on reg_date 
        #we estimate that the average mileage of a car is 8000 miles per year
        estimated_mileage = months / 12 * 8000
        input.loc[r, "mileage"] = estimated_mileage
        

def imputeNoOwners(input):
    rows = input[input["no_of_owners"].isna()].index
    for r in rows:
        row = input.iloc[r]

        #Impute number of owners based on reg_date
        regDate = row["reg_date"]
        months = calculateDateDiff(regDate)

        # We estimate the number of owners based on every 3 years or 36 months
        estimated_owners = math.ceil(months / 36)
        input.loc[r, "no_of_owners"] = estimated_owners
    
    # #Fit the model
    # print("Training the linear model...")
    # y_pred, y_test = fitLinearModel(X_train, X_test, y_train, y_test)
    
    # #Get the stats
    # print("Getting stats...")
    # getStats(y_test, y_pred)
    
    
    # #Fit another model 
    # y_pred = fitDecisionTreeModel(X_train, X_test, y_train)
    
    # #Get the stats
    # print("Getting stats...")
    # getStats(y_test, y_pred)
    
def calculateDateDiff(date: str):
    date1 = datetime.strptime(date, dateFormat)
    today = datetime.today()
    
    total_months = (today.year - date1.year) * 12 + (today.month - date1.month)
    #print(f"total months: {total_months}")

    return total_months

def add10Years(date: str):
    date1 = datetime.strptime(date, dateFormat)
    try:
        new_date = date1.replace(year=date1.year + 10)
    except ValueError:
        # This handles the case where adding 10 years results in an invalid date (like leap years)
        # For example, February 29th on a leap year
        new_date = date1.replace(year=date1.year + 10, day=date1.day - 1)

    return new_date.strftime(dateFormat)

#This function expects the date in the format dd-mmm-yyyy
def extractYear(date: str):
    return date[-4:]


#Checks for nan in the data
def checkNan(data):
    print("Checking for nan in the data...")
    print(data.isnull().sum())

#Splits the input into train and test and returns the split data
def splitInput(input):
    #Extract y
    y = input["price"]
    print(f"y shape: {y.shape}") #25000 samples

    #Get the features that we are interested in (added in coe since all values are present)
    features = ["reg_date", "no_of_owners", "mileage", "lifespan", "manufactured", "coe"]

    #Extract X 
    X = input[features]
    print(f"X shape: {X.shape}") #25000 samples x 6 features
    
    print(f"X : {X}")
    
    #Test for nan in the data
    checkNan(X)

    #Let's try to split the data into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5228)

    return X_train, X_test, y_train, y_test


#Calculates the stats and print them out
def getStats(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print the metrics
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R² Score: {r2:.2f}')
    
    
#Here we use Linear Regression model 
def fitLinearModel(X_train, X_test, y_train):
    #Now we scale the data 
    scaler = MinMaxScaler()
    
    X_train = scaler.fit_transform(X_train)
    
    X_test = scaler.transform(X_test)
    y_train
    
    
    #We try a linear model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #Test the model 
    y_pred = model.predict(X_test)
    
    return y_pred

def fitDecisionTreeModel(X_train, X_test, y_train):
    #We try a decisionTreeModel 
    
    #We try to get the best hyperparamter
    # Define the hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': [None, 'sqrt', 'log2']
    }
    
    model = DecisionTreeRegressor(random_state=5228)
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    
    #model.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)
    
    # Train a new model using the best parameters
    best_regressor = grid_search.best_estimator_
    
    #Test the model 
    #y_pred = model.predict(X_test)
    y_pred = best_regressor.predict(X_test)
    
    return y_pred

if __name__ == "__main__":
    main()