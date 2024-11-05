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
from sklearn.impute import KNNImputer
from scipy import stats


train = "../cs-5228-2410-final-project/train.csv"

dateFormat = "%d-%b-%Y"

features = ["reg_date", "no_of_owners", "mileage", "lifespan", "manufactured"]

keyWords = ["parf car", "coe car"]



# If we can get the info of whether this vehicle is parf car or coe car, we can calulate the exact date when this car was previously sold
# if parf car (0 to 10 years old) -> reg_date + 10 years - years_left (must be taxi or car) 
# if coe car (10 to 20 years old) -> reg_date + 20 years - years_left

#Imputes parf rebate to get the dereg value
class DeregValImputer():
    mean = None
    median = None
    mode = None
    
    def __init__(self):
        pass
    
    def custom_fit_transform(self, df):
        proc_df = df.copy().reset_index(drop=True)
        rows = proc_df[proc_df["dereg_value"].notna()].index
        #rows = proc_df[proc_df["coe_age_left"].notna()].index
        
        for r in rows:
            row = proc_df.iloc[r]
            coe_age_left = row["coe_age_left"]
            coe = row["coe"]
            arf = row["arf"]
            reg_date = row["reg_date"]
            
            parf_rebate = get_parf_rebate(coe_age_left, arf, reg_date)
            coe_rebate = coe * coe_age_left / 120
            dereg_value = parf_rebate + coe_rebate
                
            proc_df.loc[r, "dereg_value"] = dereg_value
            # proc_df.loc[r, "dereg_value_imputed"] = dereg_value
        return proc_df
            
    def fit(self, df):
        proc_df = df.copy().reset_index(drop=True)
        #Here we calculate the dereg_value 
        rows = proc_df[proc_df["dereg_value"].notna()].index
        
        #Initialize to 0
        accumulated = []
        
        for r in rows:
            row = proc_df.iloc[r]
            dereg = row["dereg_value"]
            accumulated.append(dereg)
        
        self.mean = np.mean(accumulated)
        self.median = np.median(accumulated)
        mode_result = stats.mode(accumulated, axis = None, keepdims=False)
        self.mode = mode_result.mode
        
        print(f"[dereg_value]  ->  [mean = {self.mean}, median = {self.median}, mode = {self.mode}]")
        
    def transform(self, df, strategy="mean"):
        if strategy != "mean" and strategy != "median" and strategy != "mode":
            raise ValueError("transform() - strategy only accepts mean, median and mode")
        proc_df = df.copy().reset_index(drop=True)
        rows = proc_df[proc_df["dereg_value"].isna()].index

        for r in rows:
            row = proc_df.iloc[r]
            
            if strategy == "mean":
                proc_df.loc[r, "dereg_value"] = self.mean
            elif strategy == "median":
                proc_df.loc[r, "dereg_value"] = self.median
            elif strategy == "mode":
                proc_df.loc[r, "dereg_value"] = self.mode
        return proc_df


#Calculates the amount of coe left for each vehicle
class CoeAgeImputer():
    def __init__(self):
        pass
    
    def fit_transform(self, df):
        #Here we find all rows with a valid depre and valid dereg value and valid price
        proc_df = df.copy().reset_index(drop=True)
        rows = proc_df[proc_df["reg_date"].notna()].index
        
        #Set 0 as the age for all entries
        proc_df["coe_age_left"] = 0
        
        for r in rows:
            row = proc_df.iloc[r]
            #depre = row["depreciation"] 
            coe_year = row["reg_date"][-4:]
            #age_range = row["age_range"
            
            #Let's calculate the number of months left for the coe
            try:
                months_left = (datetime.now().year - int(coe_year)) * 12
                if months_left >= 120 and months_left <= 240: #Vehicle is between 10 years and 20 years old 
                    months_left -= 120
                elif months_left > 240: #Probably vehicle is sold off already 
                    months_left = 0
                elif months_left < 120:
                    pass
                else:
                    print(f"Unexpected value from coe age imputer: {months_left}. reg_date is {coe_year} ")     
                proc_df.loc[r, "coe_age_left"] = months_left

                return proc_df

            except Exception as e:
                print(f"Exception encountered: {e}")

            
class AgeRangeproc_dfer():
    def __init__(self):
        pass
    
    def fit_transform(self, df):
        #Here we create a default column to represent coe car, parf car or none of the above 
        proc_df = df.copy().reset_index(drop=True)
        #Perhaps we can use 0 for neither, 1 for parf car, 2 for coe car 
        proc_df["age_range"] = -1 #Set all of the values to -1 by default
        
        rows = proc_df[proc_df["category"].notna()].index
        
        for r in rows:
            row = proc_df.iloc[r]
            cat = row["category"]
            
            for idx, kw in enumerate(keyWords):
                if idx == 0 and kw.lower() in cat.lower(): #For parf car
                    proc_df.loc[r, "age_range"] = 0
                    break
                elif idx == 1 and kw.lower() in cat.lower(): #For coe car
                    proc_df.loc[r, "age_range"] = 1
                    break
        return proc_df

#Attempts to impute X months per owner
class OwnerImputer():
    mean = None
    median = None
    mode = None
    
    def __init__(self):
        pass
    
    ##Learns the parameters (like mean and median)
    def fit(self, df):
        proc_df = df.copy().reset_index(drop=True)
        #Here we calculate the number of owners based on the number of months that have passed since reg_date
        rows = proc_df[proc_df["no_of_owners"].notna()].index
        print(len(rows))
        #Initialize to 0
        accumulated = []
        
        for r in rows:
            row = proc_df.iloc[r]
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
    def transform(self, df, strategy="mean"):

        if strategy != "mean" and strategy != "median" and strategy != "mode":
            raise ValueError("transform() - strategy only accepts mean, median and mode")
        proc_df = df.copy().reset_index(drop=True)
        rows = proc_df[proc_df["no_of_owners"].isna()].index

        for r in rows:
            #Get the number of regdate of the entry
            row = proc_df.iloc[r]
            reg_date = row["reg_date"]
            months = calculateDateDiff(reg_date)
            
            if strategy == "mean":
                proc_df.loc[r, "no_of_owners"] = months / self.mean
            elif strategy == "median":
                proc_df.loc[r, "no_of_owners"] = months / self.median
            elif strategy == "mode":
                proc_df.loc[r, "no_of_owners"] = months / self.mode
        
        return proc_df
    
    def fit_transform(self, df, strategy="mean"):
        self.fit(df)
        return self.transform(df, strategy)
        
class mileageImputer():
    mean = None
    median = None
    mode = None 
    
    def __init__(self):
        pass
    
    #Calculates the value of amount of mileage per month of use 
    def fit(self, df):
        proc_df = df.copy().reset_index(drop=True)
        rows = proc_df[proc_df["mileage"].notna()].index
        
        accumulated = []
        
        for r in rows:
            row = proc_df.iloc[r]
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
    
    def transform(self, df, strategy="mean"):
        if strategy != "mean" and strategy != "median" and strategy != "mode":
            raise ValueError("transform() - strategy only accepts mean, median and mode")
        proc_df = df.copy().reset_index(drop=True)
        rows = proc_df[proc_df["mileage"].isna()].index

        for r in rows:
            row = proc_df.iloc[r]
            reg_date = row["reg_date"]
            months = calculateDateDiff(reg_date)
            
            if strategy == "mean":
                proc_df.loc[r, "mileage"] = months * self.mean
            elif strategy == "median":
                proc_df.loc[r, "mileage"] = months * self.median
            elif strategy == "mode":
                proc_df.loc[r, "mileage"] = months * self.mode
        return proc_df

    def fit_transform(self, df, strategy="mean"):
        self.fit(df)
        return self.transform(df, strategy)

def main():
    print(f"Reading csv...")
    proc_df = pd.read_csv(train)
    
    #Split the proc_df
    print("Splitting the proc_df for train and val...")
    X_train, X_val = train_test_split(proc_df, test_size=0.2, random_state=5228, shuffle=True)
    X_train_ = X_train.copy()
    
    #Here we remove any rows with a missing depre, dereg or price value as we need it to calculate the age of the vehicle
    #X_train = dropRows(X_train)
    
    #Reset the index first 
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    
    
    nan_count = X_train.isna().sum()
    print(f"-------------- before nan count:\n{nan_count}")
    
    # [Run]
    imputer = AgeRangeproc_dfer()
    imputer.fit_transform(X_train)

    # [Run]
    imputer = CoeAgeImputer()
    imputer.fit_transform(X_train)    
    
    # imputer = DeregValImputer()
    # # imputer.custom_fit_transform(X_train)
    # imputer.fit(X_train)
    # imputer.transform(X_train, strategy="mean")
    
    # [Run] Try KNN imputer 
    imputer = KNNImputer(n_neighbors=5)
    X_train[['coe_age_left', 'coe', 'dereg_value']] = imputer.fit_transform(X_train[['coe_age_left', 'coe', 'dereg_value']])
    # Validation set -- fit

    #Lets try to plot here
    plt.scatter(X_train["dereg_value"], X_train_["dereg_value"])
    plt.xlabel("imputed dereg_value")
    plt.ylabel("original dereg_value")
    plt.show()
    
    # [Run]
    imputer = OwnerImputer()
    imputer.fit(X_train)
    imputer.transform(X_train, strategy="mode")
    imputer.transform(X_val, strategy="mode")
    
    plt.figure()
    plt.scatter(X_train["no_of_owners"], X_train_["no_of_owners"])
    plt.xlabel("original no_of_owners")
    plt.ylabel("imputed no_of_owners")
    plt.show()
    
    # [Run]
    imputer = mileageImputer()
    imputer.fit(X_train)
    imputer.transform(X_train, strategy="mean")
    imputer.transform(X_val, strategy="mean")
    
    plt.figure()
    plt.scatter(X_train["mileage"], X_train_["mileage"])
    plt.xlabel("original mileage")
    plt.ylabel("imputed mileage")
    plt.show()
    
    
    #Check to see if the imputer worked by tabulating a new column
    #print(f"{X_train.iloc[222]}")
    
    #Test to see if dropRows did a good job
    nan_count = X_train.isna().sum()
    print(f"-------------- after nan count:\n{nan_count}")
    
    
    # print(f"{proc_df.iloc[333]}")
    # print(f"{proc_df.iloc[2321]}")
    # Based on some prints, we can see that the respective entries are labelled correctly
    
    # #Retrieve only the features that we are interested in 
    # proc_df = proc_df[features]


    
    # #Reset the index and drops the original index
    # X_train.reset_index(drop=True, inplace=True)
    # X_val.reset_index(drop=True, inplace=True)
    
    # print(X_train.head())
    
    # print(f"Performing imputation...")
    # #imput = imputeNoOwners(proc_df)
    # imputer = OwnerImputer()
    # imputer.fit(X_train)
    # imputer.transform(X_train, strategy="mode")
    # imputer.transform(X_val, strategy="mode")

    # #imputeMileage(proc_df)
    # imputer = mileageImputer()
    # imputer.fit(X_train)
    # imputer.transform(X_train, strategy="mean")
    # imputer.transform(X_val, strategy="mean")
        
    # imputeLifespan(X_train)
    # imputeLifespan(X_val)
        
    # imputeManufactured(X_train)
    # imputeManufactured(X_val)
    
    # #Dates cannot be fitted into the regression model we need to extract the useful feature of the dates 
    # #For our context, I'm just using the year for now 
    # filterYearsRegDateLifespan(X_train)
    # filterYearsRegDateLifespan(X_val)
    
    # X_train.to_pickle("train.pkl")
    # X_val.to_pickle("val.pkl")
    
    # #Imputation End
    
    # #Scale the data (Useful for KNN and Linear Regression models?)
    # scaler = MinMaxScaler()
    
    # X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    
    # X_train_scaled.to_pickle("train_scaled.pkl")
    # X_val_scaled.to_pickle("val_scaled.pkl")
    

#This function takes in the amount of coe left in months and calculates the parf rebate
def get_parf_rebate(coe_age_left, arf, reg_date):
    #eg_year = reg_date[-4:]
    reg_date_ = datetime.strptime(reg_date, dateFormat)
    
    cutoff_date = datetime.strptime("22-feb-2023", dateFormat)
    
    age_months = 120 - coe_age_left
    age_years = age_months / 12 
    
    if reg_date_ < cutoff_date:
        #Check for the estimated age of the vehicle based on its reg_date
        if age_years < 5:
            return 0.75 * arf
        elif age_years < 6:
            return 0.7 * arf
        elif age_years < 7:
            return 0.65 * arf
        elif age_years < 8:
            return 0.6 * arf
        elif age_years < 9:
            return 0.55 * arf
        elif age_years < 10:
            return 0.5 * arf
        else:
            return 0
    else:
        if age_years < 5:
            return 0.75*arf if 0.75*arf < 60000 else 60000 
        elif age_years < 6:
            return 0.7*arf if 0.7*arf < 60000 else 60000
        elif age_years < 7:
            return 0.65*arf if 0.65*arf < 60000 else 60000
        elif age_years < 8:
            return 0.6*arf if 0.6*arf < 60000 else 60000
        elif age_years < 9:
            return 0.55*arf if 0.55*arf < 60000 else 60000
        elif age_years < 10:
            return 0.5*arf if 0.5*arf < 60000 else 60000
        else:
            return 0
        


#This function drops rows that have either missing depre, missing dereg or missing price
def dropRows(proc_df):    
    proc_df_cleaned = proc_df.dropna(subset=["depreciation", "price"])
    
    return proc_df_cleaned
    
    
def filterYearsRegDateLifespan(proc_df):
    for idx, row in proc_df.iterrows():
        #print(row)
        lifespan = row["lifespan"]
        #print(f"lifespan: {lifespan}")
        reg_date = row["reg_date"]
        
        #print(f"lifespan: {lifespan}, reg_date: {reg_date}")
        
        proc_df.loc[idx, "lifespan"] = extractYear(lifespan)
        proc_df.loc[idx, "reg_date"] = extractYear(reg_date)

def imputeManufactured(proc_df):
    rows = proc_df[proc_df["manufactured"].isna()].index
    for r in rows:
        row = proc_df.iloc[r]
        regDate = row["reg_date"][-4:]
        #print(f"reg date: {regDate}")
        proc_df.loc[r, "manufactured"] = float(regDate)
        

def imputeLifespan(proc_df):
    rows = proc_df[proc_df["lifespan"].isna()].index
    #print(f"len of lifespan na: {len(rows)}")
    for r in rows:
        row = proc_df.iloc[r]
        regDate = row["reg_date"]
        lifespan = add10Years(regDate)
        proc_df.loc[r, "lifespan"] = lifespan

def imputeMileage(proc_df):
    rows = proc_df[proc_df["mileage"].isna()].index
    for r in rows:
        row = proc_df.iloc[r]
        #print(f"row: {row}")
        regDate = row["reg_date"]
        months = calculateDateDiff(regDate)

        #impute mileage based on reg_date 
        #we estimate that the average mileage of a car is 8000 miles per year
        estimated_mileage = months / 12 * 8000
        proc_df.loc[r, "mileage"] = estimated_mileage
        

def imputeNoOwners(proc_df):
    rows = proc_df[proc_df["no_of_owners"].isna()].index
    for r in rows:
        row = proc_df.iloc[r]

        #Impute number of owners based on reg_date
        regDate = row["reg_date"]
        months = calculateDateDiff(regDate)

        # We estimate the number of owners based on every 3 years or 36 months
        estimated_owners = math.ceil(months / 36)
        proc_df.loc[r, "no_of_owners"] = estimated_owners
    
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

#Splits the proc_df into train and test and returns the split data
def splitproc_df(proc_df):
    #Extract y
    y = proc_df["price"]
    print(f"y shape: {y.shape}") #25000 samples

    #Get the features that we are interested in (added in coe since all values are present)
    features = ["reg_date", "no_of_owners", "mileage", "lifespan", "manufactured", "coe"]

    #Extract X 
    X = proc_df[features]
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