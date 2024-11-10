import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from scipy import stats

train = "../cs-5228-2410-final-project/train.csv"

#min and max of each vehicle type 
curb_weights= {
    "Commercial": [37.0, 190.0],
    "Hatchback" : [41.0, 310.0],
    "Luxury Sedan": [80.0, 465.0],
    "MPV": [69.0, 209.0],
    "SUV": [60, 490.0], 
    "Sedan": [57.0, 310.0],
    "Sports": [47.0, 588.0],
    "Stationwagon": [66.0, 418.0]
}

mapping = {
    "suv": 'SUV',
     'mpv': 'MPV',
    "luxury sedan" : 'Luxury Sedan',
    "mid-sized sedan": "Sedan",
     'sports car' : 'Sports',
     'truck' : 'Commercial',
     'hatchback': 'Hatchback',
     'stationwagon' : 'Stationwagon',
     'bus/mini bus':'Commercial',
     'van':'Commercial',
     'others':'Commercial'
}

def curb_weight_outlier(data):
    print("finding curb weight outliers")
    outliers = []
    # for idx, row in data.iterrows():
    #     veh_type = mapping[row['type_of_vehicle']]
    #     min = curb_weights[veh_type][0]
    #     max = curb_weights[veh_type][1]
        
    #     if row['curb_weight'] < min or row['curb_weight'] > max:
    #         data.loc[idx, "curb_weight"] = np.nan
    #         outliers.append(idx)
    
    for idx, row in data.iterrows(): 
        if row['curb_weight'] > 46000 or row['curb_weight'] < 440:
            data.loc[idx, "curb_weight"] = np.nan
            outliers.append(idx)
            #print(row)
    
    print(len(outliers))
    
def power_outlier(data):
    print("finding power outliers")
    outliers = []
    for idx, row in data.iterrows():
        if row['power'] == 0:
            data.loc[idx, "power"] = np.nan
            outliers.append(idx)
        #print(data.iloc[idx])
        
    print(len(outliers))
    
def enginecap_outlier(data):
    print("finding engine cap outliers")
    outliers = []
    for idx, row in data.iterrows():
        if row['engine_cap'] == 0:
            data.loc[idx, "engine_cap"] = np.nan
            outliers.append(idx)
        #print(data.iloc[idx])
    
    print(len(outliers))


print(f"Reading csv...")
input = pd.read_csv(train)

curb_weight_outlier(input)
power_outlier(input)
enginecap_outlier(input)

