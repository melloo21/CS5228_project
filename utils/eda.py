import pandas 
import sklearn 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math


train = "./cs-5228-2410-final-project/train.csv"

dateFormat = "%d-%b-%Y"

def calculate_date_diff(date: str):
    date1 = datetime.strptime(date, dateFormat)
    today = datetime.today()
    
    total_months = (today.year - date1.year) * 12 + (today.month - date1.month)
    #print(f"total months: {total_months}")

    return total_months

def add_10_years(date: str):
    date1 = datetime.strptime(date, dateFormat)
    try:
        new_date = date1.replace(year=date1.year + 10)
    except ValueError:
        # This handles the case where adding 10 years results in an invalid date (like leap years)
        # For example, February 29th on a leap year
        new_date = date1.replace(year=date1.year + 10, day=date1.day - 1)

    return new_date.strftime(dateFormat)


def main():
    print(f"Reading csv...")
    trainData = pandas.read_csv(train)

    print(f"Columns: {trainData.columns}")
    print(f"Shape: {trainData.shape}")

    manufactured = trainData.loc[:, "manufactured"]
    #print(f"Manufactured: {manufactured}")
    #print(f"{manufactured.isna().any()}")
    rows = trainData[trainData["manufactured"].isna()]
    #print(f"manufactured na: {rows}")

    regdate = trainData.loc[:, "reg_date"]
    rows = trainData[trainData["reg_date"].isna()]
    #print(f"reg_date na: {rows}")


    owners = trainData.loc[:, "no_of_owners"]
    rows = trainData[trainData["no_of_owners"].isna()].index
    print(f"no owners na: {rows}")

    for r in rows:
        row = trainData.iloc[r]

        #Impute number of owners based on reg_date
        regDate = row["reg_date"]
        months = calculate_date_diff(regDate)

        # We estimate the number of owners based on every 3 years or 36 months
        estimated_owners = math.ceil(months / 36)
        trainData.loc[r, "no_of_owners"] = estimated_owners

        
        
    rows = trainData[trainData["mileage"].isna()].index
    for r in rows:
        row = trainData.iloc[r]
        #print(f"row: {row}")
        regDate = row["reg_date"]
        months = calculate_date_diff(regDate)

        #impute mileage based on reg_date 
        #we estimate that the average mileage of a car is 8000 miles per year
        estimated_mileage = months / 12 * 8000
        trainData.loc[r, "mileage"] = estimated_mileage
        
    #Missing depreciation and missing dereg_value 
    #rows = trainData[trainData["depreciation"].isna() & trainData["dereg_value"].isna()].index
    #print(f"missing depre and dereg_value: {len(rows)} rows")

    #rows = trainData[trainData["depreciation"].notna() & trainData["dereg_value"].isna()].index
    
    rows = trainData[trainData["lifespan"].isna()].index
    print(f"len of lifespan na: {len(rows)}")
    for r in rows:
        row = trainData.iloc[r]
        regDate = row["reg_date"]
        lifespan = add_10_years(regDate)
        trainData.loc[r, "lifespan"] = lifespan


    rows = trainData[trainData["lifespan"].isna()].index
    print(f"len of lifespan na: {len(rows)}")
    # regDate = trainData.loc[:, "original_reg_date"]
    # print(f"regDate: {regDate}")

    # owners = trainData.loc[:, "no_of_owners"]
    # print(f"owners: {owners}")


    # typeOfVehicle = trainData.loc[:, "type_of_vehicle"].unique()
    # print(f"Type of vehicle: {typeOfVehicle}")

    #Find the type of vehicle with others 
    # others = typeOfVehicle.loc[:, "others"]


    # others = trainData.loc[trainData["type_of_vehicle"]==""]
    # print(f"others: {others}")
    #print(f"others: {others.iloc[0, :]}")

    # t = trainData.loc[ (trainData["model"] == "minor") & (trainData["make"] == "morris")]
    # print("hello")
    # print(f"t: {t}")

    # others = trainData.loc[(trainData["type_of_vehicle"]=="others") & (trainData["make"]=="morris") & (trainData["model"]=="minor")]
    #print(f"others: {others}")
    # others1 = others.iloc[0, :] 
    # print(f"others: {others1}")

    #Find all rows with a make and model of morris and minor

    # regDate = trainData.loc[:, "reg_date"].unique()
    # regDate = trainData.loc[trainData["reg_date"]==""]
    #print(f"regDate: {regDate}")

    # catTokens = {}
    # category = trainData.loc[:, "category"]
    # for c in category:
    #     tokens = c.split(",")
    #     #print(f"tokens: {tokens}")

    #     for t in tokens:
    #         if t not in catTokens:
    #             catTokens.update({t: 1})
    #         else:
    #             catTokens[t] += 1
    
    # print(f"tokens: {catTokens}") 
        
    # category = trainData.loc[trainData["category"] == ""]
    #print(f"category: {category}")

    # transmission = trainData.loc[:, "transmission"].unique()
    # print(f"transmission: {transmission}")

    # curbWeight = trainData.loc[:, "curb_weight"].unique()
    # print(f"curb weight: {curbWeight}")

    # power = trainData.loc[:, "power"].unique()
    # print(f"power: {power}")

    # typeofveh = trainData.loc[:, "type_of_vehicle"]
    # price = trainData.loc[:, "price"]

    # plt.plot(price, typeofveh, 'o')
    # plt.title("test")
    # plt.show()



if __name__ == "__main__":
    main()