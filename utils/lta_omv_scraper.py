from bs4 import BeautifulSoup
import numpy as np
import requests
import time
import pandas as pd

def parse_lta_omv_response(response):

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all year sections (div elements with class 'yearList')
    years = soup.find_all('div', class_='yearList')
    # Initialize a list to store the extracted data
    car_data = []

    # Loop through each year and extract the months, brands, models, and values
    for year in years:
        year_id = year['id'].split('_')[1]  # Extract year (e.g., "2003" from "year_2003")
        
        # Find all months for the current year
        months = year.find_all('div', class_='monthList')
        
        for month in months:
            month_id = month['id'].split('_')[1]  # Extract month (e.g., "1" from "month_1")
            
            # Find all brands for the current month
            brands = month.find_all('div', class_='brandList')
            
            for brand in brands:
                # Get the brand (make) name
                make = brand.find('input')['value']
                
                # Find all models and values for the current brand
            # Find all models and values for the current brand dynamically
                models = brand.find_all('input', id=lambda x: x and x.startswith('modelValue'))
                values = brand.find_all('input', id=lambda x: x and x.startswith('averageOmv'))
                
                for model, value in zip(models, values):
                    model_name = model['value']
                    omv_value = value['value']
                    # Append a tuple with the year, month, make, model, and value to the list
                    car_data.append((year_id, month_id, make, model_name, omv_value))
    return car_data

def get_lta_omv_data(start_year, end_year):
    year_ls = np.arange(start_year, end_year)
    result_ls = []
    for year in year_ls:
        # The URL you want to send the GET request to
        url = f"https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/upfront-vehicle-costs/open-market-value--omv-.html?year={str(year)}"
        
        # Sending a GET request
        response = requests.get(url)
        time.sleep(1)
        result = parse_lta_omv_response(response)
        result_ls.extend(result)
    return result_ls

def preprocess_lta_omv_data(result_ls):
    df_lta_car_data = pd.DataFrame(result_ls, columns=['year','month', 'make', 'model', 'omv'])
    df_lta_car_data[['model_split', 'engine_cap']] = df_lta_car_data['model'].str.split('--', n=1, expand=True)
    df_lta_car_data['omv_clean'] = df_lta_car_data['omv'].apply(lambda x:x.replace('S$', ''))
    df_lta_car_data['omv_clean'] = df_lta_car_data['omv_clean'].apply(lambda x:int(x.replace(',', '')))
    df_lta_car_data['model_split'] = df_lta_car_data['model_split'].apply(lambda x:x.lower())
    df_lta_car_data['make_clean'] = df_lta_car_data['make'].apply(lambda x:x.lower())
    df_lta_car_data['make_clean'] = df_lta_car_data['make_clean'].apply(lambda x:x.replace('.', ''))
    df_lta_car_data['make_clean'] = df_lta_car_data['make_clean'].apply(lambda x:x.replace('rolls royce', 'rolls-royce'))
    df_lta_car_data['make_clean'] = df_lta_car_data['make_clean'].apply(lambda x:x.replace('mercedes benz', 'mercedes-benz'))
    df_lta_car_data['year'] = df_lta_car_data['year'].apply(lambda x:int(x))

    return df_lta_car_data