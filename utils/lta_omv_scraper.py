from bs4 import BeautifulSoup

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