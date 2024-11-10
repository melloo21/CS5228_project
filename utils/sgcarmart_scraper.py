import re
import argparse
import difflib
import requests
import pandas as pd
from bs4 import BeautifulSoup



def get_soup_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
                    '(KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'
    }

    # Send a GET request to the webpage with headers
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check for request errors

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

def get_car_code_url(url_id):
    # URL of the webpage to scrape
    url = f'https://www.sgcarmart.com/used_cars/info.php?ID={url_id}'

    # Define headers to mimic a web browser

    soup = get_soup_from_url(url)
    # for row in table.find_all('tr'):
    mydivs = soup.find_all("tr", {"class": "row_bg1"})

    pattern = r"/new_cars/newcars_overview\.php\?CarCode=(\d+)"


    for div in mydivs:
        a_tag = div.find('a')
        if a_tag and 'href' in a_tag.attrs:
            href_data = a_tag['href']
            match = re.search(pattern, href_data)
            if match:
                car_code_id = match.group(1)
                return car_code_id
        else:
            href_data = None
            match = None
            return 'UNKNOWN'
        
def get_subcode_dict(car_code_id):
    '''
    Args: CarCode 

    Return: Dict of Specification to SubCode mapping
    '''
    if car_code_id == 'UNKNOWN':
        return dict()
    url = f'https://www.sgcarmart.com/new_cars/newcars_overview.php?CarCode={car_code_id}'

    soup = get_soup_from_url(url)
    # Find all 'tr' elements that contain subcode information
    subcode_rows = []

    spec_link_dict = dict()

    # Loop through all 'tr' elements
    for tr in soup.find_all('tr'):
        a_tag = tr.find('a', {'itemprop': 'additionalType'})
        if a_tag:
            subcode_rows.append(tr)

    # Extract subcode name, link, and spec link from each row
    for tr in subcode_rows:
        # Extract the subcode name and link
        a_tag = tr.find('a', {'itemprop': 'additionalType'})
        subcode_name = a_tag.find('span', {'itemprop': 'alternateName'}).text.strip()
        subcode_link = a_tag['href']
        
        # Extract the spec link
        last_td = tr.find_all('td')[-1]  # Get the last 'td' in the row
        spec_link_tag = last_td.find('a', text='Specs')
        spec_link = spec_link_tag['href'] if spec_link_tag else None

        if spec_link:
            spec_link_dict[subcode_name] = spec_link
    return spec_link_dict

def get_best_spec_link(car_name, spec_link_dict):
    try:
        spec_link_ls = spec_link_dict.keys()
        best_match = difflib.get_close_matches(car_name,spec_link_ls,1, cutoff=0)[0]
        return best_match, spec_link_dict[best_match]
    except Exception as e:
        # print(e)
        return 'UNKNOWN', None

def get_generic_specification_from_link(spec_link):
    # Get all specs from link
    url = f'https://www.sgcarmart.com/new_cars/{spec_link}'
    soup = get_soup_from_url(url)

    # Find the table with id 'submodel_spec'
    spec_table = soup.find('table', id='submodel_spec')

    # Initialize a dictionary to store the specifications
    specifications = {}

    # Check if the table exists
    if spec_table:
        # Iterate over all 'tr' elements in the table
        for row in spec_table.find_all('tr'):
            # Get all 'td' elements in the row
            cells = row.find_all('td')
            if len(cells) == 2:
                # Get the specification name and value
                spec_name = cells[0].get_text(strip=True)
                spec_value = cells[1].get_text(strip=True)
                specifications[spec_name] = spec_value

    return specifications

def getting_all_new_car_ref_specifications(df:pd.DataFrame):
    default_specs = {'Engine & Transmission': "unknown", 'Engine Capacity': "unknown",
    'Engine Type': "unknown", 'Fuel Type': "unknown", 'Drive Type': "unknown", 'Transmission': "unknown", 
    'Performance': "unknown", 'Power': "unknown", 'Torque': "unknown", 'Acceleration': "unknown", 'Top Speed': "unknown", 
    'Fuel Consumption': "unknown", 'CO2 Emission (LTA)': "unknown", 'Measurements': "unknown", 
    'Vehicle Type': "unknown", 'Seating Capacity': "unknown", 'Dimensions (L x W x H)': "unknown", 
    'Wheelbase': "unknown", 'Min Turning Radius': "unknown", 'Kerb Weight': "unknown", 'Fuel Tank Capacity': "unknown",
    'Boot/Cargo Capacity': "unknown", 'Suspension (Front)': "unknown", 'Suspension (Rear)': "unknown", 
    'Brakes (Front)': "unknown", 'Brakes (Rear)': "unknown", 'Rim Size': "unknown"}

    all_specs_list = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # To join the impute set and non impute
        final_dict = {"listing_id" : df.listing_id.iloc[i]}
        car_code_id = get_car_code_url(df.listing_id.iloc[i])
        subcode_dict = get_subcode_dict(car_code_id)
        best_match_name, spec_link = get_best_spec_link(df.title.iloc[i], subcode_dict)
        if spec_link:
            specs = get_generic_specification_from_link(spec_link)
            final_dict.update(specs)
        else: 
            final_dict.update(default_specs)    

def get_emission_data_from_spec_link(spec_link):
    if not spec_link:
        return 'UNKNOWN'
    
    url = f'https://www.sgcarmart.com/new_cars/{spec_link}'

    soup = get_soup_from_url(url)

        # Find the table with id 'submodel_spec'
    spec_table = soup.find('table', id='submodel_spec')

    # Initialize a dictionary to store the specifications
    specifications = {}

    # Check if the table exists
    if spec_table:
        # Iterate over all 'tr' elements in the table
        for row in spec_table.find_all('tr'):
            # Get all 'td' elements in the row
            cells = row.find_all('td')
            if len(cells) == 2:
                # Get the specification name and value
                spec_name = cells[0].get_text(strip=True)
                spec_value = cells[1].get_text(strip=True)
                specifications[spec_name] = spec_value

    # Extract the CO2 Emission (LTA) value
    co2_emission = specifications.get('CO2 Emission', None) or specifications.get('CO2 Emission (LTA)', None)

    # Print the CO2 Emission (LTA)
    return co2_emission

def get_emission_data(url_id, title):
    car_code_id = get_car_code_url(url_id)
    subcode_dict = get_subcode_dict(car_code_id)
    best_match_name, spec_link = get_best_spec_link(title, subcode_dict)
    emission_data = get_emission_data_from_spec_link(spec_link)
    return emission_data

def main():
    parser = argparse.ArgumentParser(description='Find the closest subcode match for a given car name.')
    parser.add_argument('--url_id', type=str, default='1307612', help='URL ID of the webpage to scrape.')
    parser.add_argument('--title', type=str, default='Lexus GS300 (COE till 06/2026)', help='URL ID of the webpage to scrape.')

    args = parser.parse_args()
    
    url_id = args.url_id
    title = args.title
    
    car_code_id = get_car_code_url(url_id)
    subcode_dict = get_subcode_dict(car_code_id)
    best_match_name, spec_link = get_best_spec_link(title, subcode_dict)
    print(get_emission_data_from_spec_link(spec_link))

if __name__ == "__main__":
    main()