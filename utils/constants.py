local_path = "/Users/melloo21/Desktop/NUS Items/CS5228/Project/CS5228_project"
folder= "processed_dataset"
# train_dataset = "train_data_cleaned.csv"
# val_dataset= "val_data_cleaned.csv"
train_dataset = "train_wo_simple_imputer_with_outlier_removed.csv"
val_dataset= "val_wo_simple_imputer_with_outlier_removed.csv"
# train_dataset = "train_w_outlier_knn_impute_ft_transform.csv"
# val_dataset= "val_w_outlier_knn_impute_ft_transform.csv"

VEHICLE_CATEGORIES = [
    {"sports car"},
    {"luxury sedan", "suv"},
    {"hatchback", "stationwagon", "mid-sized sedan"},
    {"others", "mpv"},
]

FEATURE_SET_W_COL_DROPPED = [ 'manufactured', 'transmission', 'curb_weight', 'power',
        'engine_cap','no_of_owners', 'depreciation', 'coe', 'road_tax', 'dereg_value',
       'mileage', 'omv', 'arf','car_age', 'type_of_vehicle_bus/mini bus',
       'type_of_vehicle_hatchback', 'type_of_vehicle_luxury sedan',
       'type_of_vehicle_mid-sized sedan', 'type_of_vehicle_mpv',
       'type_of_vehicle_others', 'type_of_vehicle_sports car',
       'type_of_vehicle_stationwagon', 'type_of_vehicle_suv',
       'type_of_vehicle_truck', 'type_of_vehicle_van', 'cylinder_cnt',
       'almost new car', 'coe car', 'consignment car', 'direct owner sale',
       'electric cars', 'hybrid cars', 'imported used vehicle',
       'low mileage car', 'opc car', 'parf car', 'premium ad car',
       'rare & exotic', 'sgcarmart warranty cars', 'sta evaluated car',
       'vintage cars', 'emission_data']

COLS_TO_DROP = ['listing_id', 'title', 'make', 'model', 'description', 'original_reg_date', 'type_of_vehicle',
'category',  '-', 'opc_scheme', 'lifespan','eco_category','features', 'accessories', 'indicative_price', 'price', 
'reg_date_dt','reg_date_year', 'reg_date_month', 'model_make', 'coe_age_left','age_range']


ENTIRE_FEATURE_LIST = ['listing_id', 'title', 'make', 'model', 'description', 'manufactured',
       'original_reg_date', 'reg_date', 'type_of_vehicle', 'category',
       'transmission', 'curb_weight', 'power', 'fuel_type', 'engine_cap',
       'no_of_owners', 'depreciation', 'coe', 'road_tax', 'dereg_value',
       'mileage', 'omv', 'arf', 'opc_scheme', 'lifespan', 'eco_category',
       'features', 'accessories', 'indicative_price', 'price', 'reg_date_dt',
       'reg_date_year', 'reg_date_month', 'model_make', 'coe_age_left',
       'age_range', 'car_age', 'type_of_vehicle_bus/mini bus',
       'type_of_vehicle_hatchback', 'type_of_vehicle_luxury sedan',
       'type_of_vehicle_mid-sized sedan', 'type_of_vehicle_mpv',
       'type_of_vehicle_others', 'type_of_vehicle_sports car',
       'type_of_vehicle_stationwagon', 'type_of_vehicle_suv',
       'type_of_vehicle_truck', 'type_of_vehicle_van', 'cylinder_cnt', '-',
       'almost new car', 'coe car', 'consignment car', 'direct owner sale',
       'electric cars', 'hybrid cars', 'imported used vehicle',
       'low mileage car', 'opc car', 'parf car', 'premium ad car',
       'rare & exotic', 'sgcarmart warranty cars', 'sta evaluated car',
       'vintage cars', 'emission_data']