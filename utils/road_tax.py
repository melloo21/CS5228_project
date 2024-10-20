
# REF :: https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/upfront-vehicle-costs/tax-structure.html
def calc_power(power):
    base_road_tax = 200
    if power <= 7.5:
        road_tax = base_road_tax
    elif power <= 30:
        road_tax = (200 + 2 * (power - 7.5))
    elif power <= 230:
        road_tax = (250 + 3.75 * (power - 30))
    else:
        road_tax = (1525 + 10 * (power - 230)) 
    return road_tax * 0.782

def calc_engine_cc(engine_capacity):
    base_road_tax = 200
    if engine_capacity <= 600:
        road_tax = base_road_tax
    elif engine_capacity <= 1000:
        road_tax = 200 + 0.125 * (engine_capacity - 600)
    elif engine_capacity <= 1600:
        road_tax = 250 + 0.375 * (engine_capacity - 1000)
    elif engine_capacity <= 3000:
        road_tax = 475 + 0.75 * (engine_capacity - 1600)
    else:
        road_tax = 1525 + 1 * (engine_capacity - 3000)   
    return road_tax * 0.782

def check_annual_rate(date):
    year_2023 = pd.Timestamp('2023-01-01')
    year_2022 = pd.Timestamp('2022-01-01')
    if date >=year_2023 :
        return 700
    elif date >= year_2022:
        return 400
    else:
        return 200

def calculate_road_tax(engine_capacity, power, age_of_car, reg_date, scheme="Normal", fuel_type="Petrol"):
    """Calculate Singapore road tax based on engine capacity, age of car, vehicle scheme, and fuel type."""

    if fuel_type == "electric":
        road_tax = calc_power(power)
        road_tax += check_annual_rate(reg_date)
        
    elif fuel_type == "petrol-electric":
        road_tax = max(calc_power(power), calc_engine_cc(engine_capacity))

    elif fuel_type == "diesel":
        # https://nea.gov.sg/our-services/pollution-control/air-pollution/air-pollution-regulations
        # Assumption Euro IV compliant
        road_tax = min(525, (engine_capacity*0.625-100))
    else:
        # Base Road Tax Calculation by Engine Capacity
        road_tax = calc_engine_cc(engine_capacity)
    
    # Adjustments for Vehicle Scheme
    if scheme == "OPC":
        # up to $500 on annual road tax, subject to a minimum road tax payment of $70 per year.
        road_tax = min(road_tax - 500, 70) 

    return road_tax

