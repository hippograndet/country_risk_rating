import random
from src.utils import countries

def list_all_series(client, provider, dataset):
    all_series = []
    offset = 0
    limit = 1000

    while True:
        data = client.list_series(provider, dataset, limit, offset)
        series = data.get("series", [])
        if not series:
            break
        all_series.extend(series)
        offset += limit

    return all_series

def check_countries_covered(dataset_dimensions_dict, threshold=2/3):
    covered_oecd_countries = []

    ref_area_col_name = ''
    if 'COU' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'COU'
    elif 'country' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'country'
    elif 'REF_AREA' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'REF_AREA'
    elif 'ref_area' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'ref_area'
    elif 'area' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'area'
    elif 'economy' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'economy'
    elif 'co_ter' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'co_ter'
    elif 'BORROWERS_CTY' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'BORROWERS_CTY'
    elif 'L_REP_CTY' in dataset_dimensions_dict.keys():
        ref_area_col_name = 'L_REP_CTY'
    else:
        print('Couldn\'t find Country dimension values')
        # Suppose it is.
        return True

        
    covered_countries = list(dataset_dimensions_dict[ref_area_col_name].keys()) + list(dataset_dimensions_dict[ref_area_col_name].values())
    for c_code in covered_countries:
        c_code_iso3 = ''
        if len(c_code) == 3:
            c_code_iso3 = c_code
        elif len(c_code) == 2:
            c_code_iso3 = countries.country_registry.get_ISO3_from_ISO2(c_code)
        else:
            c_code_iso3 = countries.country_registry.get_ISO3_from_name(c_code)

        covered_oecd_countries.append(c_code_iso3)

    countries_covered_r = len(covered_oecd_countries) / len(countries.country_registry.get_OECD_ISO3_list())
    print('Percent of OECD Countries covered:', round(countries_covered_r * 100, 2), '%')

    if countries_covered_r > threshold:
        return True
    else:
        return False
    
def check_valid_frequency(dataset_dimensions_dict):
    freq_col_name = ''
    if 'FREQ' in dataset_dimensions_dict.keys():
        freq_col_name = 'FREQ'
    elif 'freq' in dataset_dimensions_dict.keys():
        freq_col_name = 'freq'
    elif 'frequency' in dataset_dimensions_dict.keys():
        freq_col_name = 'frequency'
    
    if freq_col_name != '':
        freq_dict = dataset_dimensions_dict[freq_col_name]

        if 'A' in freq_dict.keys():
            return True
        else:
            print('No annual frequency data:', freq_dict)
            return False
    else:
        print('Not found Frequency Dimension in keys', dataset_dimensions_dict.keys())
        ## Suppose it is
        return True
    
def get_indicators_rows_for_dataset(dataset_dimensions_dict, prev_row):

    indicators_rows = []

    indic_col_name = ''
    if 'indic' in dataset_dimensions_dict.keys():
        indic_col_name = 'indic'
    elif 'indicator' in dataset_dimensions_dict.keys():
        indic_col_name = 'indicator'
    elif 'INDICATOR' in dataset_dimensions_dict.keys():
        indic_col_name = 'INDICATOR'
    elif 'CLASSIFICATION' in dataset_dimensions_dict.keys():
        indic_col_name = 'CLASSIFICATION'  
    elif 'DSR_BORROWERS' in dataset_dimensions_dict.keys():
        indic_col_name = 'DSR_BORROWERS'  
    elif 'series' in dataset_dimensions_dict.keys():
        indic_col_name = 'series'
    elif 'indices' in dataset_dimensions_dict.keys():
        indic_col_name = 'indices'
    
    if indic_col_name != '':
        indicators_dict = dataset_dimensions_dict[indic_col_name]
        for indicator_code, indicator_name in indicators_dict.items():
            indicators_rows.append({'Indicator Code': indicator_code, 'Indicator Name': indicator_name} | prev_row)


    else:
        print('Not found Indicator Dimension in keys', dataset_dimensions_dict.keys())

    return []

def get_indicator_catalog(datasets_df, client):
    datasets_df = datasets_df.sample(frac=1.0)
    providers = datasets_df['Provider Id'].unique()
    random.shuffle(providers)

    indicators_rows = []
    for provider_id in providers:
        print('Provider:', provider_id)
        provider_datasets_df = datasets_df[datasets_df['Provider Id'] == provider_id]
        for i, dataset_row in provider_datasets_df.iterrows():
            dataset_id = dataset_row['Dataset Id']
            print('\tDataset:', dataset_id)
            try:
                dataset_dict = client.list_series(provider_id, dataset_id)
                dataset_dimensions_dict = dataset_dict['dataset']['dimensions_values_labels']

                if check_countries_covered(dataset_dimensions_dict):
                    if check_valid_frequency(dataset_dimensions_dict):
                        indicators_rows = indicators_rows + get_indicators_rows_for_dataset(dataset_dimensions_dict, dataset_row)
                        print(':::New Number of Indicators:', len(indicators_rows))
                else:
                    print('Not Enough countries covered for:', provider_id, dataset_id)

            
            except SyntaxError:
                    print('Error querying dataset:', provider_id, dataset_id)