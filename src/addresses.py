latest_FE_folder = 'FE_DSP_v1/'
latest_BE_folder = 'BE_DSP_v1/'

# FE_address = 's3://lab-data-project/report/FrontEnd/'
BE_address = 's3://lab-data-project/report/BackEnd/'

latest_FE_address = FE_address + latest_FE_folder
latest_BE_address = BE_address + latest_BE_folder

country_FE_data_address = latest_FE_address + 'country/data/'
country_FE_model_address = latest_FE_address + 'country/model/'

company_FE_data_address = latest_FE_address + 'company/data/'
company_FE_model_address = latest_FE_address + 'company/model/'

sector_FE_data_address = latest_FE_address + 'sector/data/'
sector_FE_model_address = latest_FE_address + 'sector/model/'

country_BE_address = latest_BE_address + 'country/'
sector_BE_address = latest_BE_address + 'sector/'
company_BE_address = latest_BE_address + 'company/'