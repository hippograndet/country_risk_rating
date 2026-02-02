import pandas as pd

TOTAL_AVOIDED = 0

def get_code_name_dict(info_dict):
    return info_dict['code'], info_dict['name']

def get_provider_row(providers_dict):
    provider_row = {}
    id, name = get_code_name_dict(providers_dict['provider'])
    provider_row['Provider Id'] = id
    provider_row['Provider Name'] = name
    provider_row['Provider Region'] = providers_dict['provider']['region']    
    return provider_row

def get_subgroup_row(info_dict, i=1):
    dataset_row = {}
    id, name = get_code_name_dict(info_dict)
    dataset_row['Subgroup ' + str(i) + ' Id'] = id
    dataset_row['Subgroup ' + str(i) + ' Name'] = name
    return dataset_row

def get_dataset_row(datasets_dict):
    dataset_row = {}
    id, name = get_code_name_dict(datasets_dict)
    dataset_row['Dataset Id'] = id
    dataset_row['Dataset Name'] = name
    return dataset_row


def get_dataset_rows(d: dict, prev_row, depth=1):

    if depth > 3:
        # print('Too Deep Recursions for:', prev_row)
        global TOTAL_AVOIDED
        TOTAL_AVOIDED = TOTAL_AVOIDED + 1 
        return []
    
    subdataset_rows = []
    if 'children' in d.keys():
        prev_row = prev_row | get_subgroup_row(d, i=depth)
        for sub_d in d['children']:
            recursive_dataset_rows = get_dataset_rows(sub_d, prev_row, depth=depth+1)
            if recursive_dataset_rows == []:
                subdataset_rows = []
            else:
                subdataset_rows = subdataset_rows + recursive_dataset_rows

        return subdataset_rows
    else:
        dataset_row = get_dataset_row(d)
        final_row = prev_row | dataset_row
        return [final_row]

def get_complete_datasets_rows(client, providers):
    datasets_rows = []
    for provider in providers:
        providers_dict = client.list_datasets(provider)
        provider_row = get_provider_row(providers_dict)

        for dataset_dict in providers_dict['category_tree']:
            # if dataset_dict['code'] == 'GOV':
            subdatset_rows = get_dataset_rows(dataset_dict, provider_row, depth=1)
            datasets_rows = datasets_rows + subdatset_rows

    return datasets_rows

def get_all_datasets_df(client, providers):

    datasets_rows = get_complete_datasets_rows(client, providers)
    datasets_df = pd.DataFrame(datasets_rows)

    datasets_df = datasets_df.drop_duplicates(subset=['Dataset Id', 'Provider Id'])
    datasets_df = datasets_df.reset_index()

    return datasets_df 