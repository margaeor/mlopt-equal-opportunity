import pandas as pd
import os
from io import BytesIO
from zipfile import ZipFile
import requests

DATA_PATH = 'data'
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
METADATA_PATH = os.path.join(DATA_PATH, 'metadata')
OUTPUT_PATH = os.path.join(DATA_PATH, 'output')

DATASET_URLS = [
    "https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_pus.zip",
    "https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_hus.zip"
]

DEBUG = True

PERSON_DATASETS = ['psam_pusa.csv', 'psam_pusb.csv']
HOUSEHOLD_DATASETS = ['psam_husa.csv', 'psam_husb.csv']

JOIN_COLUMN = 'SERIALNO'

def download_and_extract_zip(url, output):
    with requests.get(url) as file:
        with ZipFile(BytesIO(file.content)) as zfile:
            zfile.extractall(output)


if __name__ == '__main__':


    # If dataset doesn't exist, download it and extract it
    for url, datasets in zip(DATASET_URLS, [PERSON_DATASETS, HOUSEHOLD_DATASETS]):
        for dataset in datasets:
            if not DEBUG and not os.path.exists(os.path.join(DATASET_PATH, dataset)):
                print(f"Downloading dataset {dataset}")
                download_and_extract_zip(
                    "https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_hus.zip",
                    DATASET_PATH
                )

    if DEBUG:
        PERSON_DATASETS = ['test_pa.csv', 'test_pb.csv']
        HOUSEHOLD_DATASETS = ['test_ha.csv', 'test_hb.csv']

    # Import mappings of columns
    col_mapper = pd.read_csv(os.path.join(METADATA_PATH, 'columns.csv'))

    # Extract variables, variable aliases and descriptions
    variables = list(col_mapper['Variable'])
    variable_aliases = list(col_mapper['VariableRename'])
    variable_descriptions = list(col_mapper['Description'])

    variables += [JOIN_COLUMN]
    cols_v = set(variables)
    
    ###########################
    ######## PREVIOUS ########
    #########################
    # # Read datasets
    # print("Reading datasets")
    # dfs = [pd.read_csv(os.path.join(DATASET_PATH, pdt)) for pdt in PERSON_DATASETS]
    # dfp = pd.concat(dfs)
    # cols_p = set(dfp.columns)
    # dfp = dfp[cols_p & cols_v]

    # dfs = [pd.read_csv(os.path.join(DATASET_PATH, pdt)) for pdt in HOUSEHOLD_DATASETS]
    # dfh = pd.concat(dfs)
    # cols_h = set(dfh.columns)
    # dfh = dfh[(cols_v-cols_p).union({JOIN_COLUMN})]

    ######################
    ######## NEW ########
    ####################
    # Read datasets
    # Loop because we cannot load the entire dataset at once, we need to load gradually and reduce columns first
    dfp_final = pd.DataFrame()
    dfh_final = pd.DataFrame()
    for pdt in PERSON_DATASETS:
        print(f"Reading dataset {pdt}")
        dfp = pd.read_csv(os.path.join(DATASET_PATH, pdt))
        # dfp = pd.concat(dfs)
        cols_p = set(dfp.columns)
        dfp_final = pd.concat([dfp_final, dfp[cols_p & cols_v].copy()])

    for hdt in HOUSEHOLD_DATASETS:
        print(f"Reading dataset {hdt}")
        dfh = pd.read_csv(os.path.join(DATASET_PATH, hdt)) 
        # dfh = pd.concat(dfs)
        cols_h = set(dfh.columns)
        dfh_final = pd.concat([dfh_final, dfh[(cols_v-cols_p).union({JOIN_COLUMN})].copy()])

    # (thats  a little sketchy but the rest of the stuff use dfp and dfh as names...)
    dfp = dfp_final
    dfh = dfh_final

    #df = pd.read_csv(os.path.join(DATASET_PATH, 'test.csv'))


    print(f"Columns that don't exist in Person {cols_v-cols_p}")

    print(f"Shape before merge {dfp.shape}")
    df = dfp.merge(dfh, how='inner', on=JOIN_COLUMN)
    print(f"Shape after merge {df.shape}")

    # Extract columns and rename
    df = df.rename(columns={k: v for k, v in zip(variables, variable_aliases)})

    # Write CSV to output directory (create it if doesn't exist)
    if not os.path.exists(OUTPUT_PATH):
        print(f"Creating {OUTPUT_PATH} directory")
        os.mkdir(OUTPUT_PATH)
    df.to_csv(os.path.join(OUTPUT_PATH, 'out.csv'), index=False)
