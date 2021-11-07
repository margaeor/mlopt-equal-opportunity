from collections import OrderedDict
import pandas as pd
import os
import zipfile
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import requests

DATA_PATH = 'data'
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
METADATA_PATH = os.path.join(DATA_PATH, 'metadata')
OUTPUT_PATH = os.path.join(DATA_PATH, 'output')
DATASET_NAME = 'psam_pusa.csv'


def download_and_extract_zip(url, output):
    with requests.get(url) as file:
        with ZipFile(BytesIO(file.content)) as zfile:
            zfile.extractall(output)


if __name__ == '__main__':

    dataset = os.path.join(DATASET_PATH, DATASET_NAME)

    # If dataset doesn't exist, download it and extract it
    if not os.path.exists(dataset):
        print("Downloading dataset")
        download_and_extract_zip(
            "https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_pus.zip",
            DATASET_PATH
        )

    # Import mappings of columns
    col_mapper = pd.read_csv(os.path.join(METADATA_PATH, 'columns.csv'))

    # Extract variables, variable aliases and descriptions
    variables = list(col_mapper['Variable'])
    variable_aliases = list(col_mapper['VariableRename'])
    variable_descriptions = list(col_mapper['VariableRename'])

    # Read dataset
    print("Reading dataset")
    df = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME))
    #df = pd.read_csv(os.path.join(DATASET_PATH, 'test.csv'))

    # Extract columns and rename
    df = df[variables].rename(columns={k: v for k, v in zip(variables, variable_aliases)})


    df.to_csv(os.path.join(OUTPUT_PATH, 'out.csv'), index=False)
