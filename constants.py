import os

DATA_PATH = 'data'
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
METADATA_PATH = os.path.join(DATA_PATH, 'metadata')
OUTPUT_PATH = os.path.join(DATA_PATH, 'output')

DATASET_URLS = [
    "https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_pus.zip",
    "https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_hus.zip"
]

DEBUG = False

PERSON_DATASETS = ['psam_pusa.csv', 'psam_pusb.csv']
HOUSEHOLD_DATASETS = ['psam_husa.csv', 'psam_husb.csv']

JOIN_COLUMN = 'SERIALNO'


# Files
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'out.csv')
METADATA_FILE = os.path.join(METADATA_PATH, 'columns.csv')