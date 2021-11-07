from collections import OrderedDict
import pandas as pd
import os


DATA_PATH = 'data'
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
METADATA_PATH = os.path.join(DATA_PATH, 'metadata')
OUTPUT_PATH = os.path.join(DATA_PATH, 'output')

if __name__ == '__main__':

    # Import mappings of columns
    col_mapper = pd.read_csv(os.path.join(METADATA_PATH, 'columns.csv'))

    # Extract variables, variable aliases and descriptions
    variables = list(col_mapper['Variable'])
    variable_aliases = list(col_mapper['VariableRename'])
    variable_descriptions = list(col_mapper['VariableRename'])

    # Read dataset
    df = pd.read_csv(os.path.join(DATASET_PATH, 'psam_pusa.csv'))
    #df = pd.read_csv(os.path.join(DATASET_PATH, 'test.csv'))

    # Extract columns and rename
    df = df[variables].rename(columns={k: v for k, v in zip(variables, variable_aliases)})


    df.to_csv(os.path.join(OUTPUT_PATH, 'out.csv'), index=False)
