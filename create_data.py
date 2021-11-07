from collections import OrderedDict
import pandas as pd
import os


DATASET_PATH = 'dataset'
METADATA_PATH = 'metadata'
OUTPUT_PATH = 'output'

if __name__ == '__main__':

    col_mapper = pd.read_csv('columns.csv')
    variables = list(col_mapper['Variable'])
    variable_aliases = list(col_mapper['VariableRename'])
    variable_descriptions = list(col_mapper['VariableRename'])

    df = pd.read_csv(os.path.join(DATASET_PATH, 'psam_pusa.csv'))
    #df = pd.read_csv(os.path.join(DATASET_PATH, 'test.csv'))

    df = df[variables].rename(columns={k: v for k, v in zip(variables, variable_aliases)})


    df.to_csv('out.csv', index=False)
    #df = df.head(100)
    #print(df.shape)