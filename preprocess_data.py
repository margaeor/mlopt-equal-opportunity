import os
import pandas as pd
from constants import *

input_file = OUTPUT_FILE
preprocessed_path = os.path.join(OUTPUT_PATH, 'preprocessed.csv')


age_range = (25, 55)

def mapFunc(x, mapFrom, mapTo):
    try:
        return mapTo[list(mapFrom).index(x)]    
    except:
        return x

    
    
if __name__ == '__main__':



    df_meta = pd.read_csv(METADATA_FILE)
    df = pd.read_csv(input_file)

    # Filter data by age
    df = df.loc[(df.age >= age_range[0]) & (df.age <= age_range[1]), :]

    # MAPPINGS
    # Import mappings of columns
    col_mapper = pd.read_csv(os.path.join(METADATA_PATH, 'columns.csv'))
    # Extract variables, variable aliases and descriptions
    variables = list(col_mapper['Variable'])
    variable_aliases = list(col_mapper['VariableRename'])
    variables_map = [x + '_map' for x in variables]
    variable_aliases_map = [x + '_map' for x in variable_aliases]
    mappings = pd.read_csv(os.path.join(METADATA_PATH,'mapping.csv'))
    mappings = mappings.rename(columns={k: v for k, v in zip(variables, variable_aliases)})
    mappings = mappings.rename(columns={k: v for k, v in zip(variables_map, variable_aliases_map)})


    output_mapped = pd.DataFrame()
        
    for var in variable_aliases:
        if var in mappings.columns:
            
            # print(f"If : {var}")
            mapFrom = list(mappings[var])
            mapTo = list(mappings[var+'_map'])
            initial_data = df[var]

            mapped_data = list(initial_data.map(lambda x : mapFunc(x, mapFrom, mapTo), na_action='ignore'))
            output_mapped[var] = mapped_data
        else:
            output_mapped[var] = df[var]
            # print(f"Else: {var}")

    df = output_mapped

    categorical_cols = df_meta.loc[df_meta.IsCategorical == 1, 'VariableRename'].to_list()
    non_categorical_cols = df_meta.loc[df_meta.IsCategorical == 0, 'VariableRename'].to_list()

    # Perform one-hot encoding on categorical columns
    print("Performing one-hot encoding on categorical columns")
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)

    # Replace missing values with 0 in non-categorical columns (mostly related to income)
    print("Replacing missing values of non categorical columns")
    df[non_categorical_cols] = df[non_categorical_cols].fillna(value=0)

    print(f"Dataframe size after one-hot {df.shape}")
    df.to_csv(preprocessed_path, index=False, chunksize=100000)