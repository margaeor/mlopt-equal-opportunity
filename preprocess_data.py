import os
import pandas as pd
from constants import *

import numpy as np

input_file = OUTPUT_FILE
preprocessed_path = os.path.join(OUTPUT_PATH, 'preprocessed.csv')


age_range = (25, 55)
predictor_column = 'income_total'

def mapFunc(x, mapFrom, mapTo):
    try:
        return mapTo[list(mapFrom).index(x)]    
    except:
        return x


def discard_unmapped_values(df_map, df, variable_aliases):
    mapped_cols = set(df_map.columns.to_list()).intersection(variable_aliases)
    print(mapped_cols)
    print(f"Number of rows before map removal {df.shape[0]}")
    for col in mapped_cols:
        legal_values = set(df_map[col].to_list())
        df = df.loc[(df[col].isna()) | (df[col].isin(legal_values)), :]

    print(f"Number of rows after map removal {df.shape[0]}")

    return df
if __name__ == '__main__':



    df_meta = pd.read_csv(METADATA_FILE)
    df = pd.read_csv(input_file)

    # Filter data by age
    df = df.loc[(df.age >= age_range[0]) & (df.age <= age_range[1]), :]

    # Filter out people without income
    df = df.loc[df.income_total != 0, :]

    # MAPPINGS
    # Import mappings of columns
    col_mapper = pd.read_csv(os.path.join(METADATA_PATH, 'columns.csv'))
    # Extract variables, variable aliases and descriptions
    variables = list(col_mapper['Variable'])
    variable_aliases = list(col_mapper['VariableRename'])
    variables_map = [x + '_map' for x in variables]
    variable_aliases_map = [x + '_map' for x in variable_aliases]
    mappings = pd.read_csv(os.path.join(METADATA_PATH, 'mapping.csv'))
    mappings = mappings.rename(columns={k: v for k, v in zip(variables, variable_aliases)})
    mappings = mappings.rename(columns={k: v for k, v in zip(variables_map, variable_aliases_map)})

    df = discard_unmapped_values(mappings, df, variable_aliases)

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

    # MERGE fieldOfDegree1 and fieldOfDegree2
    df["num_degrees"] = np.zeros(df.shape[0])
    for i in range(1,65):
        if any(("field_of_degree_1_"+str(i)+".0") in s for s in df.columns):
            df["field_of_degree_"+str(i)] = np.minimum(df["field_of_degree_1_"+str(i)+".0"] + df["field_of_degree_2_"+str(i)+".0"], 1)
            df["num_degrees"] = df["num_degrees"] + df["field_of_degree_1_"+str(i)+".0"] + df["field_of_degree_2_"+str(i)+".0"]
            df.drop("field_of_degree_1_"+str(i)+".0", axis=1, inplace= True)
            df.drop("field_of_degree_2_"+str(i)+".0", axis=1, inplace= True)
            # df["field_of_degree_"+str(i)] = np.minimum(df["field_of_degree_1_"+str(i)+".0"] + df["field_of_degree_2_"+str(i)+".0"], 1)
            # df["num_degrees"] = df["num_degrees"] + df["field_of_degree_1_"+str(i)+".0"] + df["field_of_degree_2_"+str(i)+".0"]
            # df.drop(columns=["field_of_degree_1_"+str(i)+".0", "field_of_degree_2_"+str(i)+".0"])

    df["field_of_degree_"+str(float("NaN"))] = np.maximum((df["field_of_degree_2_"+str(float("NaN"))]+df["field_of_degree_1_"+str(float("NaN"))]).astype('int') - 1, 0)
    df.drop("field_of_degree_2_"+str(float("NaN")), axis=1, inplace= True)
    df.drop("field_of_degree_1_"+str(float("NaN")), axis=1, inplace= True)


    # df["field_of_degree_"+str(i)] = np.minimum(df["field_of_degree_1_"+str(i)+".0"] + df["field_of_degree_2_"+str(i)+".0"], 1)
    # df["num_degrees"] = df["num_degrees"] + df["field_of_degree_1_"+str(i)+".0"] + df["field_of_degree_2_"+str(i)+".0"]
    # df.drop("field_of_degree_1_"+str(i)+".0", axis=1, inplace= True)
    # df.drop("field_of_degree_2_"+str(i)+".0", axis=1, inplace= True)
    # # df["field_of_degree_"+str(i)] = np.minimum(df["field_of_degree_1_"+str(i)+".0"] + df["field_of_degree_2_"+str(i)+".0"], 1)
    # # df["num_degrees"] = df["num_degrees"] + df["field_of_degree_1_"+str(i)+".0"] + df["field_of_degree_2_"+str(i)+".0"]
    # # df.drop(columns=["field_of_degree_1_"+str(i)+".0", "field_of_degree_2_"+str(i)+".0"])



    df = pd.get_dummies(df, columns=["num_degrees"], dummy_na=True)

    # Replace missing values with 0 in non-categorical columns (mostly related to income)
    print("Replacing missing values of non categorical columns")
    df[non_categorical_cols] = df[non_categorical_cols].fillna(value=0)




    print(f"Dataframe size after one-hot {df.shape}")
    df.to_csv(preprocessed_path, index=False, chunksize=100000)