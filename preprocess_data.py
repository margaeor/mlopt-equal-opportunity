import os
import pandas as pd
from constants import *

input_file = OUTPUT_FILE
preprocessed_path = os.path.join(OUTPUT_PATH, 'preprocessed.csv')

age_range = (25, 55)

if __name__ == '__main__':


    df_meta = pd.read_csv(METADATA_FILE)
    df = pd.read_csv(input_file)

    # Filter data by age
    df = df.loc[(df.age >= age_range[0]) & (df.age <= age_range[1]), :]

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