import os
import pandas as pd
from constants import *



if __name__ == '__main__':
    input_file = OUTPUT_FILE
    preprocessed_path = os.path.join(OUTPUT_PATH, 'preprocessed.csv')

    df_meta = pd.read_csv(METADATA_FILE)
    df = pd.read_csv(input_file)

    categorical_cols = df_meta.loc[df_meta.IsCategorical == 1, 'VariableRename'].to_list()
    print(f"Categorical columns: {categorical_cols}")
    print("Performing one-hot encoding on categorical columns")

    df = pd.get_dummies(df, columns=categorical_cols)

    print(f"Dataframe size after one-hot {df.shape}")

    df.to_csv(preprocessed_path, index=False)