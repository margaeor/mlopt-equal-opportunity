import os
import pandas as pd
from constants import *



if __name__ == '__main__':
    input_file = OUTPUT_FILE

    # PERFORM MAPPING OF CATEGORICAL DATA
    out_data = input_file.fillna(0)
    mappings = pd.read_csv(os.path.join(METADATA_PATH,'mapping.csv'))
    variables_map = [x + '_map' for x in variables]
    variable_aliases_map = [x + '_map' for x in variable_aliases]
    mappings = mappings.rename(columns={k: v for k, v in zip(variables, variable_aliases)})
    mappings = mappings.rename(columns={k: v for k, v in zip(variables_map, variable_aliases_map)})

    output_mapped = pd.DataFrame()
        
    for var in variable_aliases:
        if var in mappings.columns:
            
            # print(f"If : {var}")
            mapFrom = list(mappings[var])
            mapTo = list(mappings[var+'_map'])
            initial_data = out_data[var]

            mapped_data = list(map(lambda x : mapTo[list(mapFrom).index(x)], list(initial_data)))
            output_mapped[var] = mapped_data
            print(output_mapped[var])
            print(mapped_data)
        else:
            output_mapped[var] = out_data[var]
            # print(f"Else: {var}")

    # output_mapped

    preprocessed_path = os.path.join(OUTPUT_PATH, 'preprocessed.csv')

    df_meta = pd.read_csv(METADATA_FILE)
    df = pd.read_csv(input_file)

    categorical_cols = df_meta.loc[df_meta.IsCategorical == 1, 'VariableRename'].to_list()
    print(f"Categorical columns: {categorical_cols}")
    print("Performing one-hot encoding on categorical columns")

    df = pd.get_dummies(df, columns=categorical_cols)

    print(f"Dataframe size after one-hot {df.shape}")

    df.to_csv(preprocessed_path, index=False)