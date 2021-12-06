# ML-OPT Equal Opportunities

## Creating data

1. Download zip from [ACS 2019 per-person data](https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_pus.zip)
2. Extract the zip to the `data/dataset` folder
3. Run `create_data.py` and find the output csv in `data/output`

## Preprocessing data

1. Run `preprocess_data.py` and find the preprocessed csv in `data/output`

## Creating test data for debugging

1. Enter `data/dataset` folder
2. Run bash script `./create_test_data.sh 100` , where 100 the number of samples
3. Alternatively, run `./create_test_data_random.sh` to get a random 0.1% subset of the rows


## Running Sparse and Holistic Regression

The sparse and holistic regression frameworks are written in Sparsity.jl, but for demonstration purposes you can use Sparsity.ipynb. It was tested on `Julia 1.6.3`. Create and preprocess the data first. 

1. Open `Sparsity.ipynb` notebook.
2. Run all (Will take up to 15-20 minutes)


## Running the Prescriptive part

In order to run the prescriptive part
of the project, first run the
`create_data.py` and the `preprocess_data.py`
scripts.
Then, you can run the `prescriptive.py` 
file to create and visualize the prescriptions.
Note that the file has a lot of python requirements,
such as numpy, seaborn, XGBoost, plotly, pandas.


## Column mappings
If you want to see a comprehensive list of what
each occupation code represents in the visualizations (e.g. MGR, BUS),
take a look at the column SOCP_desc of the file `data/metadata/mapping.csv`
 
