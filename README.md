# ML-OPT Equal Opportunities

## Creating data

1. Download zip from [ACS 2019 per-person data](https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_pus.zip)
2. Extract the zip to the `data/dataset` folder
3. Run create_data.py and find the output csv in `data/output`


## Creating test data for debugging

1. Enter `data/dataset` folder
2. Run bash script `./create_test_data.sh 100` , where 100 the number of samples
3. Alternatively, run `./create_test_data_random.sh` to get a random 0.1% subset of the rows