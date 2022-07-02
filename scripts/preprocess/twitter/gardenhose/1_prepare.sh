#!/bin/sh

## Example preprocessing script. In practice, it's recommended to parallelize some
## of these steps over files and then aggregate results after.

## Process the Raw Twiter Data
python scripts/preprocess/preprocess.py \
    --input "/export/c12/mdredze/twitter/public/2020/05/" \
    --output_folder "./data/processed/twitter/gardenhose/" \
    --platform "twitter" \
    --min_date "2019-01-01" \
    --max_date "2020-12-31" \
    --jobs 4

## Apply Language and User Filtering (Isolate Users Based on External Criteria)
## Assuming a user subset has been identified using 0_count_gardenhose.py
python scripts/preprocess/preprocess_filter.py \
    "./data/processed/twitter/gardenhose/*.gz" \
    --output_dir "./data/processed/twitter/gardenhose-filtered/" \
    --platform "twitter" \
    --user_list "./data/processed/twitter/gardenhose/users.filtered.txt" \
    --lang "en" \
    --jobs 4