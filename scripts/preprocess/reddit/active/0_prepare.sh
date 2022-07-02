#!/bin/sh

## Example of Processing the Raw Reddit Data
python scripts/preprocess/preprocess.py \
    --input "./data/raw/reddit/active/histories/" \
    --output_folder "./data/processed/reddit/active/" \
    --platform "reddit" \
    --min_date "2019-01-01" \
    --max_date "2020-07-01" \
    --jobs 4