

"""
Example of script used for generating subset of users to keep in the Gardenhose dataset.
Given the size of the Twitter cache, it's probably better to create a parallelized version
with Map-reduce-esque structure
"""

## Number of Jobs
NUM_JOBS=8
## Set Minimum Threshold
MIN_THRESHOLD = 50
## Choose Output
OUTPUT_FILE = "./data/processed/twitter/gardenhose/users.filtered.txt"

## Imports
import gzip
import json
from glob import glob
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool

## Counter Function
def count_users_in_file(filename):
    """

    """
    ## Open the File
    user_counts = Counter()
    with gzip.open(filename,"r") as the_file:
        for line in the_file:
            ## Load
            line_data = json.loads(line)
            ## Check Format
            if isinstance(line_data, dict):
                line_data = [line_data]
            ## Subline Processing
            for subline in line_data:
                if not isinstance(subline, dict):
                    raise TypeError("Unexpected data type.")
                ## Case 1: Already Processed
                if "user_id_str" in subline:
                    user_counts[subline["user_id_str"]] += 1
                ## Case 2: Raw Data
                else:
                    if "delete" in line_data.keys():
                        continue
                    else:
                        user_counts[line_data["user"]["id_str"]] += 1
    return user_counts

def main():
    """

    """
    ## Get Filenames
    filenames = glob("/export/c12/mdredze/twitter/public/2019/*/*.gz") + \
                glob("/export/c12/mdredze/twitter/public/2020/*/*.gz")
    ## Get Counts
    with Pool(NUM_JOBS) as mp:
        user_counts = list(tqdm(mp.imap_unordered(count_users_in_file, filenames),
                                total=len(filenames),
                                desc="[Counting Users]"))
    ## Sum Over Files
    user_counts = sum(user_counts, Counter())
    ## Isolate Users Meeting Threshold
    users = [u for u, c in user_counts.items() if c >= MIN_THRESHOLD]
    ## Store Users
    with open(OUTPUT_FILE,"w") as the_file:
        for u in users:
            the_file.write(f"{u}\n")

## Execute
if __name__ == "__main__":
    _ = main()