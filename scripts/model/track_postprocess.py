
"""
Aggregate keyword matches that were found by parallelizing the search
across the CLSP grid.
"""

## Path to Result Directory Initialized by Scheduler Script
BASE_PATH = "/export/c01/kharrigian/semantic-shift-websci-2022/"
RESULT_PATH = f"{BASE_PATH}/data/results/track/gardenhose/results/"

## Plotting Parameters
PLOT_TOP_K = 30
PLOT_MIN_MATCHES = 100

##################
### Imports
##################

## Standard Library
import os
import sys
from glob import glob

## External Libraries
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

## Project Library
from semshift.util.logging import initialize_logger

## Subproject Speciic Helpers
_ = sys.path.append(os.path.abspath(os.path.dirname(__file__))+"/")
from track_apply import plot_match_timeseries, plot_top_matches

##################
### Globals
##################

## Logging
LOGGER = initialize_logger()

##################
### Functions
##################

def format_index(df):
    """

    """
    if df is not None and df.shape[0] > 0:
        df.index = pd.to_datetime(df.index)
    return df

def main():
    """

    """
    ## Refresh Timeseries Plotting Directory
    ts_plot_dir = f"{RESULT_PATH}/../timeseries/"
    if os.path.exists(ts_plot_dir):
        _ = os.system(f"rm -rf {ts_plot_dir}")
    _ = os.makedirs(ts_plot_dir)
    ## Cache
    post_counts = []
    match_counts = []
    missing = {"posts":[],"matches":[]}
    ## Get Posts and Matches
    directories = sorted(glob(f"{RESULT_PATH}/*/"))
    for d in tqdm(directories, file=sys.stdout, total=len(directories), desc="Loading Tracking Data"):
        if os.path.exists(f"{d}/n_posts.csv"):
            d_posts = format_index(pd.read_csv(f"{d}/n_posts.csv",index_col=0))
            post_counts.append(d_posts)
        else:
            missing["posts"].append(d)
        if os.path.exists(f"{d}/match_counts.csv"):
            d_matches = format_index(pd.read_csv(f"{d}/match_counts.csv",index_col=0))
            match_counts.append(d_matches)
        else:
            missing["matches"].append(d)
    for df_type, file_list in missing.items():
        if len(file_list) > 0:
            LOGGER.warning(f"Warning: {len(file_list)} {df_type.title()} Files Are Missing")
    ## Concatenate Counts
    post_counts = pd.concat(list(filter(lambda i: i.shape[0] > 0, post_counts))).reset_index()
    match_counts = pd.concat(list(filter(lambda i: i.shape[0] > 0, match_counts))).reset_index()
    ## Group by Time
    post_counts = post_counts.groupby(["date"]).sum().sort_index()
    match_counts = match_counts.groupby(["date"]).sum().sort_index()
    ## Save Data
    LOGGER.info("Saving Aggregated Match Counts")
    _ = post_counts.to_csv(f"{RESULT_PATH}/../n_posts.csv")
    _ = match_counts.to_csv(f"{RESULT_PATH}/../match_counts.csv")
    ## Summary Plot
    LOGGER.info("Plotting Top Matches")
    fig, ax = plot_top_matches(match_counts, PLOT_TOP_K)
    fig.savefig(f"{RESULT_PATH}/../top_matches.png",dpi=150)
    plt.close(fig)
    ## Generate Individual Plots
    for term in tqdm(match_counts.columns, desc="Generating Timeseries Plots", file=sys.stdout):
        ## Check For Total Match Criteria
        if match_counts[term].sum() < PLOT_MIN_MATCHES:
            continue
        ## Format Term for Saving
        term_clean = term.replace("/","<slash>").replace(".","<period>").replace("#","<hashtag>").replace(" ","_")
        ## Create Plot
        fig, ax = plot_match_timeseries(match_counts, post_counts, [term])
        fig.savefig(f"{ts_plot_dir}{term_clean}.png", dpi=300)
        plt.close(fig)

####################
### Execute
####################

if __name__ == "__main__":
    _ = main()