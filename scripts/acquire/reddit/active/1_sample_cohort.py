
"""
From set of authors active at end of May 2020, identify those
who were also active at beginning of 2019. Requires outputs
from 0_count_users.py
"""

## Input/Output Directories
COUNT_PATH = "./data/raw/reddit/active/counts/"
CACHE_DIR = "./data/raw/reddit/active/author_comment_counts/"
PLOT_DIR = "./plots/acquire/reddit/active/"

## Date Range Parameters
START_DATE = "2019-01-01"
END_DATE = "2019-02-01"
FREQ = 7 # Number of Days or "all"

## Query Parameters
MAX_RETRIES = 3
BACKOFF = 2

## Sample Criteria
MIN_POSTS_PER_WEEK = 1
IGNORE_ACTIVE_TOP_PERCENTILE = 1

####################
### Imports
####################

## Standard Library
import os
import sys
import gzip
import json
from glob import glob
from time import sleep
from collections import Counter
from datetime import timedelta

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from retriever import Reddit as RedditData

## Project-Specific
from semshift.util.helpers import chunks
from semshift.util.logging import initialize_logger

####################
### Globals
####################

## Logger
LOGGER = initialize_logger()

## Reddit API Wrapper
REDDIT_API = RedditData(False)

## Cache Directory/Plot Directory
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

####################
### Helpers
####################

def get_author_comment_counts(author_list,
                              api,
                              start_date=None,
                              end_date=None,
                              chunksize="1W",
                              limit=10000):
    """

    """
    ## Maximum Reqeust Length
    if len(author_list) > 100:
        raise ValueError("Input author_list can only have a maximum of 100 authors")
    ## Range Formatted
    start_epoch = api._get_start_date(start_date)
    end_epoch = api._get_end_date(end_date)
    ## Chunk Queries into Time Periods
    time_chunks = api._chunk_timestamps(start_epoch,
                                        end_epoch,
                                        chunksize)
    ## Make Query Attempt
    df_all = []
    backoff = api._backoff if hasattr(api, "_backoff") else 2
    retries = api._max_retries if hasattr(api, "_max_retries") else 3
    total = 0
    for tcstart, tcstop in zip(time_chunks[:-1], time_chunks[1:]):
        ## Check Limit
        if limit is not None and total >= limit:
            break
        for _ in range(retries):
            try:
                ## Construct Call
                query_params = {"before":tcstop+1,
                                "after":tcstart,
                                "limit":limit,
                                "filter":["id","created_utc","author","author_fullname"],
                                "author":author_list}
                ## Construct Call
                req = api.api.search_comments(**query_params)
                ## Retrieve and Parse Data
                df = api._parse_psaw_comment_request(req)
                if len(df) > 0:
                    df = df.sort_values("created_utc", ascending=True)
                    df = df.reset_index(drop=True)
                    df_all.append(df)
                    total += len(df)
                break
            except Exception as e:
                sleep(backoff)
                backoff = 2 ** backoff
    if len(df_all) > 0:
        df_all = pd.concat(df_all).reset_index(drop=True)
        if limit is not None and len(df_all) > limit:
            df_all = df_all.iloc[:limit].copy()
        comment_counts = Counter(df_all["author"].value_counts().to_dict())
    else:
        comment_counts = Counter()
    ## Fill In Missing
    for a in author_list:
        if a not in comment_counts:
            comment_counts[a] = 0
    return comment_counts

####################
### Load Counts
####################

LOGGER.info("Loading Source Comment Counts")

## Identify Files
count_files = sorted(glob(f"{COUNT_PATH}*/*.json.gz"))

## Load and Store Counts
counts = Counter()
for cf in tqdm(count_files, file=sys.stdout):
    with gzip.open(cf,"r") as the_file:
        cf_counts = json.load(the_file)
    cf_counts = Counter(cf_counts)
    counts += cf_counts

## Remove Moderators/Bots
IGNORE_USERS = set(["AutoModerator","MemesMod","[deleted]","[removed]"])
counts = dict((x, y) for x, y in counts.items() if x not in IGNORE_USERS)

####################
### Check Activity over Time
####################

LOGGER.info("Starting Activity Query")

## Create Date Range
if FREQ == "all":
    date_range = [START_DATE, END_DATE]
else:
    date_range = [pd.to_datetime(START_DATE)]
    while date_range[-1] < pd.to_datetime(END_DATE):
        date_range.append(min(date_range[-1] + timedelta(FREQ), pd.to_datetime(END_DATE)))
    date_range = [i.date().isoformat() for i in date_range]

## Get Comment Counts over Date Range
author_chunks = list(chunks(sorted(counts.keys()), n=100))
for dstart, dstop in tqdm(zip(date_range[:-1], date_range[1:]), total=len(date_range)-1, position=0, leave=False, file=sys.stdout, desc="Date Range"):
    for a, author_chunk in tqdm(enumerate(author_chunks), total=len(author_chunks), position=1, leave=False, file=sys.stdout, desc="Author Chunk"):
        ## Check Cache
        cache_file = f"{CACHE_DIR}{dstart}_{dstop}_chunk-{a}.json.gz"
        if os.path.exists(cache_file):
            continue
        ## Query Data
        chunk_comment_counts = None
        for r in range(MAX_RETRIES):
            try:
                chunk_comment_counts = get_author_comment_counts(author_chunk,
                                                                 api=REDDIT_API,
                                                                 start_date=dstart,
                                                                 end_date=dstop)
                if chunk_comment_counts is not None:
                    break
            except:
                sleep(BACKOFF**r)
        ## Cache Data
        if chunk_comment_counts is None:
            chunk_comment_counts = Counter()
        with gzip.open(cache_file,"wt") as the_file:
            the_file.write(json.dumps(dict(chunk_comment_counts)))
        ## Delay Next Call (120 requests / minute)
        sleep(0.5)

## Load Activity Data Results
activity_files = sorted(glob(f"{CACHE_DIR}*.json.gz"))
dates_from_file = lambda i: tuple(os.path.basename(i).split("_chunk")[0].split("_"))
authors = sorted(counts.keys())
author_index = dict(zip(authors, list(range(len(counts)))))
date_index = dict((y, x) for x, y in enumerate(sorted(set(list(map(dates_from_file, activity_files))), key=lambda x: x[0])))
X = np.zeros((len(author_index), len(date_index)))
for chunk_file in tqdm(activity_files, desc="Loading Activity Data", file=sys.stdout):
    ## Load File Data
    with gzip.open(chunk_file,"r") as the_file:
        chunk_activity_data = json.load(the_file)
    ## Align Data for X
    chunk_authors = sorted(list(chunk_activity_data))
    chunk_row_index = [author_index[i] for i in chunk_authors]
    chunk_col_index = date_index[dates_from_file(chunk_file)]
    chunk_values = np.array([chunk_activity_data[a] for a in chunk_authors])
    ## Update X
    X[chunk_row_index, chunk_col_index] += chunk_values

####################
### Examine Distribution
####################

LOGGER.info("Visualizing Distribution")

## Plot Distribution of Posts
total_posts = X.sum(axis=1)
total_posts_vc = pd.Series(total_posts).value_counts()
fig, ax = plt.subplots()
ax.scatter(total_posts_vc.index,
           total_posts_vc.values,
           alpha=0.5)
ax.set_yscale("symlog")
ax.set_xscale("symlog")
ax.set_xlabel("# Posts", fontweight="bold")
ax.set_ylabel("# Users", fontweight="bold")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}reddit_post_distribution.png", dpi=300)
plt.close()

####################
### Select Cohort
####################

LOGGER.info("Sampling Cohort")

## Min Post Filter
cohort_mask = np.nonzero((X[:,:-1] >= MIN_POSTS_PER_WEEK).all(axis=1))[0]
cohort_authors = [authors[i] for i in cohort_mask]
cohort_X = X[cohort_mask]

## Outlier Filter
post_threshold = np.percentile(cohort_X.sum(axis=1), 100-IGNORE_ACTIVE_TOP_PERCENTILE)
cohort_mask = np.nonzero(cohort_X.sum(axis=1)<post_threshold)[0]
cohort_authors = [cohort_authors[i] for i in cohort_mask]
cohort_X = cohort_X[cohort_mask]

## Dump
cohort_file = f"{CACHE_DIR}user_sample.txt"
with open(cohort_file,"w") as the_file:
    for author in cohort_authors:
        the_file.write(f"{author}\n")

LOGGER.info("Script Complete!")