
"""
Apply preprocessing (e.g. formatting + tokenization) to raw
Twitter or Reddit data files. Used for the unlabeled datasets.
"""

#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from glob import glob
from functools import partial
from datetime import datetime
from multiprocessing import Pool

## External Library
import pandas as pd
from tqdm import tqdm

## Local
from semshift.util.logging import initialize_logger
from semshift.preprocess.preprocess import (tokenizer,
                                            format_tweet_data)
from semshift.preprocess.preprocess import RawDataLoader

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()

## Cache Schema
DB_SCHEMA = {
    "twitter":{
        "tweet":{
                'user_id_str': 'user_id_str', 
                'created_utc': 'created_utc', 
                'text': 'text', 
                'text_tokenized': 'text_tokenized', 
                'id_str': 'tweet_id', 
                'entity_type': 'entity_type', 
                'date_processed_utc': 'date_processed_utc', 
                'source': 'source'}
    },
    "reddit":{
        "comment":{
                "author_fullname":"user_id_str",
                "created_utc":"created_utc",
                "body":"text",
                "text_tokenized":"text_tokenized",
                "subreddit":"subreddit",
                "id":"comment_id",
                "entity_type":"entity_type",
                'date_processed_utc': 'date_processed_utc', 
                'source': 'source'}
    }, 
}

#######################
### Functions
#######################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Preprocess raw Twitter or Reddit data")
    ## Generic Arguments
    parser.add_argument("--input",
                        type=str,
                        default=None,
                        help="Path to input folder of raw *.gz files or a single raw *.gz file")
    parser.add_argument("--output_folder",
                        type=str,
                        default=None,
                        help="Name of output folder for placing predictions.")
    parser.add_argument("--platform",
                        type=str,
                        choices=["twitter","reddit"],
                        help="Platform from which the data comes")
    parser.add_argument("--min_date",
                        type=str,
                        default=None,
                        help="Lower date boundary if desired.")
    parser.add_argument("--max_date",
                        type=str,
                        default=None,
                        help="Upper date boundary if desired.")
    parser.add_argument("--jobs",
                        type=int,
                        default=1,
                        help="Number of processes to spawn.")
    parser.add_argument("--keep_retweets",
                        default=False,
                        action="store_true",
                        help="If included, will preserve retweets in preprocessed data")
    parser.add_argument("--keep_non_english",
                        default=False,
                        action="store_true",
                        help="If included, will preserve non-English tweets in preprocessed data")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if args.input is None:
        raise ValueError("Must provide --input folder or .gz file")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Could not find input filepath {args.input}")
    if args.output_folder is None:
        raise ValueError("Must provide an --output_folder argument")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    return args

def get_file_list(args):
    """

    """
    if os.path.isfile(args.input):
        return [args.input]
    elif os.path.isdir(args.input):
        return glob(f"{args.input}*.gz")
    else:
        raise ValueError("Did not recognize command line --input")

def process_tweet_file(f,
                       min_date=None,
                       max_date=None,
                       output_folder=None,
                       keep_retweets=False,
                       keep_non_english=False):
    """
    Process raw tweet data and cache in a processed form

    Args:
        f (str): Path to tweet data. Expected to contain all tweets
                 desired for a single individual
        min_date
        max_date
        output_folder (str): Path to output folder for caching processed
                             data. If None, returns processed data itself
        keep_retweets (bool): If True, does not filter out retweets
        keep_non_english (bool): If True, does not filter out non-English tweets
    
    Returns:
        if output_folder is None:
            formatted_data (list): List of processed dictionaries
        else:
            fname (str): Name of file where processed data was cached
    """
    ## Check for Output
    if output_folder is not None:
        fname = os.path.basename(f).replace("tweets.json","processed.tweets.json")
        if not fname.endswith(".gz"):
            fname = fname + ".gz"
        output_folder = output_folder.rstrip("/")
        fname = f"{output_folder}/{fname}"
        if os.path.exists(fname):
            return fname
    ## Initalize Loader
    loader = RawDataLoader(platform="twitter",
                           random_state=None,
                           lang=None,
                           run_pipeline=True,
                           keep_retweets=keep_retweets,
                           keep_non_english=keep_non_english)
    ## Load Data
    formatted_data = loader.load_and_process(f,min_date=min_date,max_date=max_date)
    ## Dump Processed Data (or return)
    if output_folder is None:
        return formatted_data
    else:
        with gzip.open(fname, "wt", encoding="utf-8") as the_file:
            json.dump(formatted_data, the_file)
        return fname

def process_reddit_comment_file(f,
                                min_date=None,
                                max_date=None,
                                output_folder=None):
    """
    Process raw tweet data and cache in a processed form

    Args:
        f (str): Path to comment data. Expected to contain all comments
                 desired for a single individual
        min_date
        max_date
        output_folder (str): Path to output folder for caching processed
                             data. If None, returns processed data itself
    
    Returns:
        if output_folder is None:
            formatted_data (list): List of processed dictionaries
        else:
            fname (str): Name of file where processed data was cached
    """
    ## Output File
    if output_folder is not None:
        fname = os.path.basename(f).replace("comments.json","processed.comments.json")
        if not fname.endswith(".gz"):
            fname = fname + ".gz"
        output_folder = output_folder.rstrip("/")
        fname = f"{output_folder}/{fname}"
        if os.path.exists(fname):
            return fname
    ## Initalize Loader
    loader = RawDataLoader(platform="reddit",
                           random_state=None,
                           lang=None,
                           run_pipeline=True)
    ## Load Data
    formatted_data = loader.load_and_process(f,min_date=min_date,max_date=max_date)
    ## Dump Processed Data (or return)
    if output_folder is None:
        return formatted_data
    else:
        with gzip.open(fname, "wt", encoding="utf-8") as the_file:
            json.dump(formatted_data, the_file)
        return fname

def main():
    """

    """
    ## Parse Command-line Arguments
    args = parse_arguments()
    ## Identifty Input Files for Processing
    filenames = get_file_list(args)
    LOGGER.info("Found {} files for processing".format(len(filenames)))
    ## Parse Date Filters
    LOGGER.info("[Parsing Date Filters]")
    min_date = None if not args.min_date else pd.to_datetime(args.min_date)
    max_date = None if not args.max_date else pd.to_datetime(args.max_date)
    ## Identity Processor
    if args.platform == "twitter":
        mp = partial(process_tweet_file,
                     min_date=min_date,
                     max_date=max_date,
                     output_folder=args.output_folder,
                     keep_retweets=args.keep_retweets,
                     keep_non_english=args.keep_non_english)
    elif args.platform == "reddit":
        mp = partial(process_reddit_comment_file,
                     min_date=min_date,
                     max_date=max_date,
                     output_folder=args.output_folder)
    ## Process Files
    pool = Pool(args.jobs)
    LOGGER.info("Starting Preprocessing")
    res = list(tqdm(pool.imap_unordered(mp, filenames),
                    total=len(filenames),
                    desc="Processed Files",
                    file=sys.stdout))
    pool.close()
    LOGGER.info("Script Complete!")

#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()