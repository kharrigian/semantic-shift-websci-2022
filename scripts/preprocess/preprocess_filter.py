
"""
Load a preprocessed data file and isolate a subset of the data
based on certain criteria (e.g. date range, user subset).

input_filepath = "./data/processed/twitter/gardenhose/2019_07_31_00_00_00.processed.9.gz"
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
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

## External Libraries
import pandas as pd
from tqdm import tqdm

## Project Libraries
from semshift.util.logging import initialize_logger
from semshift.preprocess.preprocess import RawDataLoader
from semshift.model.data_loaders import LoadProcessedData

#######################
### Globals
#######################

## Logging
LOGGER = initialize_logger()

## Loader Parameters (To Preserve Everything Else)
LOADER_KWARGS = dict(filter_negate=False,
                     filter_upper=False,
                     filter_punctuation=False,
                     filter_user_mentions=False,
                     filter_url=False,
                     filter_retweet=False,
                     filter_stopwords=False,
                     keep_pronouns=True,
                     preserve_case=True,
                     filter_empty=True,
                     emoji_handling=None,
                     filter_hashtag=False,
                     strip_hashtag=False,
                     max_tokens_per_document=None,
                     max_documents_per_user=None,
                     filter_mh_subreddits=None,
                     filter_mh_terms=None,
                     stopwords=set(),
                     random_state=42)

#######################
### Classes
#######################

class MultiMap(object):
    
    """

    """

    def __init__(self,
                 num_jobs=1):
        """

        """
        self.num_jobs = num_jobs
        if self.num_jobs > 1:
            LOGGER.info(f"Using Multiprocessing with {self.num_jobs} Jobs")
            self.pool = Pool(self.num_jobs)
        else:
            LOGGER.info("Not Using Multiprocessing")
            self.pool = None
    
    def _iterable(self,
                  func,
                  iterable):
        """

        """
        for i in iterable:
            yield func(i)
        
    def imap_unordered(self,
                       func,
                       iterable):
        """

        """
        if self.num_jobs > 1:
            res = self.pool.imap_unordered(func, iterable)
        else:
            res = self._iterable(func, iterable)
        return res

    def close(self):
        """

        """
        if self.num_jobs > 1:
            _ = self.pool.close()

#######################
### Functions
#######################

def parse_arguments():
    """

    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Filter preprocessed data.")
    ## Generic Arguments
    parser.add_argument("input_filepath",
                        type=str,
                        help="Glob-supported filename path.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Path where each input data file should be saved.")
    parser.add_argument("--platform",
                        choices=["twitter","reddit"])
    parser.add_argument("--user_list",
                        type=str,
                        default=None,
                        nargs="*",
                        help="Glob supported newline delimited txt files containing user_id_str attributes to keep")
    parser.add_argument("--min_date",
                        type=str,
                        default=None,
                        help="Lower date boundary if desired.")
    parser.add_argument("--max_date",
                        type=str,
                        default=None,
                        help="Upper date boundary if desired.")
    parser.add_argument("--lang",
                        nargs="*",
                        type=str,
                        help="Languages to keep (based on langid)")
    parser.add_argument("--jobs",
                        default=1,
                        type=int,
                        help="Number of processes to use.")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Inputs
    if args.output_dir is None:
        raise ValueError("Need to specify an output directory.")
    return args

def get_filenames(input_filepath):
    """

    """
    filenames = sorted(glob(input_filepath))
    if len(filenames) == 0:
        raise FileNotFoundError("No files identified based on input filepath.")
    return filenames

def get_user_subset(user_list):
    """

    """
    ## Early Return
    if user_list is None:
        return None
    ## Get Filenames
    user_list_files = []
    for ul in user_list:
        user_list_files.extend(glob(ul))
    if len(user_list_files) == 0:
        raise FileNotFoundError("Could not identify any of the specified user lists.")
    ## Load Users
    users = set()
    for ulf in user_list_files:
        with open(ulf,"r") as the_file:
            for line in the_file:
                users.add(line.strip())
    return users

def load_and_filter(filename,
                    output_dir=None,
                    user_subset=None,
                    processor=None,
                    processor_kwargs={},
                    processed_data_loader=None):
    """

    """
    ## Format Load Data
    data = processed_data_loader.load_user_data(filename,
                                                processor=processor,
                                                processor_kwargs=processor_kwargs,
                                                ignore_text_filter=False,
                                                ignore_sentence_filter=False)
    ## Apply User Filter
    if user_subset:
        data = list(filter(lambda d: d["user_id_str"] in user_subset, data))
    ## Write New File
    base_filename = os.path.basename(filename)
    output_filename = f"{output_dir}/{base_filename}".replace("//","/")
    if os.path.exists(output_filename):
        LOGGER.warning(f"Warning: Overwriting existing filename {output_filename}")
    with gzip.open(output_filename,"wt") as the_file:
        for row in data:
            the_file.write(f"{json.dumps(row)}\n")
    return output_filename

def main():
    """

    """
    ## Parse Arguments
    LOGGER.info("[Parsing Command Line]")
    args = parse_arguments()
    ## Create Output Directory
    LOGGER.info("[Initializing Output Directory]")
    if not os.path.exists(args.output_dir):
        _ = os.makedirs(args.output_dir)
    ## Get Filenames
    LOGGER.info("[Identifying Processed Data Files]")
    filenames = get_filenames(args.input_filepath)
    ## Parse Date Filters
    LOGGER.info("[Parsing Date Filters]")
    min_date = None if not args.min_date else pd.to_datetime(args.min_date)
    max_date = None if not args.max_date else pd.to_datetime(args.max_date)
    ## Parse User Filter
    LOGGER.info("[Identifying User Subset]")
    user_list = get_user_subset(args.user_list)
    ## Initialize Raw Data Loader
    LOGGER.info("[Initializing Raw Data Loader]")
    raw_loader = RawDataLoader(platform=args.platform,
                               lang=args.lang,
                               run_pipeline=False,
                               random_state=None)
    raw_loader_kwargs = {"min_date":min_date,"max_date":max_date}
    ## Initialize Processed Data Loader
    LOGGER.info("[Initializing Processed Data Loader]")
    processed_loader_args = deepcopy(LOADER_KWARGS)
    processed_loader_args["lang"] = args.lang
    processed_loader = LoadProcessedData(**processed_loader_args)
    ## Initialize Processing Function
    processing_function = partial(load_and_filter,
                                  output_dir=args.output_dir,
                                  user_subset=user_list,
                                  processor=raw_loader,
                                  processor_kwargs=raw_loader_kwargs,
                                  processed_data_loader=processed_loader)
    ## Execute Processing
    LOGGER.info("[Beginning Data Filtering]")
    with MultiMap(args.jobs) as mp:
        processed_filenames = list(tqdm(mp.imap_unordered(processing_function, filenames), 
                                        total=len(filenames),
                                        file=sys.stdout,
                                        desc="Filtering"))
    ## Done
    LOGGER.info("[Script Complete!]")

#####################
### Execution
#####################

if __name__ == "__main__":
    _ = main()