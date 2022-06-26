
"""
Search for keywords/phrases in a dataset
"""

########################
### Imports
########################

## Standard Library
import os
import sys
import json
import argparse
from glob import glob
from datetime import datetime
from collections import Counter
from multiprocessing import Pool

## External
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

## Project Modules
from semshift.model import train
from semshift.model.track import Tracker
from semshift.util.helpers import chunks
from semshift.util.logging import initialize_logger
from semshift.preprocess.preprocess import RawDataLoader

########################
### Globals
########################

## Logging
LOGGER = initialize_logger()

## Identify Root Directory
ROOT_DIR = os.path.abspath(os.path.dirname(__file__) + "/../") + "/"

## Platforms
PLATFORM_MAP = {
    "twitter":[
        "clpsych",
        "clpsych_deduped",
        "multitask",
        "merged"
    ],
    "reddit":[
        "rsdd",
        "smhd",
        "wolohan"
    ]
}

########################
### Classes
########################

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

class TextStream(object):

    """

    """

    def __init__(self,
                 filenames,
                 processor,
                 processor_kwargs,
                 tracker,
                 agg_freq="day",
                 cache_rate=None,
                 random_seed=42):
        """

        """
        self._filenames = filenames
        self._processor = processor
        self._processor_kwargs = processor_kwargs
        self._tracker = tracker
        self._agg_freq = agg_freq
        self._cache_rate = cache_rate
        self._random_seed = np.random.RandomState(random_seed)

    def __repr__(self):
        """
        
        """
        return "TextStream()"
    
    def get_filenames(self):
        """

        """
        return self._filenames

    def extract_time(self,
                     dt,
                     frequency):
        """

        """
        if isinstance(dt, int):
            dt = datetime.fromtimestamp(dt)
        dt_info = [dt.year]
        if frequency in set(["hour","day","month"]):
            dt_info.append(dt.month)
        if frequency in set(["hour","day"]):
            dt_info.append(dt.day)
        if frequency in set(["hour"]):
            dt_info.append(dt.hour)
        return tuple(dt_info)

    def load_text(self,
                  filename):
        """

        """
        data = self._processor.load_and_process(filename,
                                                load_filters=["created_utc","created_at",
                                                              "author","user_id_str",
                                                              "body","text"],
                                                process_filters=["user_id_str","created_utc","text"],
                                                **self._processor_kwargs)
        return data
    
    def load_text_and_count(self,
                            filename):
        """

        """
        ## Load Data
        data = self.load_text(filename)
        ## Search For Terms
        matches = list(map(lambda i: self._tracker.search(i.get("text")), data))
        ## Extract Dates
        datetimes = list(map(lambda i: self.extract_time(i.get("created_utc"), self._agg_freq), data))
        ## Initialize Caches
        match_counts = {term:Counter() for term in self._tracker._terms}
        post_counts = Counter()
        post_cache = []
        ## Run Counter
        for match, match_dt, post in zip(matches, datetimes, data):
            ## Update Total Post Counts
            post_counts[match_dt] += 1
            ## Check for Match
            if not match:
                continue
            ## Cache Examples
            if self._cache_rate is not None and self._random_seed.uniform() <= self._cache_rate:
                post["matches"] = match
                post_cache.append(post)
            ## Get Terms
            match_terms = [m[0] for m in match]
            ## Check for Tracking Index
            for term in match_terms:
                match_counts[term][match_dt] += 1
        return match_counts, post_counts, post_cache

########################
### Functions
########################

def parse_arguments():
    """
    
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Run modeling experiments")
    ## Generic Arguments
    parser.add_argument("config_file",
                        type=str,
                        help="Path to your configuration JSON file")
    parser.add_argument("--output_dir",
                        type=str,
                        default=f"{ROOT_DIR}data/results/track/")
    parser.add_argument("--make_plots",
                        action="store_true",
                        default=False)
    parser.add_argument("--plot_top_k",
                        type=int,
                        default=30)
    parser.add_argument("--min_vis_matches",
                        type=int,
                        default=10)
    ## Parse Arguments
    args = parser.parse_args()
    return args

def load_config(config_file):
    """
    
    """
    LOGGER.info(f"Loading Config: {config_file}")
    if not os.path.exists(config_file):
        raise FileNotFoundError("Could not find configuration file.")
    with open(config_file,"r") as the_file:
        config = json.load(the_file)
    ## Assign Run Properties To Each Dataset
    for dataset in config.get("datasets",[]):
        dataset["jobs"] = config.get("jobs",1)
        dataset["random_seed"] = config.get("random_seed", 42)
    return config

def _get_annotated_stream(dataset_config,
                          tracker,
                          tracker_agg_freq,
                          cache_rate=None):
    """

    """
    ## Identify Datasets
    dataset = dataset_config["dataset"]
    ## Set Random Seed (for sampling)
    np.random.seed(dataset_config["random_seed"])
    ## Load Labels
    metadata = train.load_dataset_metadata(dataset,
                                           target_disorder=dataset_config["target_disorder"],
                                           random_state=dataset_config["random_seed"])
    ## Split Data into Folds
    label_dictionaries = train.split_data(dataset_config,
                                          metadata,
                                          dataset_config["downsample"],
                                          dataset_config["downsample_size"],
                                          dataset_config["rebalance"],
                                          dataset_config["class_ratio"],
                                          dataset_config["downsample"],
                                          dataset_config["downsample_size"],
                                          dataset_config["rebalance"],
                                          dataset_config["class_ratio"])
    ## Isolate Desired Files Based on Split
    filenames = []
    if dataset_config["include"]["train"]:
        filenames.extend([label_dictionaries["train"][fold]["train"] for fold in label_dictionaries["train"].keys()])
    if dataset_config["include"]["dev"]:
        filenames.extend([label_dictionaries["train"][fold]["dev"] for fold in label_dictionaries["train"].keys()])
    if dataset_config["include"]["test"]:
        filenames.append(label_dictionaries["test"][1]["test"])
    ## Initialize Label Filter
    label_filter = set()
    if dataset_config.get("include_classes",{}).get("control"):
        label_filter.add("control")
    if dataset_config.get("include_classes",{}).get(dataset_config.get("target_disorder")):
        label_filter.add(dataset_config.get("target_disorder"))
    ## Identify Unique Users Matching Desired Condition
    filtered_filenames = {}
    for user_dict in filenames:
        filtered_filenames.update({u:l for u, l in user_dict.items() if l in label_filter})
    filtered_filenames = list(filtered_filenames.keys())
    ## Platform ID
    platform = None
    for p, pset in PLATFORM_MAP.items():
        if dataset_config.get("dataset") in pset:
            platform = p
    if platform is None:
        raise ValueError("Dataset platform not identified.")
    ## Initialize Stream and Arguments
    processor = RawDataLoader(platform=platform,
                              random_state=dataset_config.get("random_seed",42),
                              lang=dataset_config.get("lang",None),
                              run_pipeline=False)
    ## Initialize Arguments
    processor_kwargs = {}
    for date_bound in ["min_date","max_date"]:
        if dataset_config.get("date_boundaries",{}).get(date_bound) is not None:
            processor_kwargs[date_bound] = pd.to_datetime(dataset_config["date_boundaries"][date_bound])
        else:
            processor_kwargs[date_bound] = None
    n_samples = dataset_config.get("post_sampling",{}).get("n_samples")
    if n_samples is not None and isinstance(n_samples, float):
        processor_kwargs["sample_rate"] = n_samples
    elif n_samples is not None and isinstance(n_samples, int):
        LOGGER.warning("Warning: Configuration with raw sample size per file will use first available samples in each file. Use a percentage if you want randomly distributed samples.")
        processor_kwargs["sample_size"] = n_samples
    ## Intiialize Text Stream
    text_stream = TextStream(filtered_filenames,
                             processor,
                             processor_kwargs,
                             tracker=tracker,
                             agg_freq=tracker_agg_freq,
                             cache_rate=cache_rate)
    return text_stream

def _get_custom_stream(dataset_config,
                       tracker,
                       tracker_agg_freq,
                       cache_rate=None):
    """

    """
    ## Identify Custom Datasets
    dataset = dataset_config["dataset"]
    ## Set Random Seed (for sampling)
    np.random.seed(dataset_config["random_seed"])
    ## Identify Filenames
    filenames = sorted(glob(dataset))
    if len(filenames) == 0:
        LOGGER.warning(f"No files found for the following custom dataset: '{dataset}'")
    ## Sample Filenames
    if dataset_config.get("downsample"):
        sample_n = dataset_config.get("downsample_size")
        if isinstance(sample_n, float) and sample_n < 1:
            sample_n = int(sample_n * len(filenames))
        sample_n = min(sample_n, len(filenames))
        sample_ind = np.random.choice(len(filenames),
                                      sample_n,
                                      replace=False)
        filenames = [filenames[s] for s in sorted(sample_ind)]
    ## Detect Processor Type
    processor_type = dataset_config["target_disorder"].split("custom-")[1]
    if processor_type not in set(["twitter","reddit"]):
        raise ValueError("Could not identify processor for custom dataset. Input form should be custom-<PLATFORM>")
    ## Initialize Data Processor
    processor = RawDataLoader(processor_type,
                              random_state=dataset_config.get("random_seed"),
                              lang=dataset_config.get("lang"),
                              run_pipeline=False)
    ## Initialize Processor Arguments
    processor_kwargs = {}
    for date_bound in ["min_date","max_date"]:
        if dataset_config.get("date_boundaries",{}).get(date_bound) is not None:
            processor_kwargs[date_bound] = pd.to_datetime(dataset_config["date_boundaries"][date_bound])
        else:
            processor_kwargs[date_bound] = None
    n_samples = dataset_config.get("post_sampling",{}).get("n_samples")
    if n_samples is not None and isinstance(n_samples, float):
        processor_kwargs["sample_rate"] = n_samples
    elif n_samples is not None and isinstance(n_samples, int):
        LOGGER.warning("Warning: Configuration with raw sample size per file will use first available samples in each file. Use a percentage if you want randomly distributed samples.")
        processor_kwargs["sample_size"] = n_samples
    ## Intiialize Text Stream
    text_stream = TextStream(filenames,
                             processor,
                             processor_kwargs,
                             tracker=tracker,
                             agg_freq=tracker_agg_freq,
                             cache_rate=cache_rate)
    return text_stream

def get_stream(dataset_config,
               tracker,
               tracker_agg_freq,
               cache_rate=None):
    """
    Create processor for files for a given dataset configuration.

    Args:
        dataset_config (dict)
        tracker (Tracker)
    
    Returns:
        dataset_processor (PostStream)
    """
    if dataset_config["target_disorder"].startswith("custom-"):
        dataset_stream = _get_custom_stream(dataset_config, tracker, tracker_agg_freq, cache_rate)
    else:
        dataset_stream = _get_annotated_stream(dataset_config, tracker, tracker_agg_freq, cache_rate)
    return dataset_stream

def load_keyword_file(filename):
    """

    """
    if not os.path.exists(filename):
        raise ValueError(f"Could not find keyword file: {filename}")
    terms = []
    with open(filename,"r") as the_file:
        for line in the_file:
            terms.append(line.strip())
    return terms

def plot_top_matches(match_counts,
                     top_k=30):
    """

    """
    ## Update top K
    top_k = min(top_k, match_counts.shape[1])
    ## Aggregate
    total_match_counts = match_counts.sum(axis=0).sort_values()
    total_match_counts = total_match_counts.nlargest(top_k)
    total_match_counts = total_match_counts.iloc[::-1]
    ## Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    total_match_counts.plot.barh(ax=ax,
                                 color="C0",
                                 alpha=0.4)
    ax.set_xlabel("# Matches", fontweight="bold", fontsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    return fig, ax

def plot_match_timeseries(match_counts,
                          post_counts,
                          terms):
    """

    """
    ## Format Terms
    if isinstance(terms, str):
        terms = [terms]
    ## Create Plot
    fig, ax = plt.subplots(1, 2, figsize=(10,5.8))
    for t, term in enumerate(terms):
        ax[0].plot(match_counts.index,
                   match_counts[term].values,
                   color=f"C{t}",
                   alpha=0.4,
                   marker="o",
                   linewidth=1.5,
                   label=term)
        ax[1].plot(match_counts.index,
                   match_counts[term].values / post_counts.loc[match_counts.index]["n_posts"].values,
                   color=f"C{t}",
                   alpha=0.4,
                   marker="o",
                   linewidth=1.5,
                   label=term)
    for i in range(2):
        ax[i].set_xlabel("Date",fontweight="bold",fontsize=12)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].tick_params(labelsize=12)
        ax[i].set_xlim(match_counts.index.min(), match_counts.index.max())
        ax[i].set_ylim(0)
    ax[0].set_ylabel("# Matches", fontweight="bold", fontsize=12)
    ax[1].set_ylabel("Matches Per Post", fontweight="bold", fontsize=12)
    if len(terms) > 1:
        ax[1].legend(loc="upper left",
                  frameon=False,
                  bbox_to_anchor=(1.025, 1))
    else:
        fig.suptitle(f"Query: '{terms[0]}'",
                     fontweight="bold",
                     fontstyle="italic",
                     fontsize=14)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax

def format_counts_df_dates(df):
    """

    """
    df = df.rename(columns={"level_0":"year",
                            "level_1":"month",
                            "level_2":"day",
                            "level_3":"hour"})
    for field in ["month","day","hour"]:
        if field not in df.columns:
            df[field] = {"month":1,"day":1,"hour":0}.get(field)
    df["date"] = df.apply(lambda row: datetime(*[row[i] for i in ["year","month","day","hour"]]), axis=1)
    df = df.drop(["year","month","day","hour"],axis=1).set_index("date")
    return df

def main():
    """

    """
    ## Parse Command Line
    args = parse_arguments()
    ## Load Configuration
    LOGGER.info("Loading Configuration")
    config = load_config(config_file=args.config_file)
    ## Initialize Output Directory
    output_dir = "{}/{}".format(args.output_dir, config.get("experiment_name")).replace("//","/")
    if os.path.exists(output_dir):
        LOGGER.warning(f"Warning: Output directory '{output_dir}' already exists. Consider renaming your experiment to avoid overwriting.")
    else:
        LOGGER.info(f"Initializing output directory at: {output_dir}")
        _ = os.makedirs(output_dir)
    ## Cache Config
    with open(f"{output_dir}/config.json","w") as the_file:
        json.dump(config, the_file)
    ## Initialize Tracker
    LOGGER.info("Initializing Tracker")
    tracker = Tracker(**config.get("model_kwargs"))
    for term_set in config.get("terms",[]):
        ## Identify Terms
        term_set_terms = term_set.get("terms",[])
        ## Load Terms if Desired
        if isinstance(term_set_terms,str) and term_set_terms.endswith(".keywords"):
            term_set_terms = load_keyword_file(term_set_terms)
        ## Add Terms to Tracker
        tracker = tracker.add_terms(term_set_terms,
                                    include_hashtags=term_set.get("include_hashtags",True))
    ## Initialize Global Counts
    match_counts = {term:Counter() for term in tracker._terms}
    post_counts = Counter()
    post_examples = []
    ## Process Each Dataset
    datasets = config.get("datasets",[])
    with MultiMap(config.get("jobs",1)) as mp:
        for d, dataset in enumerate(datasets):
            ## Update User
            LOGGER.info(f"Starting Dataset {d+1}/{len(datasets)}")
            ## Initialize Stream
            dataset_stream = get_stream(dataset,
                                        tracker,
                                        config.get("agg_kwargs",{}).get("frequency"),
                                        config.get("cache_rate"))
            ## Get Filenames
            dataset_filenames = dataset_stream.get_filenames()
            ## Create Chunks
            dataset_filenames_chunked = list(chunks(dataset_filenames, config.get("chunksize")))
            ## Cycle Through Chunks
            for filenames_chunk in tqdm(dataset_filenames_chunked,
                                        file=sys.stdout,
                                        desc="File Chunk",
                                        position=0,
                                        leave=True):
                matches = list(tqdm(mp.imap_unordered(dataset_stream.load_text_and_count, filenames_chunk),
                                    file=sys.stdout,
                                    desc="File",
                                    position=1,
                                    leave=False,
                                    total=len(filenames_chunk)))
                for match, dates, examples in matches:
                    post_counts += dates
                    post_examples.extend(examples)
                    for term, term_counts in match.items():
                        match_counts[term].update(term_counts)
    ## Format The Results
    match_counts = pd.DataFrame(match_counts, dtype=int).fillna(0).sort_index().astype(int)
    match_counts = match_counts.reset_index()
    post_counts = pd.Series(post_counts, dtype=int).to_frame("n_posts").sort_index().astype(int)
    post_counts = post_counts.reset_index()
    ## Format The Dates
    if match_counts.shape[0] > 0:
        match_counts = format_counts_df_dates(match_counts)
    if post_counts.shape[0] > 0:
        post_counts = format_counts_df_dates(post_counts) 
    ## Save Counts
    LOGGER.info("Saving Counts")
    _ = match_counts.to_csv(f"{output_dir}/match_counts.csv")
    _ = post_counts.to_csv(f"{output_dir}/n_posts.csv")
    ## Save Examples of Matches
    if config.get("cache_rate") is not None:
        LOGGER.info("Saving Example Matches")
        with open(f"{output_dir}/matches.json","w") as the_file:
            for example in post_examples:
                _ = the_file.write(f"{json.dumps(example)}\n")
    ## Early Exit
    if match_counts.shape[0] == 0:
        LOGGER.info("No data found. Script complete!")
        return None
    ## Check for Plots
    if args.make_plots:
        ## Summary Plot
        LOGGER.info("Plotting Top Matches")
        fig, ax = plot_top_matches(match_counts, args.plot_top_k)
        fig.savefig(f"{output_dir}/top_matches.png",dpi=150)
        plt.close(fig)
        ## Initialize Fresh Timeseries Plot Directory
        if os.path.exists(f"{output_dir}/timeseries/"):
            _ = os.system(f"rm -rf {output_dir}/timeseries/")
        _ = os.makedirs(f"{output_dir}/timeseries/")
        ## Generate Individual Plots
        for term in tqdm(match_counts.columns, desc="Generating Timeseries Plots", file=sys.stdout):
            ## Check For Total Match Criteria
            if match_counts[term].sum() < args.min_vis_matches:
                continue
            ## Format Term for Saving
            term_clean = term.replace("/","<slash>").replace(".","<period>").replace("#","<hashtag>").replace(" ","_")
            ## Create Plot
            fig, ax = plot_match_timeseries(match_counts, post_counts, [term])
            fig.savefig(f"{output_dir}/timeseries/{term_clean}.png", dpi=300)
            plt.close(fig)
    ## Done
    LOGGER.info("Script Complete!")

######################
### Execute
######################

if __name__ == "__main__":
    _ = main()