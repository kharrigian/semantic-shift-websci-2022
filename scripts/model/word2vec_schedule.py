
"""
Train multiple word embedding models for a dataset under
a variety of samples and time periods.
"""

######################
### Configuration
######################

## Location of Repository
BASE_DIR="/export/c01/kharrigian/semantic-shift-websci-2022/"

## Desired Location of Outputs
BASE_OUTPUT_DIR = f"{BASE_DIR}/data/results/word2vec-context/" ## Base Output directory

## Baseline Flag (One of None, "file", or "post" - The latter two are for experimental purposes)
BASELINES = None

## Experiment Name
EXPERIMENT_ID = "clpsych"

## Script Parameters
DRY_RUN = False ## See how many jobs would be run by setting to True
USE_SCHEDULER = True ## If False, only generate the files passed to word2vec_train.py
RM_EXISTING = False ## Whether to remove existing output directory if it exists
SUPPORT_NUM_JOBS = 8 ## Number of jobs to use in this script when computing number of users

## CLSP Grid Parameters
GRID_USERNAME = "kharrigian"
GRID_MEMORY_REQUEST_SIZE = 32
GRID_MAX_ARRAY = 500
GRID_NUM_JOBS = 4
GRID_LOG_DIR = f"{BASE_DIR}/logs/word2vec-context/{EXPERIMENT_ID}/{BASELINES}/"

## Specify Base Config Template
BASE_WORD2VEC_CONFIG = "./configs/word2vec/embed_template.json" ## Base parameters for the word2vec Model

## Specify Dataset Parameters (e.g., whether to balance the dataset or subsample on a post level)
DATASET_PARAMETERS = {
    "dataset":"clpsych",
    "dataset-id":"clpsych",
    "target_disorder":"depression",
    "include_classes":{"control":True, "depression":True},
    "kfolds":5,
    "stratified":True,
    "test_size":0.2,
    "include": {"train":True, "dev":True, "test":True},
    "downsample":False,
    "downsample_size":None,
    "rebalance":True,
    "class_ratio":[1,1],
    "post_sampling":{
        "n_samples":None,
        "randomized":True
    }
}

## Data Splitting Parameters
SAMPLE_PROTOCOL = {
        "resample":True, ## Set this to True if you want to resample the training data
        "n_sample":5, ## Specify number of random sample if resample is True
        "test_size":0.2, ## Size of the held-out test set in each sample
        "resample_files":True ## Whether sampling is done at file-level (user-level)
}

## Temporal Boundaries (Dataset-specific)
DATE_BOUNDARIES = [
    # "2011-01-01",
    "2012-01-01",
    "2013-01-01",
    "2014-01-01",
    # "2015-01-01",
    # "2016-01-01",
    # "2017-01-01",
    # "2018-01-01",
    # "2019-01-01",
    # "2020-01-01"
]

## Support Thresholds (Set None to Use Default 100->1000)
SUPPORT_THRESHOLDS = [
    10,
    25,
    50,
    75,
    100,
    125,
    150,
    200,
    300
]

#######################
### Imports
#######################

## Standard Library
import os
import json
import subprocess
from copy import deepcopy
from textwrap import dedent
from pprint import pformat

## External Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## General Project
from semshift.model import train
from semshift.model.file_vectorizer import File2Vec
from semshift.util.logging import initialize_logger

#######################
### Globals
#######################

## Logger
LOGGER = initialize_logger()

#######################
### Helpers
#######################

def get_time_period_id(config):
    """

    """
    time_period_id = "_".join([config["date_boundaries"].get(d) for d in ["min_date","max_date"]])
    return time_period_id

def count_posts(filenames,
                config):
    """

    """
    ## Separate Configs
    dataset_config = config["datasets"][0]
    general_config = config
    ## Initialize Configuration
    f2v = File2Vec(vocab_kwargs=general_config.get("vocab_kwargs"),
                   favor_dense=False,
                   processor=None,
                   processor_kwargs={})
    ## Assign Vocabulary to Vectorizer
    f2v.vocab = f2v.vocab.assign(["<dummy>"])
    _ = f2v._initialize_dict_vectorizer()
    ## Vectorize Data
    _filenames, _, _n = f2v._vectorize_files(
                                            filenames,
                                            jobs=SUPPORT_NUM_JOBS,
                                            min_date=dataset_config.get("date_boundaries").get("min_date"),
                                            max_date=dataset_config.get("date_boundaries").get("max_date"),
                                            n_samples=None,
                                            randomized=True,
                                            return_post_counts=True,
                                            resolution=None,
                                            reference_point=None,
                                            chunk_n_samples=None,
                                            chunk_randomized=False,
                                            return_references=False,
                                            return_users=False)
    ## Format Counts
    n_posts = pd.Series(index=_filenames, data=_n)
    n_posts = n_posts.to_frame(get_time_period_id(dataset_config))
    return n_posts

def get_splits(dataset_config):
    """
    Create stream of files for a given dataset configuration.

    Args:
        dataset_config (dict)
    
    Returns:
        dataset_stream (PostStream)
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
                                          dataset_config["class_ratio"], 
                                          )
    return label_dictionaries

def get_header(log_dir,
               job_name,
               njobs=1,
               memory=8):
    """

    """
    header=f"""
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -e {log_dir}{job_name}.err
    #$ -o {log_dir}{job_name}.out
    #$ -pe smp {njobs}
    #$ -l 'gpu=0,mem_free={memory}g,ram_free={memory}g'
    """
    return header

def get_init_env(run_loc=BASE_DIR):
    """
    
    """
    init_env=f"""
    ENV="semshift"
    RUN_LOC="{run_loc}"
    """
    init_env+="""
    ## Move to Home Directory (Place Where Virtual Environments Live)
    cd /home/kharrigian/
    ## Activate Conda Environment
    source .bashrc
    conda activate ${ENV}
    ## Move To Run Directory
    cd ${RUN_LOC}
    """
    return init_env

def get_qsub_script(config_file,
                    log_dir,
                    output_dir,
                    job_name,
                    njobs,
                    memory,
                    rm_existing=True,
                    grid_max_array=100):
    """

    """
    ## Get Header
    header=get_header(log_dir=log_dir,
                      job_name=job_name,
                      njobs=njobs,
                      memory=memory)
    header = dedent(header)
    ## Get Environment Initialization
    init_env = get_init_env()
    init_env = dedent(init_env)
    ## Put Together Script
    script="""
    #!/bin/bash
    {}
    {}
    python scripts/model/word2vec_train.py {} --output_dir {} --stream_cache_dir {} {} --resample_parallel --grid_max_array {} --grid_memory_request_size {} --grid_log_dir {}
    """.format(header,
               init_env,
               config_file,
               output_dir,
               f"{output_dir}/cache/",
               "--rm_existing" if rm_existing else "",
               grid_max_array,
               memory,
               f"{output_dir}/logs/")
    script = dedent(script)
    if not USE_SCHEDULER:
        script = script.split("--resample_parallel")[0]
    script = "\n".join([i.lstrip() for i in script.split("\n")])
    return script

def main():
    """
    
    """
    ## Initialize Output Directory
    output_dir = f"{BASE_OUTPUT_DIR}/{EXPERIMENT_ID}/".replace("//","/")
    if not os.path.exists(output_dir):
        _ = os.makedirs(output_dir)
    ## Initialize Log Directory
    if not os.path.exists(GRID_LOG_DIR):
        _ = os.makedirs(GRID_LOG_DIR)
    ## Update Sample Protocol
    SAMPLE_PROTOCOL["baselines"] = BASELINES
    ## Load Base Configuration
    with open(BASE_WORD2VEC_CONFIG,"r") as the_file:
        config = json.load(the_file)
    config["experiment_name"] = EXPERIMENT_ID
    config["sample_protocol"] = SAMPLE_PROTOCOL
    config["jobs"] = GRID_NUM_JOBS
    ## Add Dataset(s) to Base Configuration
    if BASELINES is not None:
        config["datasets"] = [deepcopy(DATASET_PARAMETERS), deepcopy(DATASET_PARAMETERS)]
    else:
        config["datasets"] = [deepcopy(DATASET_PARAMETERS)]
    for d in range(len(config["datasets"])):
        config["datasets"][d]["date_boundaries"] = {}
        config["datasets"][d]["random_seed"] = config["random_seed"]
        if BASELINES is not None:
            config["datasets"][d]["dataset-id"] = "{}-{}".format(config["datasets"][d]["dataset-id"], ["cumulative","discrete"][d])
    ## Show Configuration
    LOGGER.info("Configuration:\n" + "~"*100)
    LOGGER.info(pformat(config))
    LOGGER.info("~"*100)
    ## Format Date Boundaries
    date_boundaries_sorted = [i.date().isoformat() for i in sorted(pd.to_datetime(DATE_BOUNDARIES))]
    date_boundaries_scheduler = {"discrete":[],"cumulative":[]}
    for d, (dstart, dstop) in enumerate(zip(date_boundaries_sorted[:-1], date_boundaries_sorted[1:])):
        if d > 0:
            date_boundaries_scheduler["discrete"].append((dstart, dstop))
        if dstop != date_boundaries_sorted[-1]:
            date_boundaries_scheduler["cumulative"].append((date_boundaries_sorted[0], dstop))
    ## Flatten
    if BASELINES is not None:
        date_boundaries_scheduler_flat = []
        for i, cum in enumerate(date_boundaries_scheduler["cumulative"]):
            for j, dis in enumerate(date_boundaries_scheduler["discrete"]):
                if j < i:
                    continue
                date_boundaries_scheduler_flat.append([cum, dis])
    else:
        date_boundaries_scheduler_flat = [[db] for db in date_boundaries_scheduler["discrete"] + date_boundaries_scheduler["cumulative"]]
    ## Write Configuration Files
    configuration_files = []
    configuration_dicts = {"discrete":[],"cumulative":[],"baseline-post":[],"baseline-file":[]}
    for date_bounds in date_boundaries_scheduler_flat:
        ## Make a Copy
        config_copy = deepcopy(config)
        ## Parse Flat Bounds
        if BASELINES is not None:
            prefix = "baseline-{}".format(BASELINES)
            config_copy["experiment_name"] = "{}/{}_{}-{}_{}".format(prefix, date_bounds[0][0], date_bounds[0][1], date_bounds[1][0], date_bounds[1][1])        
        else:
            prefix = "discrete" if (date_bounds[0][0], date_bounds[0][1]) in date_boundaries_scheduler["discrete"] else "cumulative"
            config_copy["experiment_name"] = "{}/{}_{}".format(prefix, date_bounds[0][0], date_bounds[0][1])
        ## Add Boundaries
        for d, db in enumerate(date_bounds):
            config_copy["datasets"][d]["date_boundaries"]["min_date"] = db[0]
            config_copy["datasets"][d]["date_boundaries"]["max_date"] = db[1]
        ## Suffix
        suffix = os.path.basename(config_copy["experiment_name"])
        ## Write file
        config_filename = f"{output_dir}/{suffix}.json".replace("//","/")
        if not DRY_RUN:
            with open(config_filename,"w") as the_file:
                json.dump(config_copy, the_file)
        configuration_files.append((config_copy["experiment_name"].replace("/","_"), config_filename))
        configuration_dicts[prefix].append(config_copy)
    ## Write Bash Scripts
    jobs = []
    for exp_id, exp_file in configuration_files:
        ## Get Script
        exp_script = get_qsub_script(config_file=exp_file,
                                     log_dir=os.path.abspath(GRID_LOG_DIR)+"/",
                                     output_dir=output_dir,
                                     job_name=exp_id,
                                     njobs=GRID_NUM_JOBS,
                                     grid_max_array=GRID_MAX_ARRAY,
                                     memory=GRID_MEMORY_REQUEST_SIZE,
                                     rm_existing=RM_EXISTING)
        ## Write Script
        exp_script_file = f"{output_dir}/{EXPERIMENT_ID}_{exp_id}.sh"
        if not DRY_RUN:
            with open(exp_script_file, "w") as the_file:
                the_file.write(exp_script)
        else:
            jobs.append("Dry Run Job ID")
            continue
        ## Schedule
        if USE_SCHEDULER:
            command = f"qsub {exp_script_file}"
            job_id = subprocess.check_output(command, shell=True)
            jobs.append(job_id)
            LOGGER.info(job_id)
    ## Update User
    if DRY_RUN:
        LOGGER.info("Found {} Jobs Total (Dry Run Only).".format(len(jobs)))
    else:
        LOGGER.info("Scheduled {} Jobs Total".format(len(jobs)))
    ## Summarization (Non-Baselines Only)
    if BASELINES is None:
        LOGGER.info("Calculating Data Support by Time Period")
        ## Get Splits and Labels for All Files
        data_splits = get_splits(configuration_dicts["cumulative"][0]["datasets"][0])
        label_dictionary = {"train":{},"test":{}}
        for split in ["train","test"]:
            for fold, fold_dict in data_splits.get(split).items():
                for group, group_dict in fold_dict.items():
                    if group_dict is not None:  
                        label_dictionary[split].update(group_dict)
        filenames = list(label_dictionary["train"].keys()) + list(label_dictionary["test"].keys())
        ## Count Posts by Time Period
        n_posts_discrete = []
        n_posts_cumulative = []
        for n_posts, configs in zip([n_posts_discrete, n_posts_cumulative],
                                    [configuration_dicts["discrete"], configuration_dicts["cumulative"]]):
            for config in configs:
                n_posts.append(count_posts(filenames, config))
        n_posts_discrete = pd.concat(n_posts_discrete, axis=1, sort=True).reindex(filenames).fillna(0).astype(int)
        n_posts_cumulative = pd.concat(n_posts_cumulative, axis=1, sort=True).reindex(filenames).fillna(0).astype(int)
        ## Cache Support
        _ = n_posts_discrete.to_csv(f"{output_dir}n_posts_discrete.csv")
        _ = n_posts_cumulative.to_csv(f"{output_dir}n_posts_cumulative.csv")
        ## Join Discrete Posts with First Cumulative Bin
        n_posts_discrete = n_posts_cumulative.iloc[:,0].to_frame().join(n_posts_discrete)
        ## Compute Data Support by Threshold
        if SUPPORT_THRESHOLDS is None:
            thresholds = np.arange(100, 1100, 100)
        else:
            thresholds = np.array(SUPPORT_THRESHOLDS)
        support = {"raw":{},"all":{}}
        for df_type, df in zip(["discrete","cumulative"],[n_posts_discrete,n_posts_cumulative]):
            df_support_raw = np.zeros((thresholds.shape[0], df.shape[1]), dtype=int)
            df_support_all = np.zeros((thresholds.shape[0], df.shape[1]), dtype=int)
            for t, threshold in enumerate(thresholds):
                df_support_raw[t] = (df >= threshold).sum(axis=0).values 
                periods_above_per_user = (df >= threshold).sum(axis=1).values
                for r in range(1,df.shape[1]+1):
                    df_support_all[t, r-1] = (periods_above_per_user >= r).sum()
            support["raw"][df_type] = pd.DataFrame(df_support_raw, index=thresholds, columns=df.columns.tolist())
            support["all"][df_type] = pd.DataFrame(df_support_all, index=thresholds, columns=list(map(str,list(range(1,df.shape[1]+1)))))
        ## Plot Support
        fig, ax = plt.subplots(2, 2, figsize=(10,5.8), sharey=True)
        for d, df_type in enumerate(["discrete","cumulative"]):
            for j, jtype in enumerate(["raw","all"]):
                ax[j, d].imshow(support[jtype][df_type].values,
                                aspect="auto",
                                cmap=plt.cm.Purples,
                                alpha=0.5,
                                interpolation="nearest")
                for r, row_data in enumerate(support[jtype][df_type].values):
                    for c, cell_data in enumerate(row_data):
                        ax[j, d].text(c,
                                    r,
                                    cell_data,
                                    fontsize=8,
                                    ha="center",
                                    va="center",
                                    color="black")
                columns_dt = list(map(lambda i: i.replace("_","-\n"), support[jtype][df_type].columns))
                ax[j,d].set_xticks(list(range(support[jtype][df_type].shape[1])))
                ax[j,d].set_xticklabels(columns_dt, rotation=45, ha="right")
                if j == 0:
                    ax[j,d].set_title(f"{df_type.title()} Time Periods", fontweight="bold", fontstyle="italic")
                ax[j,d].tick_params(labelsize=8)
        for i in range(2):
            ax[i,0].set_yticks(list(range(thresholds.shape[0])))
            ax[i,0].set_yticklabels(thresholds)
            ax[i,0].set_ylabel("Post Threshold", fontweight="bold", fontsize=12)
            ax[0,i].set_xlabel("Time Bin", fontweight="bold", fontsize=12)
            ax[1,i].set_xlabel("# Valid Time Bins at Threshold", fontweight="bold", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{output_dir}support.png", dpi=150)
        plt.close(fig)
    ## Done
    LOGGER.info("Scheduler Complete.")

#########################
### Execution
#########################

if __name__ == "__main__":
    _ = main()