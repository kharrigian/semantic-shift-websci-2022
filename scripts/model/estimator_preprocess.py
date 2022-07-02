
"""
Prepare data for estimation. Primarily designed to filter large, unlabeled Twitter and Reddit datasets
so that they are in a manageable and workable state. In general, can be applied to any preprocessed
dataset.
"""

#############################
### Imports
#############################

## Standard Library
import os
import sys
import json
import gzip
import argparse
import subprocess
from glob import glob
from time import sleep
from multiprocessing import Pool
from collections import Counter
from functools import partial
from textwrap import dedent

## External Libraries
import numpy as np
import pandas as pd
from langid import langid
from tqdm import tqdm
from scipy import sparse

## Private
from semshift.util.helpers import flatten, chunks
from semshift.util.logging import initialize_logger
from semshift.model.embed import Word2Vec
from semshift.model.data_loaders import PostStream
from semshift.model.train import load_dataset_metadata, _rebalance

## Local Project
_ = sys.path.append(os.path.abspath(os.path.dirname(__file__))+"/")
from helpers import DVec

#############################
### Globals
#############################

## Logging
LOGGER = initialize_logger()

## Language ID
_ = langid.load_model()

#############################
### Parallel Computing
#############################

def get_header(nstart,
               nend,
               log_dir,
               memory=8,
               num_jobs=1,
               max_tasks=10):
    """

    """
    header=f"""
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -N estimator_preprocess
    #$ -t {nstart}-{nend}
    #$ -e {log_dir}
    #$ -o {log_dir}
    #$ -tc {max_tasks}
    #$ -pe smp {num_jobs}
    #$ -l 'gpu=0,mem_free={memory}g,ram_free={memory}g'
    """
    header = dedent(header)
    return header

def get_init_env():
    """
    
    """
    init_env=f"""
    ENV="semshift"
    RUN_LOC="/export/c01/kharrigian/semantic-shift-websci-2022/"
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
    init_env = dedent(init_env)
    return init_env

def format_parallel_script_counter(lower_bound,
                                   upper_bound,
                                   config,
                                   memory=8,
                                   num_jobs=1,
                                   max_tasks_concurrent=8,
                                   skip_existing=False):
    """

    """
    ## Paths
    config_filename = "{}/config.json".format(config["output"])
    log_dir = "{}/logs/counter/".format(config["output"])
    if not os.path.exists(log_dir):
        _ = os.makedirs(log_dir)
    ## Construct Script
    header = get_header(lower_bound,
                        upper_bound,
                        log_dir=log_dir,
                        memory=memory,
                        num_jobs=num_jobs,
                        max_tasks=max_tasks_concurrent)
    init = get_init_env()
    script="""
    #!/bin/bash
    {}
    {}
    python scripts/experiments/estimator_preprocess.py {} --counter parallel --counter_id $SGE_TASK_ID --jobs {} {}
    """.format(header, init, config_filename, num_jobs, "--counter_skip_existing" if skip_existing else "")
    script = dedent(script)
    script = script.replace("//","/")
    return script

def format_parallel_script_filter(lower_bound,
                                  upper_bound,
                                  config,
                                  memory=8,
                                  num_jobs=1,
                                  max_tasks_concurrent=8,
                                  skip_existing=False):
    """

    """
    ## Paths
    config_filename = "{}/config.json".format(config["output"])
    log_dir = "{}/logs/filter/".format(config["output"])
    if not os.path.exists(log_dir):
        _ = os.makedirs(log_dir)
    ## Construct Script
    header = get_header(lower_bound,
                        upper_bound,
                        log_dir=log_dir,
                        memory=memory,
                        num_jobs=num_jobs,
                        max_tasks=max_tasks_concurrent)
    init = get_init_env()
    script="""
    #!/bin/bash
    {}
    {}
    python scripts/experiments/estimator_preprocess.py {} --filter_users parallel --filter_id $SGE_TASK_ID --jobs {} {}
    """.format(header, init, config_filename, num_jobs, "--filter_skip_existing" if skip_existing else "")
    script = dedent(script)
    script = script.replace("//","/")
    return script

def format_parallel_script_vectorize(lower_bound,
                                     upper_bound,
                                     config,
                                     memory=8,
                                     num_jobs=1,
                                     max_tasks_concurrent=8,
                                     skip_existing=False):
    """

    """
    ## Paths
    config_filename = "{}/config.json".format(config["output"])
    log_dir = "{}/logs/vectorize/".format(config["output"])
    if not os.path.exists(log_dir):
        _ = os.makedirs(log_dir)
    ## Construct Script
    header = get_header(lower_bound,
                        upper_bound,
                        log_dir=log_dir,
                        memory=memory,
                        num_jobs=num_jobs,
                        max_tasks=max_tasks_concurrent)
    init = get_init_env()
    script="""
    #!/bin/bash
    {}
    {}
    python scripts/experiments/estimator_preprocess.py {} --vectorize_users parallel --vectorize_id $SGE_TASK_ID --jobs {} {}
    """.format(header, init, config_filename, num_jobs, "--vectorize_skip_existing" if skip_existing else "")
    script = dedent(script)
    script = script.replace("//","/")
    return script

def schedule_counter(config,
                     jobs=1,
                     skip_existing=False,
                     memory_per_job=8,
                     max_array_size=500,
                     max_tasks_concurrent=8):
    """
    
    """
    ## Array Bounds
    filenames = get_filenames(config)
    filenames_bounds = [(i[0], i[-1]) for i in chunks(list(range(1, len(filenames)+1)), max_array_size)]
    n_sample = len(filenames_bounds)
    ## Create Job Directory
    job_dir = "{}/scheduler/".format(config["output"])
    if not os.path.exists(job_dir):
        _ = os.makedirs(job_dir)
    ## Create Scripts
    job_files = []
    for lower, upper in filenames_bounds:
        ## Generate Script
        script = format_parallel_script_counter(lower_bound=lower,
                                                upper_bound=upper,
                                                config=config,
                                                memory=memory_per_job,
                                                num_jobs=jobs,
                                                max_tasks_concurrent=max_tasks_concurrent,
                                                skip_existing=skip_existing)
        ## Write Script File
        script_file = f"{job_dir}/counter_{lower}_{upper}.sh".replace("//","/")
        with open(script_file,"w") as the_file:
            the_file.write(script)
        ## Cache
        job_files.append(script_file)
    ## Schedule Jobs
    LOGGER.info(f"[Scheduling {len(job_files)} Job Arrays for {n_sample} Samples]")
    for job_file in job_files:
        command = f"qsub {job_file}"
        job_id = subprocess.check_output(command, shell=True)
        LOGGER.info(job_id)

def schedule_filter(config,
                    jobs=1,
                    skip_existing=False,
                    memory_per_job=8,
                    max_array_size=500,
                    max_tasks_concurrent=8):
    """
    
    """
    ## Array Bounds
    filenames = sorted(glob("{}/chunks/*.txt".format(config["output"])))
    filenames_bounds = [(i[0], i[-1]) for i in chunks(list(range(1, len(filenames)+1)), max_array_size)]
    n_sample = len(filenames_bounds)
    ## Create Job Directory
    job_dir = "{}/scheduler/".format(config["output"])
    if not os.path.exists(job_dir):
        _ = os.makedirs(job_dir)
    ## Create Scripts
    job_files = []
    for lower, upper in filenames_bounds:
        ## Generate Script
        script = format_parallel_script_filter(lower_bound=lower,
                                               upper_bound=upper,
                                               config=config,
                                               memory=memory_per_job,
                                               num_jobs=jobs,
                                               max_tasks_concurrent=max_tasks_concurrent,
                                               skip_existing=skip_existing)
        ## Write Script File
        script_file = f"{job_dir}/filter_{lower}_{upper}.sh".replace("//","/")
        with open(script_file,"w") as the_file:
            the_file.write(script)
        ## Cache
        job_files.append(script_file)
    ## Schedule Jobs
    LOGGER.info(f"[Scheduling {len(job_files)} Job Arrays for {n_sample} Samples]")
    for job_file in job_files:
        command = f"qsub {job_file}"
        job_id = subprocess.check_output(command, shell=True)
        LOGGER.info(job_id)
    
def schedule_vectorization(config,
                           jobs=1,
                           skip_existing=False,
                           memory_per_job=4,
                           max_array_size=500,
                           max_tasks_concurrent=8):
    """
    
    """
    ## Array Bounds
    filenames = sorted(glob("{}/filtered/*.json.gz".format(config["output"])))
    filenames_bounds = [(i[0], i[-1]) for i in chunks(list(range(1, len(filenames)+1)), max_array_size)]
    n_sample = len(filenames_bounds)
    ## Create Job Directory
    job_dir = "{}/scheduler/".format(config["output"])
    if not os.path.exists(job_dir):
        _ = os.makedirs(job_dir)
    ## Create Scripts
    job_files = []
    for lower, upper in filenames_bounds:
        ## Generate Script
        script = format_parallel_script_vectorize(lower_bound=lower,
                                                  upper_bound=upper,
                                                  config=config,
                                                  memory=memory_per_job,
                                                  num_jobs=jobs,
                                                  max_tasks_concurrent=max_tasks_concurrent,
                                                  skip_existing=skip_existing)
        ## Write Script File
        script_file = f"{job_dir}/vectorize_{lower}_{upper}.sh".replace("//","/")
        with open(script_file,"w") as the_file:
            the_file.write(script)
        ## Cache
        job_files.append(script_file)
    ## Schedule Jobs
    LOGGER.info(f"[Scheduling {len(job_files)} Job Arrays for {n_sample} Samples]")
    for job_file in job_files:
        command = f"qsub {job_file}"
        job_id = subprocess.check_output(command, shell=True)
        LOGGER.info(job_id)

#############################
### Functions
#############################

def parse_command_line():
    """
    
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("config",
                            type=str)
    _ = parser.add_argument("--counter",
                            type=str,
                            choices=["serial","parallel"],
                            default=None)
    _ = parser.add_argument("--counter_id",
                            type=int,
                            default=None)
    _ = parser.add_argument("--counter_skip_existing",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--analyze_counts",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--chunk_users",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--chunk_users_dryrun",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--filter_users",
                            type=str,
                            choices=["serial","parallel"],
                            default=None)
    _ = parser.add_argument("--filter_id",
                            type=int,
                            default=None)
    _ = parser.add_argument("--filter_skip_existing",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--vectorize_users",
                            type=str,
                            choices=["serial","parallel"],
                            default=None)
    _ = parser.add_argument("--vectorize_id",
                            type=int,
                            default=None)
    _ = parser.add_argument("--vectorize_skip_existing",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--symlink",
                            type=str,
                            default=None,
                            help="Path to root directory to symlink.")
    _ = parser.add_argument("--symlink_items",
                            nargs="*",
                            type=str,
                            default=None)
    _ = parser.add_argument("--jobs",
                            type=int,
                            default=1)
    _ = parser.add_argument("--grid_max_tasks_concurrent",
                            default=8,
                            type=int)
    _ = parser.add_argument("--grid_max_array_size",
                            default=500,
                            type=int)
    _ = parser.add_argument("--grid_memory_per_job",
                            default=4,
                            type=int)
    args = parser.parse_args()
    return args

def load_configuration(filename):
    """
    
    """
    ## Check and Load Config File
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Config File not Found: {filename}")
    with open(filename, "r") as the_file:
        config = json.load(the_file)
    ## Initialize Output Directory
    if not os.path.exists(config["output"]):
        try:
            _ = os.makedirs(config["output"])
        except FileExistsError:
            pass
    ## Identify Config File and Existing Config
    config_file = "{}/config.json".format(config["output"])
    existing_config = None
    if os.path.exists(config_file):
        with open(config_file,"r") as the_file:
            existing_config = json.load(the_file)
    ## Cache
    if existing_config is not None and existing_config == config:
        LOGGER.info("[Configuration matches existing config file.]")
    else:
        ## Alert User
        if existing_config is not None:
            LOGGER.info("[Configuration file does not match existing configuration. Overwriting.]")
        ## Dump
        with open("{}/config.json".format(config["output"]),"w") as the_file:
            json.dump(config, the_file, indent=4, sort_keys=False)
    return config

def get_filenames(config):
    """
    
    """
    filenames = []
    for inp in config["inputs"]:
        if inp.startswith("labeled-"):
            ## Load
            inp_meta = load_dataset_metadata(inp.split("labeled-")[1], "depression", 42)
            ## Rebalance (Current Default in Experiments)
            inp_meta = _rebalance(inp_meta, "depression", [1, 1], 42)
            ## Get Files from DataFrame
            inp_files = inp_meta["source"].tolist()
        else:
            ## Get Files from Disk
            inp_files = glob(inp)
        filenames.extend(inp_files)
    filenames = sorted(filenames)
    return filenames
        
def _instance_date_boundary(instance_timestamp,
                            date_boundaries):
    """
    
    """
    if instance_timestamp is None:
        raise TypeError("Timestamp should not be a null value.")
    for d, (db_min, db_max) in enumerate(date_boundaries):
        if instance_timestamp >= db_min and instance_timestamp <= db_max:
            return d
    return None

def _is_retweet(text):
    """
    
    """
    if text is None:
        raise TypeError("Text should not be a null value.")
    is_rt = text.startswith("RT @")
    return is_rt

def _classify_languages(text,
                        max_chunksize=250):
    """
    
    """
    ## Input Check
    if len(text) == 0:
        return []
    ## Format As List
    is_singular = False
    if isinstance(text, str):
        is_singular = True
        text = [text]
    ## Initialize Result Cache
    all_preds = []
    all_conf = []
    for text_chunk in chunks(text, max_chunksize):
        ## Get Feature Space
        features = np.vstack(list(map(langid.identifier.instance2fv, text_chunk)))
        ## Get Probs
        probs = langid.identifier.nb_classprobs(features)
        cl = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        preds = [langid.identifier.nb_classes[c] for c in cl]
        ## Cache
        all_preds.extend(preds)
        all_conf.extend(conf)
    ## Format
    results = list(zip(all_preds, all_conf))
    if is_singular:
        results = results[0]
    return results

def _filter_by_language(data,
                        lang):
    """
    
    """
    ## Run Classification
    language_ids = _classify_languages([i.get("text") for i in data])
    ## Filter
    data = [lc for lc, li in zip(data, language_ids) if li[0] in lang]
    return data

def _check_language_and_count(language_check_cache, lang, counts):
    """
    
    """
    ## filter
    language_check_cache = _filter_by_language(language_check_cache, lang)
    ## Cache Filtered Data
    for lc_instance in language_check_cache:
        counts[lc_instance["instance_db"]][lc_instance["user_id_str"]] += 1
    return counts, []

def _count_users_in_file(fid_filename,
                         date_boundaries,
                         lang,
                         include_retweets,
                         counts_cache_dir,
                         skip_existing=True,
                         check_every=1000):
    """
    
    """
    ## Update Lang
    if lang is not None:
        if isinstance(lang, str):
            lang = [lang]
        lang = set(lang)
    ## Parse Input
    fid, filename = fid_filename
    ## Check to See If Output Exists Already
    if skip_existing:
        all_exists = True
        for d in range(len(date_boundaries)):
            d = f"{counts_cache_dir}/{d}/{fid}.json".replace("//","/")
            if not os.path.exists(d):
                all_exists = False
        if all_exists:
            return None
    ## Count Cache
    counts = [Counter() for _ in date_boundaries]
    ## Initialize Language Cache
    language_check_cache = []
    ## Iterate Through Data
    with gzip.open(filename, "r") as the_file:
        for line in the_file:
            ## Parse the Instance
            line_data = json.loads(line)
            ## Format Type (If File Formatted as New Lines instead of a List)
            if isinstance(line_data, dict):
                line_data = [line_data]
            ## Iterate Over Instances
            for instance in line_data:
                ## Timestamp Check
                instance_db = _instance_date_boundary(instance.get("created_utc"), date_boundaries)
                if instance_db is None:
                    continue
                ## Retweet Check
                if not include_retweets and _is_retweet(instance.get("text")):
                    continue
                ## Optionally (Cache without Language Check)
                if lang is None or len(lang) == 0:
                    counts[instance_db][instance["user_id_str"]] += 1
                    continue
                ## Cache for Future Language Check
                language_check_cache.append({
                    "user_id_str":instance.get("user_id_str"),
                    "text":instance.get("text"),
                    "instance_db":instance_db
                })
                ## Check Language Cache
                if len(language_check_cache) == check_every:
                    ## Run Check and Update Count
                    counts, language_check_cache = _check_language_and_count(language_check_cache, lang, counts)
    ## Final Language Check
    if len(language_check_cache) > 0:
        ## Run Check and Update Counts
        counts, _ = _check_language_and_count(language_check_cache, lang, counts)
    ## Write Counts
    for d, cd in enumerate(counts):
        ## Check
        cd_file = f"{counts_cache_dir}/{d}/{fid}.json".replace("//","/")
        cur_attempt = 0
        while True:
            try:
                with open(cd_file,"w") as the_file:
                    json.dump(cd, the_file)
                break
            except:
                cur_attempt += 1
                _ = sleep(5)
            if cur_attempt == 3:
                raise ValueError("File Save Failed.")

def count_users(config,
                jobs=1,
                file_id=None,
                skip_existing=False):
    """
    
    """
    ## Identify Filenames
    filenames = get_filenames(config)
    if len(filenames) == 0:
        raise FileNotFoundError("Input files do not exist.")
    ## Date Boundaries
    date_boundaries = [(int(pd.to_datetime(i["min_date"]).timestamp()), int(pd.to_datetime(i["max_date"]).timestamp())) for i in config["cache_user_criteria"]["date_boundaries"]]
    date_boundaries = sorted(date_boundaries, key=lambda x: x[0])
    ## Cache Directory (Divided by Date Boundary)
    counts_cache_dir = "{}/counts/".format(config["output"]).replace("//","/")
    for d in range(len(date_boundaries)):
        d_counts_cache_dir = f"{counts_cache_dir}/{d}/".replace("//","/")
        if not os.path.exists(d_counts_cache_dir):
            try:
                _ = os.makedirs(d_counts_cache_dir)
            except FileExistsError:
                pass
    ## Compare Filenames
    filenames_cache = f"{counts_cache_dir}/filenames.txt"
    if os.path.exists(filenames_cache):
        ## Load Existing Filenames
        with open(filenames_cache,"r") as the_file:
            existing_filenames = [i.strip() for i in the_file]
        ## Remove Filenames if Not Equivalent
        if filenames != existing_filenames:
            LOGGER.warning("Warning: Existing set of filenames does match new set of filenames. Removing old set.")
            _ = os.system(f"rm -rf {counts_cache_dir}/*/*.json")
            files_updated = True
        else:
            files_updated = False

    else:
        files_updated = True
    ## Cache Filenames
    if files_updated:
        with open(filenames_cache,"w") as the_file:
            for fn in filenames:
                the_file.write(f"{fn}\n")    
    ## Format Inputs
    if file_id is not None:
        filename_inputs = [(file_id - 1, filenames[file_id-1])]
    else:
        filename_inputs = list(enumerate(filenames))
    ## Parameterize Multiprocessor
    counter = partial(_count_users_in_file,
                      date_boundaries=date_boundaries,
                      lang=config["cache_user_criteria"]["lang"],
                      include_retweets=config["cache_user_criteria"]["include_retweets"],
                      counts_cache_dir=counts_cache_dir,
                      skip_existing=skip_existing)
    ## Run Counting Procedure
    if len(filename_inputs) > 1 and jobs > 1:
        with Pool(jobs) as mp:
            _ = list(tqdm(mp.imap_unordered(counter, filename_inputs), total=len(filename_inputs), desc="[Counting User Activity]", file=sys.stdout))
    else:
        for fi in tqdm(filename_inputs, total=len(filename_inputs), desc="[Counting User Activity]", file=sys.stdout):
            _ = counter(fi)
        
def _load_counts(count_file):
    """

    """
    ## Load File
    with open(count_file,"r") as the_file:
        cf_data = json.load(the_file)
    ## Check Length
    if len(cf_data) == 0:
        return None, None
    ## Parse and Update
    cf_db = int(os.path.dirname(count_file).split("/")[-1])
    cf_data = Counter(cf_data)
    return cf_db, cf_data

def load_counts(config,
                jobs=1):
    """
    
    """
    ## Initialze Count Cache
    ndb = len(config["cache_user_criteria"]["date_boundaries"])
    counts = [Counter() for _ in range(ndb)]
    ## Count Files
    count_files = glob("{}/counts/*/*.json".format(config["output"]))
    with Pool(jobs) as mp:
        for cf_db, cf_data in mp.imap_unordered(_load_counts, tqdm(count_files, desc="[Loading Counts]", file=sys.stdout, total=len(count_files))):
            if cf_db is None:
                continue
            counts[cf_db].update(cf_data)
    return counts
        
def analyze_counts(config,
                   thresholds=[1, 5, 25, 50, 100, 150, 200, 250, 500, 1000],
                   jobs=1):
    """
    
    """
    ## Load Counts
    counts = load_counts(config, jobs=jobs)
    ## Count Totals
    for tp, tp_counts in enumerate(counts):
        LOGGER.info("Time Period {}: {:,d} Users, {:,d} Posts".format(tp, len(tp_counts), sum(tp_counts.values())))
    ## Threshold Analysis
    X = np.zeros((len(thresholds), len(counts) + 2), dtype=int)
    X_posts = np.zeros_like(X)
    for t, threshold in enumerate(thresholds):
        tusers = []
        for db, db_counts in enumerate(counts):
            db_t_users = [x for x, y in db_counts.items() if y >= threshold]
            db_t_posts = sum(db_counts[uu] for uu in db_t_users)
            tusers.append(set(db_t_users))
            X[t, db] = len(db_t_users)
            X_posts[t, db] = db_t_posts
        ## Union and Intersection
        uni_users = set.union(*tusers)
        int_users = set.intersection(*tusers)
        ## Aggregate Number of Posts
        counts_agg = [Counter({x:y for x, y in tp_counts.items() if x in uni_users}) for tp_counts in counts]
        counts_agg = sum(counts_agg, Counter())
        ## Cache
        X[t, -2] = len(uni_users)
        X[t, -1] = len(int_users)
        X_posts[t, -2] = sum(counts_agg[uu] for uu in uni_users)
        X_posts[t, -1] = sum(counts_agg[uu] for uu in int_users)
    ## Format
    X = pd.DataFrame(X, index=thresholds, columns=[i for i in range(len(counts))] + ["union","intersection"])
    X_posts = pd.DataFrame(X_posts, index=thresholds, columns=[i for i in range(len(counts))] + ["union","intersection"])
    X.index.name = "threshold"
    X_posts.index.name = "threshold"
    ## Cache
    _ = X.to_csv("{}/support.users.csv".format(config["output"]))
    _ = X_posts.to_csv("{}/support.posts.csv".format(config["output"]))
    return X, X_posts

def chunk_users(config,
                dryrun,
                jobs=1):
    """
    
    """
    ## Load Counts
    counts = load_counts(config, jobs=jobs)
    ## Get Users Which Meet Criteria
    LOGGER.info("[Filtering Users Meeting Criteria]")
    threshold = config["cache_user_criteria"]["min_posts_per_date_boundary"]
    counts_users = []
    for db, db_counts in enumerate(counts):
        db_t_users = [x for x, y in db_counts.items() if y >= threshold]
        counts_users.append(set(db_t_users))
    ## Intersection or Union
    LOGGER.info("[Merging Users Across Time Periods]")
    if config["cache_user_criteria"]["intersection"]:
        counts_users = set.intersection(*counts_users)
    else:
        counts_users = set.union(*counts_users)
    ## Flatten Counts and Aggregate Over Time Periods to Estimate Number of Files
    counts = [Counter({x:y for x, y in tp_counts.items() if x in counts_users}) for tp_counts in counts]
    counts_agg = sum(counts, Counter())
    ## Sort When Performing Real Chunking
    if dryrun:
        counts_users = list(counts_users)
    else:
        counts_users = sorted(list(counts_users))
    LOGGER.info("{} Users Meet Criteria".format(len(counts_users)))
    ## Chunk Sizes
    user_chunksize = config["cache_kwargs"]["chunksize"]
    within_user_chunksize = config["cache_kwargs"]["within_chunk_chunksize"]
    ## Chunks
    counts_users_chunks = list(chunks(counts_users, user_chunksize))
    approx_file_count = sum([sum([counts_agg[u] for u in uchunk]) // within_user_chunksize for uchunk in counts_users_chunks])
    LOGGER.info("With Chunksize of {}, This Will Create {} Chunks".format(user_chunksize, len(counts_users_chunks)))
    LOGGER.info("With Within-Chunk Chunk Limit of {}, This Will Create Approximately {} Files".format(within_user_chunksize, approx_file_count))
    ## Early Exit
    if dryrun:
        return None
    ## Initialize Chunk Directory
    chunks_dir = "{}/chunks/".format(config["output"])
    if os.path.exists(chunks_dir):
        LOGGER.info("Chunk Directory Already Exists. Overwriting.")
        _ = os.system("rm -rf {}".format(chunks_dir))
    _ = os.makedirs(chunks_dir)
    ## Write User Chunks
    for c, users in enumerate(counts_users_chunks):
        with open(f"{chunks_dir}/{c}.txt","w") as the_file:
            for user_id in users:
                the_file.write(f"{user_id}\n")

def _extract_filtered_users(input_filename,
                            usernames,
                            min_date=None,
                            max_date=None,
                            lang=None,
                            include_retweets=True,
                            user_per_file=False,
                            sample_rate=None,
                            random_state=42):
    """
    
    """
    ## Seed
    seed = np.random.RandomState(random_state)
    ## Initialize Output Cache
    data_cache = []
    ## Process File
    opener = gzip.open if input_filename.endswith(".gz") else open
    with opener(input_filename,"r") as the_file:
        for line in the_file:
            ## Downsampling (if Desired)
            if sample_rate is not None and seed.rand() > sample_rate:
                continue
            ## Load Line
            line_data = json.loads(line)
            ## Check Type
            if isinstance(line_data, dict):
                line_data = [line_data]
            ## Iterate within
            exit_loop = False
            for instance in line_data:
                ## Date Check
                if min_date is not None and instance["created_utc"] < min_date:
                    continue
                if max_date is not None and instance["created_utc"] > max_date:
                    continue
                ## User Check
                if instance["user_id_str"] not in usernames:
                    ## File Already Based on User, Can Skip Remaining Posts
                    if user_per_file:
                        exit_loop = True
                        break
                    continue
                ## Retweet Check
                if not include_retweets and _is_retweet(instance["text"]):
                    continue
                ## Cache
                data_cache.append(instance)
            ## Early Exit
            if exit_loop:
                break
    ## Language Filter
    if len(data_cache) > 0:
        data_cache = _filter_by_language(data=data_cache, lang=lang)
    ## Return
    return data_cache

def _filter_users_in_chunk(chunk_file,
                           config,
                           jobs=1,
                           skip_existing=False):
    """
    
    """
    ## File Interpretation
    chunk_id = os.path.basename(chunk_file).split(".txt")[0]
    ## Initialize Completion Directory
    filter_completion_dir = "{}/filtered/done/".format(config["output"])
    if not os.path.exists(filter_completion_dir):
        try:
            _ = os.makedirs(filter_completion_dir)
        except FileExistsError:
            pass
    ## Check for Existing
    filter_outfile = "{}/filtered/{}.json.gz".format(config["output"], chunk_id)
    done_outfile = "{}/filtered/done/{}.txt".format(config["output"], chunk_id)
    if skip_existing and os.path.exists(done_outfile):
        LOGGER.info("[User Chunk Already Exists. Skipping.]")
        return None
    ## Identify Raw Files
    input_filenames = get_filenames(config)
    ## Load Usernames in Chunk
    chunk_users = set()
    with open(chunk_file,"r") as the_file:
        for line in the_file:
            chunk_users.add(line.strip())
    ## Initialize Helper
    filter_func = partial(_extract_filtered_users,
                          usernames=chunk_users,
                          min_date=int(pd.to_datetime(config["cache_kwargs"]["date_boundaries"]["min_date"]).timestamp()),
                          max_date=int(pd.to_datetime(config["cache_kwargs"]["date_boundaries"]["max_date"]).timestamp()),
                          lang=config["cache_user_criteria"]["lang"],
                          include_retweets=config["cache_user_criteria"]["include_retweets"],
                          user_per_file=config["inputs_metadata"]["user_per_file"],
                          sample_rate=config["cache_kwargs"]["sample_rate"],
                          random_state=config["random_state"])
    ## Parallel Processing
    data_cache = []
    with Pool(jobs) as mp:
        for result in mp.imap_unordered(filter_func, tqdm(input_filenames, desc="[Filter]", total=len(input_filenames), file=sys.stdout)):
            if len(result) > 0:
                data_cache.extend(result)
    ## Sort (Size -> User ID -> Timestamp)
    LOGGER.info("[Sorting Result]")
    user_counts = Counter([x["user_id_str"] for x in data_cache])
    data_cache = sorted(data_cache, key=lambda x: (user_counts[x["user_id_str"]], x["user_id_str"], x["created_utc"]))
    ## Break Up Indices
    LOGGER.info("[Establishing Chunk Spans]")
    cache_breaks = [0]
    cur_length = 0
    cur_user = data_cache[0]["user_id_str"]
    max_posts_per_chunk = config["cache_kwargs"]["within_chunk_chunksize"]
    for ind, dc in enumerate(data_cache):
        cur_length += 1
        if cur_length >= max_posts_per_chunk and dc["user_id_str"] != cur_user:
            cache_breaks.append(ind)
            cur_user = dc["user_id_str"]
            cur_length = 0
        elif cur_length < max_posts_per_chunk and dc["user_id_str"] != cur_user:
            cur_user = dc["user_id_str"]
    cache_breaks.append(ind + 1)
    ## Cache in Chunks
    user2chunk = {}
    n_written = 0
    for i, (x, y) in tqdm(enumerate(zip(cache_breaks[:-1], cache_breaks[1:])), total=len(cache_breaks)-1, desc="[Caching]", file=sys.stdout):
        ## Update Filename
        chunk_filter_outfile = filter_outfile.replace(".json.gz",".{}.json.gz".format(i))
        ## Dump the File and Check Correctness
        with gzip.open(chunk_filter_outfile,"wt",encoding="utf-8") as the_file:
            for instance in data_cache[x:y]:
                n_written += 1
                the_file.write("{}\n".format(json.dumps(instance)))
                ## Cache Chunk Of User
                if instance["user_id_str"] not in user2chunk:
                    user2chunk[instance["user_id_str"]] = i
                else:
                    if user2chunk[instance["user_id_str"]] != i:
                        raise ValueError("Chunk Caching Went Wrong")
                    else:
                        pass
    ## Write done outfile
    LOGGER.info("[Writing Completion File]")
    with open(done_outfile,"w") as the_file:
        the_file.write("Chunk complete.")
    ## Ensure Writing Length Matches Data Length
    assert n_written == len(data_cache)

def filter_users(config,
                 filter_id=None,
                 jobs=1,
                 skip_existing=False):
    """
    
    """
    ## Identify Filter Files
    if filter_id is None:
        chunk_filenames = sorted(glob("{}/chunks/*.txt".format(config["output"])))
    else:
        chunk_filenames = ["{}/chunks/{}.txt".format(config["output"], filter_id - 1)]
    ## Output Directory
    filtered_output_dir = "{}/filtered/".format(config["output"])
    if not os.path.exists(filtered_output_dir):
        try:
            _ = os.makedirs(filtered_output_dir)
        except FileExistsError:
            pass
    ## Iterate Over Chunk Files
    for f, chunk_file in enumerate(chunk_filenames):
        LOGGER.info("[Filtering Chunk {}/{}]".format(f+1, len(chunk_filenames)))
        _ = _filter_users_in_chunk(chunk_file=chunk_file,
                                   config=config,
                                   jobs=jobs,
                                   skip_existing=skip_existing)
    return None

def create_symlinks(config,
                    symlink,
                    symlink_items):
    """
    
    """
    ## Identify Output
    output_dir = config["output"]
    ## Ensure Symbolic Directory Exists
    if not os.path.exists(symlink):
        raise FileNotFoundError("Could not find desired path for symlinking.")
    ## Ensure Items Exist
    for i in symlink_items:
        if not os.path.exists(f"{symlink}/{i}"):
            raise FileNotFoundError("Could not find item within directory: {}".format(i))
    ## Establish Links
    for i in symlink_items:
        ## Determine Filepaths
        ipath = os.path.abspath(f"{symlink}/{i}")
        opath = os.path.abspath(f"{output_dir}/{i}")
        ## Check For Existing
        if os.path.exists(opath):
            raise FileExistsError("Output file already")
        ## Link File
        _ = os.system(f"ln -s {ipath} {opath}")

def _vectorize(file_data,
               date_boundaries,
               dvec):
    """
    
    """
    ## Users and Timestamps
    user_ids = [d.get("user_id_str") for d in file_data]
    timestamps = [d.get("created_utc") for d in file_data]
    ## Timestamp Assignment
    date_assignments = list(map(lambda t: _instance_date_boundary(t, date_boundaries), timestamps))
    date_mask = [i for i, d in enumerate(date_assignments) if d is not None]
    ## Apply Filter
    user_ids = [user_ids[m] for m in date_mask]
    date_assignments = [date_assignments[m] for m in date_mask]
    file_data = [file_data[m] for m in date_mask]
    ## Group
    instance_groups = {}
    for ind, (user, da) in enumerate(zip(user_ids, date_assignments)):
        if (user, da) not in instance_groups:
            instance_groups[(user, da)] = []
        instance_groups[(user, da)].append(ind)
    ## Vectorize
    X = []
    tau = []
    u = []
    n = []
    for (user, da), inds in instance_groups.items():
        ## Flatten Text and Count
        token_counts = Counter(flatten([file_data[ind]["text_tokenized"] for ind in inds]))
        ## Vectorize
        x = dvec.transform(token_counts)
        ## Cache
        X.append(x)
        tau.append(da)
        u.append(user)
        n.append(len(inds))
    ## Concatenate/Format
    X = sparse.vstack(X)
    tau = np.array(tau)
    n = np.array(n)
    return X, n, tau, u

def _load_vocabulary(vocabulary_files):
    """

    """
    ## Expand Files
    vocabulary_files_all = []
    for vf in vocabulary_files:
        if "*" not in vf:
            vocabulary_files_all.append(vf)
        else:
            vocabulary_files_all.extend(glob(vf))
    ## Initialize Cache
    vocabulary = set()
    for txt_file in vocabulary_files_all:
        ## Check for File
        if not os.path.exists(txt_file):
            raise FileNotFoundError("Vocabulary file does not exist: {}".format(txt_file))
        ## Load and Update Vocabulary
        with open(txt_file,"r") as the_file:
            for line in the_file:
                vocabulary.add(line.strip())
    ## Format
    vocabulary = sorted(set(vocabulary))
    return vocabulary

def vectorize_users(config,
                    vectorize_id=None,
                    skip_existing=False,
                    jobs=1):
    """
    
    """
    ## Isolate Vectorization Arguments
    vectorize_kwargs = config["vectorize_kwargs"]
    ## Initialize Output Directory
    vector_output = "{}/vectors/".format(config["output"])
    if not os.path.exists(vector_output):
        try:
            _ = os.makedirs(vector_output)
        except FileExistsError:
            pass
    ## Filenames to Vectorize
    filenames = sorted(glob("{}/filtered/*.json.gz".format(config["output"])))
    if vectorize_id is not None:
        filenames = filenames[vectorize_id - 1 : vectorize_id]
    ## Helper
    get_file_id = lambda filename: os.path.basename(filename).split(".json.gz")[0]
    get_output_file = lambda filename: "{}/data.{}.npz".format(vector_output, get_file_id(filename)).replace("//","/")
    ## Filter Existing and Early Exit
    if skip_existing:
        filenames = list(filter(lambda file: not os.path.exists(get_output_file(file)), filenames))
        if len(filenames) == 0:
            LOGGER.info("[All Files Already Processed. Exiting.]")
            return None
    ## Load Phrasers (if Provided)
    phrasers = None
    if vectorize_kwargs["phrasers"] is not None:
        phrasers = Word2Vec.load_phrasers(vectorize_kwargs["phrasers"])
    else:
        LOGGER.warning("Warning: No phrasers specified. Vectorization restricted to unigrams.")
    ## Load Vocabulary
    vocabulary = _load_vocabulary(vectorize_kwargs["vocabulary"])
    if len(vocabulary) == 0:
        raise ValueError("Vocabulary cannot be empty.")
    ## Check Vocabulary Consistency
    vocabulary_outfile = f"{vector_output}/vocabulary.txt"
    if os.path.exists(vocabulary_outfile):
        ## Load Existing Vocabulary
        existing_vocabulary = []
        with open(vocabulary_outfile,"r") as the_file:
            for line in the_file:
                existing_vocabulary.append(line.strip())
        ## Compare
        if existing_vocabulary != vocabulary:
            raise ValueError("Cached vocabulary does not match the existing vocabulary.")
    else:
        ## Save Vocabulary
        with open(vocabulary_outfile,"w") as the_file:
            for term in vocabulary:
                the_file.write(f"{term}\n")
    ## Date Boundaries
    date_boundaries = [(int(pd.to_datetime(i["min_date"]).timestamp()), int(pd.to_datetime(i["max_date"]).timestamp())) for i in config["cache_user_criteria"]["date_boundaries"]]
    date_boundaries = sorted(date_boundaries, key=lambda x: x[0])
    ## Initialize Dictionary Vectorizer
    dvec = DVec(items=vocabulary)
    ## Initialize Stream
    stream = PostStream(filenames=filenames,
                        loader_kwargs=vectorize_kwargs["vocab_kwargs"],
                        processor=None,
                        processor_kwargs={},
                        min_date=date_boundaries[0][0],
                        max_date=date_boundaries[-1][1],
                        n_samples=None,
                        randomized=True,
                        mode=2,
                        phrasers=phrasers,
                        jobs=min(jobs, len(filenames)),
                        preserve_order=True,
                        verbose=False,
                        check_filename_size=False,
                        yield_filename=True,
                        cache_data=False)
    ## Load and Process Iteratively
    for file_data, filename in stream:
        ## Vectorize
        X, n_posts, tau, user_ids = _vectorize(file_data,
                                               date_boundaries=date_boundaries,
                                               dvec=dvec)
        ## Get Output Filename/IDs
        output_id = get_file_id(filename)
        output_filename = get_output_file(filename)
        ## Cache to Disk
        _ = np.save(f"{vector_output}/n_posts.{output_id}.npy", n_posts)
        _ = np.save(f"{vector_output}/tau.{output_id}.npy", tau)
        with open(f"{vector_output}/users.{output_id}.txt","w") as the_file:
            for user in user_ids:
                the_file.write(f"{user}\n")
        _ = sparse.save_npz(output_filename, X, compressed=True)
        

def main():
    """
    
    """
    ## Parse Command Line
    args = parse_command_line()
    ## Load Configuration
    config = load_configuration(args.config)
    ## Run Counting Procedure
    if args.counter is not None:
        ## Parallel Scheduling
        if args.counter == "parallel" and args.counter_id is None:
            ## Schedule and Exit
            LOGGER.info("[Scheduling Counter]")
            _ = schedule_counter(config=config,
                                 jobs=args.jobs,
                                 skip_existing=args.counter_skip_existing,
                                 memory_per_job=args.grid_memory_per_job,
                                 max_array_size=args.grid_max_array_size,
                                 max_tasks_concurrent=args.grid_max_tasks_concurrent)
            LOGGER.info("[Scheduling Complete. Exiting]")
            return None
        ## Execute Count
        LOGGER.info("[Starting Count Procedure]")
        _ = count_users(config=config,
                        jobs=args.jobs,
                        file_id=args.counter_id,
                        skip_existing=args.counter_skip_existing)
    ## Analyze Counter Results
    if args.analyze_counts:
        LOGGER.info("[Analyzing Activity Counts]")
        _ = analyze_counts(config=config,
                           jobs=args.jobs)
    ## Build User Chunks
    if args.chunk_users:
        LOGGER.info("[Generating User Chunks]")
        _ = chunk_users(config=config,
                        dryrun=args.chunk_users_dryrun,
                        jobs=args.jobs)
    ## Run Data Filtering
    if args.filter_users is not None:
        ## Parallel Processing (Scheduling)
        if args.filter_users == "parallel" and args.filter_id is None:
            ## Schedule
            LOGGER.info("[Scheduling User Data Filtering]")
            _ = schedule_filter(config=config,
                                jobs=args.jobs,
                                skip_existing=args.filter_skip_existing,
                                memory_per_job=args.grid_memory_per_job,
                                max_array_size=args.grid_max_array_size,
                                max_tasks_concurrent=args.grid_max_tasks_concurrent)
            LOGGER.info("[Scheduling Complete. Exiting]")
            return None
        ## Execution
        LOGGER.info("[Filtering User Data]")
        _ = filter_users(config=config,
                         filter_id=args.filter_id,
                         skip_existing=args.filter_skip_existing,
                         jobs=args.jobs)
        LOGGER.info("[Filtering Complete]")
    ## Initialization (Symbolic Link)
    if args.symlink is not None:
        ## Create Symlinks
        LOGGER.info("[Creating Symbolic Links]")
        _ = create_symlinks(config=config,
                            symlink=args.symlink,
                            symlink_items=args.symlink_items)
        LOGGER.info("[Symbolic Links Created. Exiting.]")
        return None
    ## Vectorization
    if args.vectorize_users is not None:
        ## Parallel Processing (Scheduling)
        if args.vectorize_users == "parallel" and args.vectorize_id is None:
            ## Schedule
            LOGGER.info("[Scheduling Vectorization]")
            _ = schedule_vectorization(config=config,
                                       jobs=args.jobs,
                                       skip_existing=args.vectorize_skip_existing,
                                       memory_per_job=args.grid_memory_per_job,
                                       max_array_size=args.grid_max_array_size,
                                       max_tasks_concurrent=args.grid_max_tasks_concurrent)
            LOGGER.info("[Scheduling Complete]")
            return None
        ## Execution
        LOGGER.info("[Vectorizing Data]")
        _ = vectorize_users(config=config,
                            vectorize_id=args.vectorize_id,
                            skip_existing=args.vectorize_skip_existing,
                            jobs=args.jobs)
    ## Done
    LOGGER.info("[Script Complete]")

#############################
### Execution
#############################

if __name__ == "__main__":
    _ = main()
