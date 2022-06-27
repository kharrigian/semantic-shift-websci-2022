
"""
Semantically Stable Feature Selection Analysis
"""

########################
#### Imports
########################

## Standard Library
import os
import sys
import json
import argparse
import subprocess
from glob import glob
from copy import deepcopy
from functools import partial
from collections import Counter
from multiprocessing import Pool
from textwrap import dedent, wrap

## External Libraries
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from langid import langid
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.gridspec import GridSpec
from sklearn import feature_selection

## Private
from semshift.util.helpers import flatten, chunks
from semshift.util.logging import initialize_logger
from semshift.model.vocab import Vocabulary
from semshift.model import classifiers
from semshift.model.embed import Word2Vec
from semshift.model.data_loaders import PostStream
from semshift.model.feature_extractors import FeaturePreprocessor
from semshift.preprocess.tokenizer import STOPWORDS, CONTRACTIONS

## Local Project
_ = sys.path.append(os.path.abspath(os.path.dirname(__file__))+"/")
from helpers import get_feature_names, score_predictions, align_vocab, DVec

########################
#### Globals
########################

## Initialize Logger
LOGGER = initialize_logger()

## Load Language Identification Model ## Needed for Multiprocessing Environment
_ = langid.load_model()

## Platform Mapping
PLATFORM_MAP = {
    "clpsych":"twitter",
    "multitask":"twitter",
    "wolohan":"reddit",
    "smhd":"reddit"
}

########################
#### General Helpers
########################

def load_json(filename):
    """
    
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("Could not find JSON file: {}".format(filename))
    with open(filename,"r") as the_file:
        data = json.load(the_file)
    return data

def load_text(filename):
    """
    
    """
    data = []
    with open(filename,"r") as the_file:
        for line in the_file:
            data.append(line.strip())
    return data    

def bootstrap_ci(x,
                 alpha=0.05,
                 n_samples=100,
                 aggfunc=np.nanmean,
                 random_state=42):
    """
    Compute the confidence value of an estimate using the bootstrap

    Args:
        x (array): Data to evaluate
        alpha (float): Confidence Level
        n_samples (int): Number of bootstrap samples
        aggfunc (callable): Estimation function
    
    Returns:
        q_range (tuple): (lower, raw, upper) estimates
    """
    ## Initialize Sampler
    seed = np.random.RandomState(random_state)
    ## Check Data
    if isinstance(x, pd.Series):
        x = x.values
    if len(x) == 0:
        return np.nan
    ## Run Bootstrap
    n = len(x)
    q = aggfunc(x)
    q_cache = np.zeros(n_samples)
    for sample in range(n_samples):
        x_sample = seed.choice(x, n, replace=True)
        q_cache[sample] = aggfunc(x_sample)
    q_range = np.nanpercentile(q - q_cache,
                               [alpha/2*100, 100-(alpha/2*100)])
    q_range = (q + q_range[0], q, q + q_range[1])
    return q_range

def bootstrap_tuple_to_df(series):
    """
    Translate a series with confidence interval tuples into a dataframe

    Args:
        series (pandas Series): Series where values are tuples of length 3 (lower, middle, upper) estimates
    
    Returns:
        vals (pandas DataFrame): Series re-formatted as a dataframe
    """
    vals = {}
    for i, name in enumerate(["lower","median","upper"]):
        vals[name] = series.map(lambda j: j[i])
    vals = pd.DataFrame(vals)
    return vals

########################
#### Parallelization
########################

def get_header(nstart,
               nend,
               log_dir,
               memory=8,
               num_jobs=1):
    """

    """
    header=f"""
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -N postprocess_word2vec_v2
    #$ -t {nstart}-{nend}
    #$ -e {log_dir}
    #$ -o {log_dir}
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

def format_parallel_script_vectorize(lower_bound,
                                     upper_bound,
                                     analysis_directory,
                                     memory=8,
                                     num_jobs=1,
                                     skip_existing=False):
    """

    """
    ## Configuration Path
    analysis_config_filename = f"{analysis_directory}/analysis.config.json"
    ## Log Directoriy
    log_dir = f"{analysis_directory}/logs/vectorize/"
    if not os.path.exists(log_dir):
        _ = os.makedirs(log_dir)
    ## Construct Script
    header = get_header(lower_bound,
                        upper_bound,
                        log_dir=log_dir,
                        memory=memory,
                        num_jobs=num_jobs)
    init = get_init_env()
    script="""
    #!/bin/bash
    {}
    {}
    python scripts/model/word2vec_postprocess.py {} --vectorize parallel --vectorize_id $SGE_TASK_ID --jobs {} {}
    """.format(header, init, analysis_config_filename, num_jobs, "--vectorize_skip_existing" if skip_existing else "")
    script = dedent(script)
    script = script.replace("//","/")
    return script

def format_parallel_script_classify(lower_bound,
                                    upper_bound,
                                    analysis_directory,
                                    memory=8,
                                    num_jobs=1,
                                    skip_existing=False,
                                    enforce_frequency_filter=False,
                                    enforce_classwise_frequency_filter=False,
                                    enforce_min_shift_frequency_filter=False,
                                    compute_cumulative=False,
                                    compute_discrete=False):
    """

    """
    ## Configuration Path
    analysis_config_filename = f"{analysis_directory}/analysis.config.json"
    ## Log Directory
    log_dir = f"{analysis_directory}/logs/classify/"
    if not os.path.exists(log_dir):
        _ = os.makedirs(log_dir)
    ## Construct Script
    header = get_header(lower_bound,
                        upper_bound,
                        log_dir=log_dir,
                        memory=memory,
                        num_jobs=num_jobs)
    init = get_init_env()
    script="""
    #!/bin/bash
    {}
    {}
    python scripts/model/word2vec_postprocess.py {} --classify parallel --classify_id $SGE_TASK_ID --jobs {} {} {} {} {} {} {}
    """.format(header,
               init,
               analysis_config_filename,
               num_jobs,
               "--classify_skip_existing" if skip_existing else "",
               "--classify_enforce_frequency_filter" if enforce_frequency_filter else "",
               "--classify_enforce_classwise_frequency_filter" if enforce_classwise_frequency_filter else "",
               "--classify_enforce_min_shift_frequency_filter" if enforce_min_shift_frequency_filter else "",
               "--classify_compute_cumulative" if compute_cumulative else "",
               "--classify_compute_discrete" if compute_discrete else ""
    )
    script = dedent(script)
    script = script.replace("//","/")
    return script
    
def schedule_vectorization_jobs(sample_id_list,
                                analysis_directory,
                                memory_per_job=8,
                                jobs=8,
                                max_array_size=500,
                                skip_existing=False):
    """
    
    """
    ## Array Bounds
    sample_bounds = [(i[0], i[-1]) for i in chunks(list(range(1, len(sample_id_list)+1)), max_array_size)]
    n_sample = len(sample_bounds)
    ## Create Job Directory
    job_dir = f"{analysis_directory}/scheduler/"
    if not os.path.exists(job_dir):
        _ = os.makedirs(job_dir)
    ## Create Scripts
    job_files = []
    for lower, upper in sample_bounds:
        ## Generate Script
        script = format_parallel_script_vectorize(lower_bound=lower,
                                                  upper_bound=upper,
                                                  analysis_directory=analysis_directory,
                                                  memory=memory_per_job,
                                                  num_jobs=jobs,
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

def schedule_classification_jobs(n_sample,
                                 analysis_directory,
                                 memory_per_job=8,
                                 jobs=8,
                                 max_array_size=500,
                                 skip_existing=False,
                                 enforce_frequency_filter=False,
                                 enforce_classwise_frequency_filter=False,
                                 enforce_min_shift_frequency_filter=False,
                                 compute_cumulative=False,
                                 compute_discrete=False):
    """

    """
    ## Job Array Bounds
    sample_bounds = [(i[0], i[-1]) for i in chunks(list(range(1, n_sample+1)), max_array_size)]
    ## Create Job Directory
    job_dir = f"{analysis_directory}/scheduler/"
    if not os.path.exists(job_dir):
        _ = os.makedirs(job_dir)
    ## Create Scripts
    job_files = []
    for lower, upper in sample_bounds:
        ## Generate Script
        script = format_parallel_script_classify(lower_bound=lower,
                                                 upper_bound=upper,
                                                 analysis_directory=analysis_directory,
                                                 memory=memory_per_job,
                                                 num_jobs=jobs,
                                                 skip_existing=skip_existing,
                                                 enforce_frequency_filter=enforce_frequency_filter,
                                                 enforce_classwise_frequency_filter=enforce_classwise_frequency_filter,
                                                 enforce_min_shift_frequency_filter=enforce_min_shift_frequency_filter,
                                                 compute_cumulative=compute_cumulative,
                                                 compute_discrete=compute_discrete)
        ## Write Script File
        script_file = f"{job_dir}/classify_{lower}_{upper}.sh".replace("//","/")
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

########################
#### Functions
########################

def parse_command_line():
    """
    
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("config",
                            type=str,
                            help="Path to analysis configuration file (JSON).")
    _ = parser.add_argument("--rm_existing",
                            action="store_true",
                            default=False,
                            help="If flag is included and analysis directory already exists, we will remove it in entirety.")
    _ = parser.add_argument("--vectorize",
                            type=str,
                            default=None,
                            choices={"serial","parallel","symbolic"},
                            help="Whether to preprocess the data. If included, specify either 'serial' or 'parallel' (CLSP grid)")
    _ = parser.add_argument("--vectorize_id",
                            type=int,
                            default=None,
                            help="Which vectorization process should be running (used for parallel processing).")
    _ = parser.add_argument("--vectorize_skip_existing",
                            action="store_true",
                            default=False,
                            help="If included, skip any existing vectors.")
    _ = parser.add_argument("--vectorize_symbolic_path",
                            type=str,
                            default=None,
                            help="Path to data-cache containing pre-vectorized data to symbolically copy.")
    _ = parser.add_argument("--classify",
                            type=str,
                            default=None,
                            choices={"serial","parallel"},
                            help="Whether to run classification procedures. If included, specify either 'serial' or 'parallel' (CLSP grid).")
    _ = parser.add_argument("--classify_id",
                            type=int,
                            default=None,
                            help="Which sample iteration (embedding training split) to use for classification.")
    _ = parser.add_argument("--classify_compute_cumulative",
                            action="store_true",
                            default=False,
                            help="If True, run computations for cumulative baseline model.")
    _ = parser.add_argument("--classify_compute_discrete",
                            action="store_true",
                            default=False,
                            help="If True, run computations for discrete baseline model.")
    _ = parser.add_argument("--classify_skip_existing",
                            default=False,
                            action="store_true",
                            help="Whether to skip sample is existing classification results exist.")
    _ = parser.add_argument("--classify_enforce_frequency_filter",
                            default=False,
                            action="store_true",
                            help="Whether to enforce frequency filter in baseline classification.")
    _ = parser.add_argument("--classify_enforce_classwise_frequency_filter",
                            default=False,
                            action="store_true",
                            help="Whether frequency filtering should be applied to both classes independently.")
    _ = parser.add_argument("--classify_enforce_min_shift_frequency_filter",
                            action="store_true",
                            default=False,
                            help="Whether to enforce min_shift_freq filter in baseline experiments.")
    _ = parser.add_argument("--analyze",
                            action="store_true",
                            default=False,
                            help="Whether to analyze existing results.")
    _ = parser.add_argument("--jobs",
                            type=int,
                            default=1,
                            help="Number of processes to leverage within the processing pipeline.")
    _ = parser.add_argument("--grid_memory_per_job",
                            type=int,
                            default=4,
                            help="Size of memory request for parallel processing within the pipeline.")
    _ = parser.add_argument("--grid_max_array_size",
                            type=int,
                            default=500,
                            help="Maximum number of tasks to include within a single job array.")
    args = parser.parse_args()
    return args

def check_configuration_equivalence_vectorization(configuration_a,
                                                  configuration_b):
    """

    """
    ## Parameters used by vectorization procedure
    vector_params = [
        "base_output_dir",
        "base_output_dir_ext",
        "min_train_posts",
        "min_test_posts",
        "train_posts",
        "test_posts"
    ]
    ## Check Each Parameter
    for vp in vector_params:
        if configuration_a.get(vp) != configuration_b.get(vp):
            return False
    return True

def cache_configuration(configuration,
                        analysis_directory):
    """

    """
    ## Filepath
    configuration_filepath = f"{analysis_directory}/analysis.config.json"
    ## Look for Current Configuration
    if os.path.exists(configuration_filepath):
        ## Load
        configuration_current = load_json(configuration_filepath)
        ## Compare and Skip Equivalent
        if configuration != configuration_current:
            LOGGER.warning("[Warning: Configuration being cached is different than the configuration file that already exists.]")
        else:
            LOGGER.warning("[Warning: Configuration being cached is equivalent to the configuration file that already exists.]")
            return None
    ## Cache
    with open(configuration_filepath,"w") as the_file:
        json.dump(configuration, the_file, indent=4, sort_keys=False)

def load_embeddings(embeddings_directory):
    """
    
    """
    ## Load
    model = Word2Vec.load(embeddings_directory)
    ## Parse Embeddings
    vocabulary = model.get_ordered_vocabulary()
    vectors = model.model.wv.vectors
    frequency = np.array([model.model.wv.get_vecattr(v,"count") for v in vocabulary])
    phrasers = model.phrasers
    return vocabulary, vectors, frequency, phrasers

def _is_stopword(term,
                 stopset=set()):
    """
    
    """
    term = term.lower()
    if term in stopset or f"not_{term}" in stopset:
        return True
    return False

def _get_filter_mask(vocabulary,
                     frequency,
                     min_vocab_freq=5,
                     rm_top=250,
                     rm_stopwords=True):
    """
    
    """
    ## Filtering
    mask = set(range(len(vocabulary)))
    if min_vocab_freq is not None:
        mask = mask & set(i for i, f in enumerate(frequency) if f >= min_vocab_freq)
    if rm_top is not None and rm_top > 0:
        mask = mask & set(frequency.argsort()[:-rm_top])
    if rm_stopwords:
        stopset = set(STOPWORDS) | set(CONTRACTIONS) | set(CONTRACTIONS.values())
        stopset = set(f"not_{term}" for term in stopset) | stopset
        mask = mask & set(i for i, v in enumerate(vocabulary) if not _is_stopword(v, stopset))
    mask = sorted(mask)
    return mask

def filter_embeddings(vectors,
                      vocabulary,
                      frequency,
                      min_vocab_freq=5,
                      rm_top=250,
                      rm_stopwords=True):
    """

    """
    ## Get Filter Mask
    mask = _get_filter_mask(vocabulary, frequency, min_vocab_freq, rm_top, rm_stopwords)
    ## Apply Filtering
    vocabulary = [vocabulary[i] for i in mask]
    vectors = vectors[mask]
    frequency = frequency[mask]
    return vectors, vocabulary, frequency

def compute_neighborhood(vectors,
                         vocabulary,
                         global_vocabulary,
                         top_k=100,
                         chunksize=500):
    """
    Args:
        vectors (2d-array): Embeddings
        vocabulary (list of str): Row-wise aligned vocabulary
        global_vocabulary (list of str): Global vocabulary ordering
        top_k (int): Number of neighbors to cache
        chunksize (int): Cosine similarity chunksize
    """
    ## Compute Neighborhood
    neighbors = []
    for index_chunk in list(chunks(list(range(vectors.shape[0])), chunksize)):
        sim = cosine_similarity(vectors[index_chunk], vectors)
        sim = np.apply_along_axis(lambda x: np.argsort(x)[-top_k-1:-1][::-1], 1, sim)
        neighbors.append(sim)
    neighbors = np.vstack(neighbors)
    ## Alignment
    globalvocab2ind = dict(zip(global_vocabulary, range(len(global_vocabulary))))
    neighbors = np.apply_along_axis(lambda x: list(map(lambda i: globalvocab2ind[vocabulary[i]], x)), 1, neighbors)
    return neighbors

def _get_time_period_samples(time_period_config,
                             experiment_directory):
    """
    
    """
    ## First-Level Information
    time_period_directory = "{}/{}/".format(experiment_directory, time_period_config["experiment_name"])
    time_period_dataset = time_period_config["datasets"][0]["dataset"]
    ## Split-Level Information
    clean = lambda x: x.replace("//","/")
    sample2ind = lambda i: int(os.path.basename(i).split(".")[0])
    sample_split_filenames = list(map(clean, sorted(glob("{}/splits/{}/*.json".format(time_period_directory, time_period_dataset)), key=sample2ind)))
    sample_embedding_directories = list(map(clean, ["{}/models/sample-{}/".format(time_period_directory, sample2ind(i)) for i in sample_split_filenames]))
    sample_metadata = list(zip(range(len(sample_split_filenames)), sample_split_filenames, sample_embedding_directories))
    ## Return
    return sample_metadata

def get_samples(time_period_configs,
                experiment_directory):
    """
    
    """
    ## Sample ID Cache
    sample_id_list = []
    ## Iterate Through Time Periods
    for t, time_period_config in enumerate(time_period_configs):
        ## Time Period IDs
        time_period_sample_ids = _get_time_period_samples(time_period_config,
                                                          experiment_directory)
        ## Append Config ID
        time_period_sample_ids = [(t, *tps) for tps in time_period_sample_ids]
        ## Cache
        sample_id_list.extend(time_period_sample_ids)
    ## Return
    return sample_id_list

def _update_split_keys(splits):
    """
    
    """
    local_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../") + "/"
    filename2root = lambda f: os.path.abspath(os.path.dirname(f) + "/../../../../") + "/"
    filename2local = lambda f: f.replace(filename2root(f), local_root)
    updated_splits = {"train":{},"test":{}}
    reverse_mapping = {}
    for group in ["train","test"]:
        for key, value in splits[group].items():
            keynew = filename2local(key)
            reverse_mapping[keynew] = key
            assert os.path.exists(keynew)
            updated_splits[group][keynew] = value
    return updated_splits, reverse_mapping

def _count_time_period_data_sample_user(stream_input,
                                        stream,
                                        min_n_samples=None,
                                        n_samples=None,
                                        random_state=42):
    """
    
    """
    ## Format Sample Number
    if min_n_samples is not None and n_samples is not None:
        _required = max(min_n_samples, n_samples)
        min_n_samples = _required
        n_samples = _required
    elif min_n_samples is not None and n_samples is None:
        pass
    elif min_n_samples is None and n_samples is not None:
        min_n_samples = n_samples
    elif min_n_samples is None and n_samples is None:
        min_n_samples = 0
    ## Parse Tuple
    fid, filename = stream_input
    ## Load Posts
    posts = stream._iter_inner_loop(fid, filename)
    ## Base Case
    if len(posts) == 0:
        return filename, None
    posts = posts[0]
    ## Sampling
    if len(posts) < min_n_samples:
        return (filename, None)
    if n_samples is not None:
        seed = np.random.RandomState(random_state)
        posts = seed.choice(posts, n_samples, replace=False)
    ## Counting
    counts = Counter(flatten(p["text_tokenized"] for p in posts))
    return filename, counts

def vectorize_time_period_data_sample(sample_ind,
                                      sample_split_filename,
                                      sample_embedding_directory,
                                      analysis_directory,
                                      time_period_config,
                                      min_n_posts_train=None,
                                      min_n_posts_test=None,
                                      n_posts_train=None,
                                      n_posts_test=None,
                                      random_state=42,
                                      mp=None,
                                      skip_existing=True,
                                      verbose=False):
    """
    
    """
    ## Input Helpers
    time_period_id = time_period_config["experiment_name"]
    cache_dir = f"{analysis_directory}/data-cache/{time_period_id}/{sample_ind}/"
    ## Post Information
    min_n_posts = {"train":min_n_posts_train,"test":min_n_posts_test}
    n_posts = {"train":n_posts_train,"test":n_posts_test}
    ## Check for Existing
    if skip_existing and os.path.exists(cache_dir):
        ## Check for Filenames
        all_exists = True
        for suf in ["embeddings.npy","frequency.npy","labels.npy","groups.npy","vocabulary.txt","users.txt","data.npz"]:
            if not os.path.exists(f"{cache_dir}/{suf}"):
                all_exists = False
        ## Early Exit
        if all_exists:
            if verbose:
                LOGGER.info("[Vectors Already Exist ({}). Skipping.]".format(cache_dir))
            return None
    ## Load Splits (Filenames)
    splits = load_json(sample_split_filename)
    splits, splits_filename_reverse = _update_split_keys(splits)
    ## Load Embeddings
    vocabulary, vectors, frequency, phrasers = load_embeddings(sample_embedding_directory)
    ## Initialize Vectorizer
    dvec = DVec(vocabulary)
    ## Prepare Loaders
    X = []
    y = []
    users = []
    groups = []
    for group in ["train","test"]:
        ## Group Integer ID
        group_int_id = int(group == "test")
        ## Initialize Stream
        group_stream = PostStream(filenames=list(splits[group].keys()),
                                  loader_kwargs=time_period_config["vocab_kwargs"],
                                  processor=None,
                                  processor_kwargs={},
                                  min_date=time_period_config["datasets"][0]["date_boundaries"]["min_date"],
                                  max_date=time_period_config["datasets"][0]["date_boundaries"]["max_date"],
                                  randomized=True,
                                  n_samples=None,
                                  mode=2,
                                  phrasers=phrasers,
                                  jobs=1,
                                  check_filename_size=False,
                                  cache_data=False)
        ## Count Tokens
        counter = partial(_count_time_period_data_sample_user, stream=group_stream, min_n_samples=min_n_posts[group], n_samples=n_posts[group], random_state=random_state)
        iterable = list(enumerate(group_stream.filenames))
        if verbose:
            iterable = tqdm(iterable, file=sys.stdout, desc=f"[Counting Terms ({group.title()})]", total=len(group_stream.filenames))
        if mp is None:
            group_counts = list(map(counter, iterable))
        else:
            group_counts = list(mp.imap_unordered(counter, iterable))
        ## Filter
        group_counts = list(filter(lambda gc: gc[0] is not None and gc[1] is not None, group_counts))
        ## Sort
        group_counts = sorted(group_counts, key=lambda x: x[0])
        ## Vectorize and Cache
        group_users = [g[0] for g in group_counts]
        users.extend(group_users)
        X.append(dvec.transform([g[1] for g in group_counts]))
        y.extend([int(splits[group][user] != "control") for user in group_users])
        groups.extend([group_int_id for _ in group_counts])
    ## Stack and Format
    X = sparse.vstack(X)
    y = np.array(y)
    groups = np.array(groups)
    users = [splits_filename_reverse[u] for u in users]
    ## Cache Data to Disk
    if not os.path.exists(cache_dir):
        _ = os.makedirs(cache_dir)
    _ = np.save(f"{cache_dir}/embeddings.npy", vectors)
    _ = np.save(f"{cache_dir}/frequency.npy", frequency)
    _ = np.save(f"{cache_dir}/labels.npy", y)
    _ = np.save(f"{cache_dir}/groups.npy", groups)
    with open(f"{cache_dir}/vocabulary.txt","w") as the_file:
        for term in vocabulary:
            the_file.write(f"{term}\n")
    with open(f"{cache_dir}/users.txt", "w") as the_file:
        for user in users:
            the_file.write(f"{user}\n")
    _ = sparse.save_npz(f"{cache_dir}/data.npz", X)

def initialize_symbolic_vector_directory(analysis_directory,
                                         symbolic_directory,
                                         analysis_config):
    """

    """
    ## Check Input
    if symbolic_directory is None:
        raise ValueError("Need non-null symbolic directory input.")
    ## Check for Data-Cache
    if "data-cache" not in symbolic_directory:
        symbolic_directory = f"{symbolic_directory}/data-cache/".replace("//","/")
    ## Check for Directory
    if not os.path.exists(symbolic_directory):
        raise FileNotFoundError("Could not find desired symbolic directory.")
    ## Load Symbolic Analysis Configuration
    symbolic_analysis_config = load_json(os.path.abspath(f"{symbolic_directory}/../analysis.config.json"))
    ## Check Analysis Config Alignment
    if not check_configuration_equivalence_vectorization(analysis_config, symbolic_analysis_config):
        raise ValueError("Vectorization parameters of symbolic path do not align with desired analysis vectorization parameters")
    ## Setup Symbolic Path
    ref = os.path.abspath(symbolic_directory).rstrip("/")
    tar = os.path.abspath(f"{analysis_directory}/data-cache".replace("//","/"))
    _ = os.system("ln -s {} {}".format(ref, tar))
    LOGGER.info("[Symbolic path initialized at: {}]".format(tar))

def compute_overlap(*neighbors):
    """
    Compute overlap between multiple neighborhoods

    Args:
        neighbors (list of array): Each array is an ordered list of term IDs, aligned globally.
    
    Returns:
        overlap_at_k (float): Percentage of neighboring terms that overlap
    """
    ## Check to see if any neighborhoods don't have any neighbors
    for n in neighbors:
        if n.min() == -1:
            return np.nan
    ## Computations
    top_k = neighbors[0].shape[0]
    neighbors = [set(n) for n in neighbors]
    intersection = set.intersection(*neighbors)
    return len(intersection) / top_k

def align_and_compute_similarity(n,
                                 analysis_directory,
                                 analysis_config,
                                 time_period_configs):
    """
    Args:
        n (int): Sample index
    """
    ## Sample Cache Directories
    n_cache_dirs = ["{}/data-cache/{}/{}/".format(analysis_directory,i["experiment_name"],n) for i in time_period_configs]
    ## Get Global Vocabulary
    global_vocabulary = set()
    for cache_dir in tqdm(n_cache_dirs, desc="[Building Global Vocabulary]", total=len(n_cache_dirs), file=sys.stdout):
        ## Load Resources
        cache_dir_vocabulary = load_text(f"{cache_dir}/vocabulary.txt")
        cache_dir_frequency = np.load(f"{cache_dir}/frequency.npy")
        ## Get Mask
        cache_dir_mask = _get_filter_mask(cache_dir_vocabulary,
                                          cache_dir_frequency,
                                          min_vocab_freq=analysis_config["min_vocab_freq"],
                                          rm_top=analysis_config["rm_top"],
                                          rm_stopwords=analysis_config["rm_stopwords"])
        ## Add Terms to Global Vocabulary
        for m in cache_dir_mask:
            if cache_dir_vocabulary[m] not in global_vocabulary:
                global_vocabulary.add(cache_dir_vocabulary[m])
    global_vocabulary = sorted(global_vocabulary)
    ## Compute Neighborhoods
    neighbors = np.zeros((len(time_period_configs), len(global_vocabulary), analysis_config["top_k_neighbors"]), dtype=int)
    frequencies = np.full((len(time_period_configs), len(global_vocabulary)), -1, dtype=int)
    for i, cache_dir in tqdm(enumerate(n_cache_dirs), desc="[Computing Semantic Shift]", total=len(n_cache_dirs), file=sys.stdout):
        ## Load Resources
        cache_dir_vocabulary = load_text(f"{cache_dir}/vocabulary.txt")
        cache_dir_frequency = np.load(f"{cache_dir}/frequency.npy")
        cache_dir_vectors = np.load(f"{cache_dir}/embeddings.npy")
        ## Filter Embeddings
        cache_dir_vectors, cache_dir_vocabulary, cache_dir_frequency = filter_embeddings(vectors=cache_dir_vectors,
                                                                                         vocabulary=cache_dir_vocabulary,
                                                                                         frequency=cache_dir_frequency,
                                                                                         min_vocab_freq=analysis_config["min_vocab_freq"],
                                                                                         rm_top=analysis_config["rm_top"],
                                                                                         rm_stopwords=analysis_config["rm_stopwords"])
        ## Similarity Computations
        cache_dir_neighbors = compute_neighborhood(vectors=cache_dir_vectors,
                                                   vocabulary=cache_dir_vocabulary,
                                                   global_vocabulary=global_vocabulary,
                                                   top_k=analysis_config["top_k_neighbors"],
                                                   chunksize=500)
        ## Align Rows with Global Vocabulary
        cache_dir_vocab2ind = dict(zip(cache_dir_vocabulary, range(len(cache_dir_vocabulary))))
        cache_dir_neighbors = np.vstack(list(map(lambda term: cache_dir_neighbors[cache_dir_vocab2ind[term]] if term in cache_dir_vocab2ind else np.ones(cache_dir_neighbors.shape[1], dtype=int) * -1,
                                                 global_vocabulary)))
        cache_dir_frequency = np.array(list(map(lambda term: cache_dir_frequency[cache_dir_vocab2ind[term]] if term in cache_dir_vocab2ind else np.nan, global_vocabulary)))
        ## Store
        neighbors[i] = cache_dir_neighbors
        frequencies[i] = cache_dir_frequency    
    ## Discrete vs. Cumulative Periods
    cumulative_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("cumulative/")]
    discrete_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("discrete/")]
    ## Compute Semantic Similarity
    similarity = np.full((len(cumulative_time_period_configs), len(discrete_time_period_configs), len(global_vocabulary)), np.nan)
    for i, cind in enumerate(cumulative_time_period_configs):
        for j, dind in enumerate(discrete_time_period_configs):
            if j < i:
                continue
            similarity[i, j] = list(map(lambda x: compute_overlap(*x),zip(neighbors[cind], neighbors[dind])))
    return global_vocabulary, neighbors, similarity, frequencies

def load_sample_users(n,
                      analysis_directory,
                      time_period_configs):
    """
    
    """
    ## Sample Cache Directories
    n_cache_dirs = ["{}/data-cache/{}/{}/".format(analysis_directory,i["experiment_name"],n) for i in time_period_configs]
    ## Identify Available Users by Time Period
    splits = []
    for cache_dir in n_cache_dirs:
        ## Load Resources
        cache_dir_users = load_text(f"{cache_dir}/users.txt")
        cache_dir_groups = np.load(f"{cache_dir}/groups.npy")
        cache_dir_labels = np.load(f"{cache_dir}/labels.npy")
        ## Separate by Group
        cache_dir_splits = [{},{}]
        for group in [0, 1]:
            group_ind = (cache_dir_groups == group).nonzero()[0]
            for user, label in zip([cache_dir_users[g] for g in group_ind], cache_dir_labels[group_ind]):
                cache_dir_splits[group][user] = label
        ## Store
        splits.append(cache_dir_splits)
    ## Check That There isn't Train/Test Overlap
    assert len(set.union(*[set(s[0].keys()) for s in splits]) & set.union(*[set(s[1].keys()) for s in splits])) == 0
    return splits

def _sample_from_splits(splits,
                        uniform_class_balance=True,
                        uniform_period_balance=True,
                        random_state=None):
    """
    
    """
    ## State
    seed = np.random.RandomState(random_state)
    ## Count User Availability
    n_available = np.zeros((len(splits), 2, 2), dtype=int) ## Number of Splits x Group x Label
    for s, split in enumerate(splits):
        for g, group_labels in enumerate(split):
            for lbl in [0, 1]:
                n_available[s, g, lbl] = len([x for x,y in group_labels.items() if y == lbl])
    ## Determine Appropriate Sample Sizes
    if uniform_class_balance and uniform_period_balance:
        ## Enforce Class Balance
        n_train = np.vstack([i[0] for i in n_available]).min(axis=1, keepdims=True)
        n_test = np.vstack([i[1] for i in n_available]).min(axis=1, keepdims=True)
        ## Enforce Period Balance
        n_train = n_train.min()
        n_test = n_test.min()
        ## Update Availability
        n_available[:,0,:] = n_train
        n_available[:,1,:] = n_test
    elif uniform_class_balance and not uniform_period_balance:
        ## Enforce Class Balance
        n_train = np.vstack([i[0] for i in n_available]).min(axis=1, keepdims=True)
        n_test = np.vstack([i[1] for i in n_available]).min(axis=1, keepdims=True)
        ## Update Availablity
        n_available[:,0,:] = np.ones_like(n_available[:,0,:]) * n_train
        n_available[:,1,:] = np.ones_like(n_available[:,1,:]) * n_test
    elif not uniform_class_balance and uniform_period_balance:
        ## Enforce Period Balance
        n_train = np.vstack([i[0] for i in n_available]).min(axis=0,keepdims=True)
        n_test = np.vstack([i[1] for i in n_available]).min(axis=0,keepdims=True)
        ## Update Availability
        n_available[:,0,:] = np.ones_like(n_available[:,0,:]) * n_train
        n_available[:,1,:] = np.ones_like(n_available[:,1,:]) * n_test
    ## Sample The Users
    splits_sampled = [({},{}) for _ in splits]
    for s, split in enumerate(splits):
        for g, group_labels in enumerate(split):
            for lbl in [0, 1]:
                s_g_lbl = seed.choice([x for x,y in group_labels.items() if y == lbl],
                                      n_available[s, g, lbl],
                                      replace=False)
                splits_sampled[s][g].update({u:lbl for u in s_g_lbl})
    return splits_sampled

def sample_from_splits(splits,
                       analysis_config,
                       time_period_configs,
                       random_state=42):
    """
    
    """
    ## Discrete vs. Cumulative Periods
    cumulative_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("cumulative/")]
    discrete_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("discrete/")]
    ## Get Cumulative and Discrete Splits Separately
    sampled_splits = [None for _ in range(len(time_period_configs))]
    for config_inds in [cumulative_time_period_configs, discrete_time_period_configs]:
        config_sampled_splits = _sample_from_splits([splits[c] for c in config_inds],
                                                    uniform_class_balance=analysis_config["uniform_class_balance"],
                                                    uniform_period_balance=analysis_config["uniform_period_balance"],
                                                    random_state=random_state)
        for i, c in enumerate(config_inds):
            sampled_splits[c] = config_sampled_splits[i]
    ## Return
    return sampled_splits

def load_document_term_data(n,
                            vocabulary,
                            analysis_directory,
                            time_period_configs):
    """
    
    """
    ## Sample Cache Directories
    n_cache_dirs = ["{}/data-cache/{}/{}/".format(analysis_directory,i["experiment_name"],n) for i in time_period_configs]
    ## Initialize Data Cache
    X, y, tau, group, users = [], [], [], [], []
    ## Cycle Through Directories
    for i, cache_dir in enumerate(n_cache_dirs):
        ## Load Resources
        X_cache_dir = sparse.load_npz(f"{cache_dir}/data.npz")
        y_cache_dir = np.load(f"{cache_dir}/labels.npy")
        groups_cache_dir = np.load(f"{cache_dir}/groups.npy")
        users_cache_dir = load_text(f"{cache_dir}/users.txt")
        vocabulary_cache_dir = load_text(f"{cache_dir}/vocabulary.txt")
        ## Vocabulary Alignment
        X_cache_dir = align_vocab(X_cache_dir, vocabulary_cache_dir, vocabulary)
        ## Store
        X.append(X_cache_dir)
        y.append(y_cache_dir)
        group.append(groups_cache_dir)
        tau.append([i for _ in range(X_cache_dir.shape[0])])
        users.extend(users_cache_dir)
    ## Format
    X = sparse.vstack(X)
    y = np.hstack(y)
    tau = np.hstack(tau)
    group = np.hstack(group)
    return X, y, tau, group, users

def filter_document_term_data(X,
                              y,
                              tau,
                              group,
                              users,
                              time_period_splits):
    """
    
    """
    ## Initialize Mask
    split_mask = []
    ## Cycle Through Time Periods
    for t, tsplit in enumerate(time_period_splits):
        ## Time Period Index
        tinds = (tau == t).nonzero()[0]
        ## Time Period Mask
        t_split_mask = [ind for ind in tinds if users[ind] in tsplit[0] or users[ind] in tsplit[1]]
        ## Update
        split_mask.extend(t_split_mask)
    ## Format
    split_mask = sorted(split_mask)
    ## Filter
    X = X[split_mask]
    y = y[split_mask]
    tau = tau[split_mask]
    group = group[split_mask]
    return X, y, tau, group

def fit_classifier(X_train,
                   y_train,
                   vocabulary,
                   analysis_config,
                   njobs=1,
                   random_state=None):
    """
    
    """
    ## Initialize Vocabulary
    vocabulary_obj = Vocabulary()
    vocabulary_obj = vocabulary_obj.assign(vocabulary)
    ## Extract Features
    preprocessor = FeaturePreprocessor(vocab=deepcopy(vocabulary_obj),
                                       feature_flags=analysis_config["feature_flags"],
                                       feature_kwargs=analysis_config["feature_kwargs"],
                                       standardize=analysis_config["feature_standardize"],
                                       min_variance=None,
                                       verbose=False)
    X_train_T = preprocessor.fit_transform(X_train)
    ## Fit Model
    model = classifiers.linear_model.LogisticRegressionCV(Cs=10,
                                                          fit_intercept=True,
                                                          cv=10,
                                                          n_jobs=njobs,
                                                          scoring=analysis_config["cv_metric_opt"],
                                                          max_iter=1000,
                                                          verbose=0,
                                                          solver="lbfgs",
                                                          random_state=random_state)
    model = model.fit(X_train_T, y_train)
    ## Make Training Predictions
    y_pred_train = model.predict_proba(X_train_T)[:,1]
    ## Score Predictions
    scores_train = score_predictions(y_train, y_pred_train, 0.5)
    ## Return
    return preprocessor, model, scores_train

def evaluate_classifier(X_apply,
                        y_apply,
                        preprocessor,
                        model):
    """
    
    """
    ## Transform
    X_apply_T = preprocessor.transform(X_apply)
    ## Make Predictions
    y_pred_apply = model.predict_proba(X_apply_T)[:,1]
    ## Score Predictions
    scores_apply = score_predictions(y_apply, y_pred_apply, 0.5)
    return scores_apply

def format_vocabulary(terms):
    """
    Translate a list of n-grams (whitespace delimited) into the tuple format of the Vocabulary class

    Args:
        terms (list of str): List of strings representing n-grams in vocabulary
    
    Returns:
        terms (list of tuple): List of tuples representing the n-grams in vocabulary
    """
    terms = list(map(lambda t: tuple(t.split()) if not isinstance(t, tuple) else t, terms))
    return terms

def compute_vocabulary_statistics(X,
                                  y,
                                  vmask,
                                  vocabulary):
    """

    """
    vocabulary_statistics = []
    for lbl in [0, 1]:
        lbl_mask = (y == lbl).nonzero()[0]
        lbl_freq = X[lbl_mask][:,vmask].sum(axis=0).A[0].astype(int)
        lbl_sample_freq = (X[lbl_mask][:,vmask] != 0).sum(axis=0).A[0].astype(int)
        vocabulary_statistics.append(pd.DataFrame(
            index=["term","frequency","sample_frequency","label","support"],
            data=[[" ".join(f) for f in vocabulary], lbl_freq, lbl_sample_freq, np.ones_like(lbl_sample_freq) * lbl, np.ones_like(lbl_sample_freq) * len(lbl_mask)]
        ).T)
    vocabulary_statistics = pd.concat(vocabulary_statistics, axis=0,sort=False)
    vocabulary_statistics[["frequency","sample_frequency","label","support"]] = vocabulary_statistics[["frequency","sample_frequency","label","support"]].astype(int)
    vocabulary_statistics = pd.pivot_table(vocabulary_statistics,
                                           index=["term"],
                                           columns=["label"],
                                           values=["frequency","sample_frequency","support"])
    return vocabulary_statistics

def compute_vocabulary_scores(vocabulary_statistics,
                              alpha=1):
    """
    Reference (nPMI): https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf
    Reference (maxPMI,modifiedPMI): https://link.springer.com/chapter/10.1007/978-3-030-89880-9_26
    """
    ## Probabilities
    p_w_1 = (vocabulary_statistics["frequency",1] + alpha) / (vocabulary_statistics["frequency",1] + alpha).sum() # p(w|y=1)
    p_w_0 = (vocabulary_statistics["frequency",0] + alpha) / (vocabulary_statistics["frequency",0] + alpha).sum() # p(w|y=0)
    p_w = (vocabulary_statistics["frequency"].sum(axis=1) + alpha) / (vocabulary_statistics["frequency"].sum(axis=1) + alpha).sum() # p(w)
    p_w_s_1 = (vocabulary_statistics["sample_frequency",1] + alpha) / (vocabulary_statistics["support",1] + alpha) # p(ws|y=1)
    p_w_s_0 = (vocabulary_statistics["sample_frequency",0] + alpha) / (vocabulary_statistics["support",0] + alpha) # p(ws|y=0)
    p_w_s = (vocabulary_statistics["sample_frequency"].sum(axis=1) + alpha) / (vocabulary_statistics["sample_frequency"].sum(axis=1) + alpha).sum() # p(ws)
    p_1_w = (vocabulary_statistics["frequency",1] + alpha) / (vocabulary_statistics["frequency"] + alpha).sum(axis=1) # p(y=1|w)
    p_0_w =  (vocabulary_statistics["frequency",0] + alpha) / (vocabulary_statistics["frequency"] + alpha).sum(axis=1) # p(y=0|w)
    p_1_w_s = (vocabulary_statistics["sample_frequency",1] + alpha) / (vocabulary_statistics["sample_frequency"] + alpha).sum(axis=1) # p(y=1|ws)
    p_0_w_s =  (vocabulary_statistics["sample_frequency",0] + alpha) / (vocabulary_statistics["sample_frequency"] + alpha).sum(axis=1) # p(y=0|ws)
    ## Derived Probabilities
    p_1w = p_w * p_1_w # p(w,y=1) = p(w) * p(y=1|w)
    p_1w_s = p_w_s * p_1_w_s # p(ws, y=1) = p(ws) * p(y=1|ws)
    ## Scores
    vocabulary_scores = {
        "frequency":vocabulary_statistics["frequency"].sum(axis=1),
        "sample_frequency":vocabulary_statistics["sample_frequency"].sum(axis=1),
        "pmi": np.log(p_w_1 / p_w),
        "pmi_support": np.log(p_w_s_1 / p_w_s),
        "normalized_pmi": np.divide(np.log(p_w_1 / p_w),  -np.log(p_1w)),
        "normalized_pmi_support": np.divide(np.log(p_w_s_1 / p_w_s), -np.log(p_1w_s)),
        "max_pmi": np.where(np.log(p_w_1 / p_w) > np.log(p_w_0 / p_w), np.log(p_w_1 / p_w), np.log(p_w_0 / p_w)),
        "max_pmi_support": np.where(np.log(p_w_s_1 / p_w_s) > np.log(p_w_s_0 / p_w_s), np.log(p_w_s_1 / p_w_s), np.log(p_w_s_0 / p_w_s)),
        "modified_pmi" : p_w_1 * np.log(p_w_1 / p_w),
        "modified_pmi_support": p_w_s_1 * np.log(p_w_s_1 / p_w_s)
    }
    vocabulary_scores = pd.DataFrame(vocabulary_scores)
    return vocabulary_scores

def run_classification_baseline(X,
                                y,
                                tau,
                                groups,
                                vocabulary,
                                frequency,
                                analysis_config,
                                time_period_configs,
                                enforce_frequency_filter=False, ## Vocabulary Frequency Filtering
                                enforce_classwise_frequency_filter=False, ## Requires Both Classes to Meet Frequency Filtering Criteria (Not Total of Both)
                                enforce_min_shift_frequency_filter=False, ## Requires Data to Abide by Learned Data Frequencies
                                compute_cumulative=False, ## Computes weights/scores for models that aren't used for vocabulary selection
                                compute_discrete=False,
                                njobs=1,
                                random_state=42):
    """
    
    """
    ## Discrete vs. Cumulative Periods
    cumulative_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("cumulative/")]
    discrete_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("discrete/")]
    ## Frequency Filtering (Sample Specific)
    if enforce_classwise_frequency_filter:
        if not isinstance(analysis_config["classifier_min_vocab_freq"], list):
            analysis_config["classifier_min_vocab_freq"] = list(map(lambda i: analysis_config["classifier_min_vocab_freq"] if analysis_config["classifier_min_vocab_freq"] is not None else 1, range(2)))
        if not isinstance(analysis_config["classifier_min_vocab_sample_freq"], list):
            analysis_config["classifier_min_vocab_sample_freq"] = list(map(lambda i: analysis_config["classifier_min_vocab_sample_freq"] if analysis_config["classifier_min_vocab_sample_freq"] is not None else 1, range(2)))
        for key in ["classifier_min_vocab_freq","classifier_min_vocab_sample_freq"]:
            if len(analysis_config[key]) != 2:
                raise ValueError("Classwise Frequency Filter Should Have Length 2")
    else:
        for key in ["classifier_min_vocab_freq","classifier_min_vocab_sample_freq"]:
            if isinstance(analysis_config[key], list):
                raise TypeError("No classwise frequency filtering means these filter values should not be lists.")
    if enforce_frequency_filter:
        min_freq = analysis_config["classifier_min_vocab_freq"]
        min_sample_freq = analysis_config["classifier_min_vocab_sample_freq"]
    elif not enforce_frequency_filter and enforce_classwise_frequency_filter:
        min_freq = [1, 1]
        min_sample_freq = [1, 1]
    elif not enforce_frequency_filter and not enforce_classwise_frequency_filter:
        min_freq = 1
        min_sample_freq = 1
    else:
        raise ValueError("Something went wrong with this logic.")
    ## Initialize Score and Coefficient Cache
    scores = []
    weights = []
    vocabulary_scores = []
    ## Identify Which Models Have Been Fit
    models_fit = set()
    ## Cycle Through Combinations
    for i, cind in enumerate(cumulative_time_period_configs):
        ## Isolate Relevant Data Indices
        cumulative_train_mask = [m for m, (t, g) in enumerate(zip(tau, groups)) if t == cind and g == 0]
        cumulative_test_mask = [m for m, (t, g) in enumerate(zip(tau, groups)) if t == cind and g == 1]
        ## Isolate Relevant Data
        X_train_cumulative, y_train_cumulative = X[cumulative_train_mask], y[cumulative_train_mask]
        X_test_cumulative, y_test_cumulative = X[cumulative_test_mask], y[cumulative_test_mask]
        ## Vocabulary Filtering (Cumulative Time-Period)
        if enforce_min_shift_frequency_filter:
            cumulative_freq_vmask = (frequency[cind] >= analysis_config["min_shift_freq"])
        else:
            cumulative_freq_vmask = (frequency[cind] >= 0)
        ## Vocabulary Filtering (Sample Specific)
        if enforce_classwise_frequency_filter:
            cumulative_vmask = np.logical_and.reduce([(X_train_cumulative[y_train_cumulative == 0].sum(axis=0) >= min_freq[0]).A[0],
                                                      (X_train_cumulative[y_train_cumulative == 1].sum(axis=0) >= min_freq[1]).A[0],
                                                      ((X_train_cumulative[y_train_cumulative == 0] != 0).sum(axis=0) >= min_sample_freq[0]).A[0],
                                                      ((X_train_cumulative[y_train_cumulative == 1] != 0).sum(axis=0) >= min_sample_freq[1]).A[0]])
        else:
            cumulative_vmask = np.logical_and((X_train_cumulative.sum(axis=0) >= min_freq).A[0],
                                              ((X_train_cumulative != 0).sum(axis=0) >= min_sample_freq).A[0])
        cumulative_vmask = np.logical_and(cumulative_freq_vmask, cumulative_vmask).nonzero()[0]
        ## Format Vocabulary
        cumulative_vocabulary = format_vocabulary([vocabulary[m] for m in cumulative_vmask])
        ## Fit Classifier
        if compute_cumulative:
            cumulative_preprocessor, cumulative_model, cumulative_scores_train = fit_classifier(X_train_cumulative[:,cumulative_vmask], y_train_cumulative, cumulative_vocabulary, analysis_config, njobs, random_state)
            cumulative_scores_test = evaluate_classifier(X_test_cumulative[:,cumulative_vmask], y_test_cumulative, cumulative_preprocessor, cumulative_model)
            ## Format Scores
            cumulative_scores = [cumulative_scores_train, cumulative_scores_test]
            for cs, group, y_true in zip(cumulative_scores, ["train","test"], [y_train_cumulative, y_test_cumulative]):
                cs["time_period_train"] = cind
                cs["time_period_apply"] = cind
                cs["group"] = group
                cs["support"] = y_true.shape[0]
                cs["class_balance"] = y_true.mean()
                cs["time_period_train_type"] = "cumulative"
                cs["time_period_apply_type"] = "cumulative"
                cs["vocabulary_id"] = "cumulative"
        ## Extract Feature Weights (or Create Dummys)
        if compute_cumulative:
            cumulative_weights = pd.Series(cumulative_model.coef_[0], [" ".join(f) for f in get_feature_names(cumulative_preprocessor)]).to_frame("weight")
        else:
            cumulative_weights = pd.Series(np.zeros(len(cumulative_vmask)), [" ".join(f) for f in cumulative_vocabulary]).to_frame("weight")
        cumulative_weights["time_period_train"] = cind
        cumulative_weights["time_period_train_type"] = "cumulative"
        cumulative_weights["time_period_apply"] = cind
        cumulative_weights["time_period_apply_type"] = "cumulative"
        cumulative_weights["support"] = X_train_cumulative[:,cumulative_vmask].sum(axis=0).A[0].astype(int)
        cumulative_weights["sample_support"] = (X_train_cumulative[:,cumulative_vmask] != 0).sum(axis=0).A[0].astype(int)
        cumulative_weights["vocabulary_id"] = "cumulative"
        ## Cumulative Vocabulary Scores
        if compute_cumulative:
            cumulative_chi2, _ = feature_selection.chi2(X_train_cumulative[:,cumulative_vmask], y_train_cumulative)
            cumulative_vocabulary_statistics = compute_vocabulary_statistics(X=X_train_cumulative, y=y_train_cumulative, vmask=cumulative_vmask, vocabulary=cumulative_vocabulary)
            cumulative_vocabulary_scores = compute_vocabulary_scores(cumulative_vocabulary_statistics, alpha=1)
            cumulative_vocabulary_scores["chi2"] = cumulative_chi2
            cumulative_vocabulary_scores["time_period_train"] = cind
            cumulative_vocabulary_scores["time_period_train_type"] = "cumulative"
            cumulative_vocabulary_scores["time_period_apply"] = cind
            cumulative_vocabulary_scores["time_period_apply_type"] = "cumulative"
            cumulative_vocabulary_scores["vocabulary_id"] = "cumulative"
        ## Cache Scores and Weights
        weights.append(cumulative_weights)
        if compute_cumulative:
            scores.extend(cumulative_scores)
            vocabulary_scores.append(cumulative_vocabulary_scores)
        ## Update Models Fit
        models_fit.add(cind)
        ## Iterate Through Discrete Time Periods
        for j, dind in enumerate(discrete_time_period_configs):
            ## Skip Historical Data
            if j < i:
                continue
            ## Isolate Relevant Data Indices
            discrete_train_mask = [m for m, (t, g) in enumerate(zip(tau, groups)) if t == dind and g == 0]
            discrete_test_mask = [m for m, (t, g) in enumerate(zip(tau, groups)) if t == dind and g == 1]
            ## Isolate Relevant Data
            X_train_discrete, y_train_discrete = X[discrete_train_mask], y[discrete_train_mask]
            X_test_discrete, y_test_discrete = X[discrete_test_mask], y[discrete_test_mask]
            ## Apply and Evaluate Cumulative Classifier on Discrete Data
            if compute_cumulative:
                discrete_scores_test_cumulative_model = evaluate_classifier(X_test_discrete[:,cumulative_vmask], y_test_discrete, cumulative_preprocessor, cumulative_model)
                discrete_scores_test_cumulative_model["time_period_train"] = cind
                discrete_scores_test_cumulative_model["time_period_apply"] = dind
                discrete_scores_test_cumulative_model["group"] = group
                discrete_scores_test_cumulative_model["support"] = y_test_discrete.shape[0]
                discrete_scores_test_cumulative_model["class_balance"] = y_test_discrete.mean()
                discrete_scores_test_cumulative_model["time_period_train_type"] = "cumulative"
                discrete_scores_test_cumulative_model["time_period_apply_type"] = "discrete"
                discrete_scores_test_cumulative_model["vocabulary_id"] = "cumulative"
                ## Cache scores
                scores.append(discrete_scores_test_cumulative_model)
            ## Vocabulary Filtering (Discrete Time-Period)
            if enforce_min_shift_frequency_filter:
                discrete_freq_vmask = (frequency[dind] >= analysis_config["min_shift_freq"])
            else:
                discrete_freq_vmask = (frequency[dind] >= 0)
            ## Vocabulary Filtering (Sample Specific)
            if enforce_classwise_frequency_filter:
                discrete_vmask = np.logical_and.reduce([(X_train_discrete[y_train_discrete == 0].sum(axis=0) >= min_freq[0]).A[0],
                                                        (X_train_discrete[y_train_discrete == 1].sum(axis=0) >= min_freq[1]).A[0],
                                                        ((X_train_discrete[y_train_discrete == 0] != 0).sum(axis=0) >= min_sample_freq[0]).A[0],
                                                        ((X_train_discrete[y_train_discrete == 1] != 0).sum(axis=0) >= min_sample_freq[1]).A[0]])
            else:
                discrete_vmask = np.logical_and((X_train_discrete.sum(axis=0) >= min_freq).A[0],
                                               ((X_train_discrete != 0).sum(axis=0) >= min_sample_freq).A[0])
            discrete_vmask = np.logical_and(discrete_freq_vmask, discrete_vmask).nonzero()[0]
            combined_vmask = np.array(sorted(set(cumulative_vmask) & set(discrete_vmask)))
            ## Format Vocabulary
            discrete_vocabulary = format_vocabulary([vocabulary[m] for m in discrete_vmask])
            combined_vocabulary = format_vocabulary([vocabulary[m] for m in combined_vmask])
            ## Train and Evaluate Model using Combined Vocabulary
            combined_preprocessor, combined_model, combined_scores_train = fit_classifier(X_train_cumulative[:,combined_vmask], y_train_cumulative, combined_vocabulary, analysis_config, njobs, random_state)
            combined_scores_cumulative_test = evaluate_classifier(X_test_cumulative[:,combined_vmask], y_test_cumulative, combined_preprocessor, combined_model)
            combined_scores_discrete_test = evaluate_classifier(X_test_discrete[:,combined_vmask], y_test_discrete, combined_preprocessor, combined_model)
            ## Format Scores
            combined_scores = [combined_scores_train, combined_scores_cumulative_test, combined_scores_discrete_test]
            for cs, group, y_true, tp_test in zip(combined_scores, ["train","test","test"], [y_train_cumulative, y_test_cumulative, y_test_discrete], [(cind,"cumulative"),(cind,"cumulative"),(dind,"discrete")]):
                cs["time_period_train"] = cind
                cs["time_period_apply"] = tp_test[0]
                cs["group"] = group
                cs["support"] = y_true.shape[0]
                cs["class_balance"] = y_true.mean()
                cs["time_period_train_type"] = "cumulative"
                cs["time_period_apply_type"] = tp_test[1]
                cs["vocabulary_id"] = "intersection"
            ## Cache Scores
            scores.extend(combined_scores)
            models_fit.add((cind,dind))
            ## Weights
            combined_weights = pd.Series(combined_model.coef_[0], [" ".join(f) for f in get_feature_names(combined_preprocessor)]).to_frame("weight")
            combined_weights["time_period_train"] = cind
            combined_weights["time_period_train_type"] = "cumulative"
            combined_weights["time_period_apply"] = dind
            combined_weights["time_period_apply_type"] = "discrete"
            combined_weights["support"] = X_train_cumulative[:,combined_vmask].sum(axis=0).A[0].astype(int)
            combined_weights["sample_support"] = (X_train_cumulative[:,combined_vmask] != 0).sum(axis=0).A[0].astype(int)
            combined_weights["vocabulary_id"] = "intersection"
            weights.append(combined_weights)
            ## Vocabulary Scores
            combined_chi2, _ = feature_selection.chi2(X_train_cumulative[:,combined_vmask], y_train_cumulative)
            combined_vocabulary_statistics = compute_vocabulary_statistics(X=X_train_cumulative, y=y_train_cumulative, vmask=combined_vmask, vocabulary=combined_vocabulary)
            combined_vocabulary_scores = compute_vocabulary_scores(combined_vocabulary_statistics, alpha=1)
            combined_vocabulary_scores["chi2"] = combined_chi2
            combined_vocabulary_scores["time_period_train"] = cind
            combined_vocabulary_scores["time_period_train_type"] = "cumulative"
            combined_vocabulary_scores["time_period_apply"] = dind
            combined_vocabulary_scores["time_period_apply_type"] = "discrete"
            combined_vocabulary_scores["vocabulary_id"] = "intersection"
            vocabulary_scores.append(combined_vocabulary_scores)
            ## Fit Within-Domain Classifier (if Necessary)
            if dind not in models_fit and compute_discrete:
                ## Train/Evaluate
                discrete_preprocessor, discrete_model, discrete_scores_train_discrete_model = fit_classifier(X_train_discrete[:,discrete_vmask], y_train_discrete, discrete_vocabulary, analysis_config, njobs, random_state)
                discrete_scores_test_discrete_model = evaluate_classifier(X_test_discrete[:,discrete_vmask], y_test_discrete, discrete_preprocessor, discrete_model)
                ## Extract Feature Weights
                discrete_weights = pd.Series(discrete_model.coef_[0], [" ".join(f) for f in get_feature_names(discrete_preprocessor)]).to_frame("weight")
                discrete_weights["time_period_train"] = dind
                discrete_weights["time_period_train_type"] = "discrete"
                discrete_weights["time_period_apply"] = dind
                discrete_weights["time_period_apply_type"] = "discrete"
                discrete_weights["support"] = X_train_discrete[:,discrete_vmask].sum(axis=0).A[0].astype(int)
                discrete_weights["sample_support"] = (X_train_discrete[:,discrete_vmask] != 0).sum(axis=0).A[0].astype(int)
                discrete_weights["vocabulary_id"] = "discrete"
                ## Format Scores
                discrete_scores_discrete_model = [discrete_scores_train_discrete_model, discrete_scores_test_discrete_model]
                for cs, group, y_true in zip(discrete_scores_discrete_model, ["train","test"], [y_train_discrete, y_test_discrete]):
                    cs["time_period_train"] = dind
                    cs["time_period_apply"] = dind
                    cs["group"] = group
                    cs["support"] = y_true.shape[0]
                    cs["class_balance"] = y_true.mean()
                    cs["time_period_train_type"] = "discrete"
                    cs["time_period_apply_type"] = "discrete"
                    cs["vocabulary_id"] = "discrete"
                ## Vocabulary Scores
                discrete_chi2, _ = feature_selection.chi2(X_train_discrete[:,discrete_vmask], y_train_discrete)
                discrete_vocabulary_statistics = compute_vocabulary_statistics(X=X_train_discrete, y=y_train_discrete, vmask=discrete_vmask, vocabulary=discrete_vocabulary)
                discrete_vocabulary_scores = compute_vocabulary_scores(discrete_vocabulary_statistics, alpha=1)
                discrete_vocabulary_scores["chi2"] = discrete_chi2
                discrete_vocabulary_scores["time_period_train"] = dind
                discrete_vocabulary_scores["time_period_train_type"] = "discrete"
                discrete_vocabulary_scores["time_period_apply"] = dind
                discrete_vocabulary_scores["time_period_apply_type"] = "discrete"
                discrete_vocabulary_scores["vocabulary_id"] = "discrete"
                ## Cache
                scores.extend(discrete_scores_discrete_model)
                weights.append(discrete_weights)
                vocabulary_scores.append(discrete_vocabulary_scores)
                ## Update Models Fit
                models_fit.add(dind)
    ## Format Scores
    scores = pd.DataFrame(scores)
    ## Format Weights
    weights = pd.concat(weights)
    weights = weights.reset_index().rename(columns={"index":"feature"})
    ## Format Vocabulary Scores
    vocabulary_scores = pd.concat(vocabulary_scores, sort=False)
    vocabulary_scores = vocabulary_scores.reset_index().rename(columns={"term":"feature"})
    return scores, weights, vocabulary_scores

def get_selectors(analysis_config,
                  use_support=False):
    """

    """
    ## Vocabulary Selectors
    selectors = [
        "cumulative",
        "intersection",
        "random",
        "frequency",
        "chi2",
        "coefficient",
        "overlap",
    ]
    if isinstance(analysis_config["selection_beta"], list):
        for beta in analysis_config["selection_beta"]:
            selectors.append(f"weighted-{beta}")
    else:
        selectors.append("weighted")
    if use_support:
        selectors = list(map(lambda s: s if "pmi" not in s else f"{s}_support", selectors))
    ## Color Map
    selector2col = dict(zip(selectors, ["navy","black"] + [f"C{i}" for i in range(len(selectors)-2)]))
    return selectors, selector2col

def run_vocabulary_selection(vocabulary,
                             frequency,
                             similarity,
                             baseline_classification_weights,
                             baseline_vocabulary_scores,
                             time_period_configs,
                             analysis_config,
                             random_state=42):
    """
    
    """
    ## Support Filtering
    support_filter = baseline_classification_weights.groupby(["time_period_train","time_period_apply","feature"]).agg({"weight":[len]})["weight"]
    support_filter = support_filter.loc[support_filter["len"] >= (analysis_config["classifier_k_models"] * analysis_config["selection_sample_support"])]
    ## Aggregated Weights
    if not analysis_config["selection_sample_specific"]:
        baseline_classification_weights_agg = baseline_classification_weights.groupby(["time_period_train","time_period_apply","feature"]).agg({"weight":np.mean})["weight"]
        baseline_vocabulary_scores_agg = baseline_vocabulary_scores.groupby(["time_period_train","time_period_apply","feature"]).mean().drop(["fold"],axis=1)
    ## Vocabulary Mapping
    vocab2ind = dict(zip(vocabulary, range(len(vocabulary))))
    f_vocab2ind = lambda terms: list(map(lambda term: vocab2ind[term], terms))
    ## Discrete vs. Cumulative Periods
    cumulative_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("cumulative/")]
    discrete_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("discrete/")]
    ## Selectors
    selectors, _ = get_selectors(analysis_config)
    ## Initialize Cache
    vocabulary_selection = {}
    vocabulary_sizes = {}
    for i, cind in tqdm(enumerate(cumulative_time_period_configs), desc="[Cumulative Time Period]", total=len(cumulative_time_period_configs), file=sys.stdout, position=0, leave=True):
        for j, dind in tqdm(enumerate(discrete_time_period_configs), desc="[Discrete Time Period]", total=len(discrete_time_period_configs), file=sys.stdout, position=1, leave=False):
            ## Skip Historical
            if j < i:
                continue
            ## Initialize Combination Cache
            vocabulary_sizes[(cind, dind)] = []
            vocabulary_selection[(cind, dind)] = {
                "cumulative":[],
                "intersection":[],
            }
            for percentile in analysis_config["vocab_percentiles"]:
                for prefix in selectors:
                    if prefix in ["cumulative","intersection"]:
                        continue
                    vocabulary_selection[(cind, dind)][f"{prefix}_{percentile}"] = []
            ## Combine Data
            cd_df = [
                frequency[cind],
                frequency[dind],
                similarity[i, j],
            ]
            cd_df = pd.DataFrame(cd_df, columns=vocabulary, index=["cumulative_frequency","discrete_frequency","overlap"]).T
            ## Filter Vocabulary
            cd_df = cd_df.loc[cd_df.index.isin(set(support_filter.loc[cind,cind].index))]
            cd_df = cd_df.fillna(0)
            ## Initialize Random Seed
            seed = np.random.RandomState(random_state)
            ## Iterate Through Samples
            for sample in tqdm(range(analysis_config["classifier_k_models"]), total=analysis_config["classifier_k_models"], file=sys.stdout, desc="[Sample]", position=2, leave=False) :
                ## Sample Copy
                sample_cd_df = cd_df.copy()
                ## Generate Random Numbers
                sample_cd_df["random"] = seed.normal(size=sample_cd_df.shape[0])
                ## Get Weights to Use
                if not analysis_config["selection_sample_specific"]:
                    sample_cd_df["weight"] = baseline_classification_weights_agg.loc[cind, dind].reindex(sample_cd_df.index)
                    sample_cd_df = pd.merge(sample_cd_df,
                                            baseline_vocabulary_scores_agg.loc[cind, dind].reindex(sample_cd_df.index),
                                            left_index=True,
                                            right_index=True)
                    
                else:
                    baseline_sample_c_df = baseline_classification_weights.loc[(baseline_classification_weights["time_period_train"]==cind)&
                                                                               (baseline_classification_weights["time_period_apply"]==dind)&
                                                                               (baseline_classification_weights["fold"]==sample)]
                    baseline_vocabulary_scores_sample_df = baseline_vocabulary_scores.loc[(baseline_vocabulary_scores["time_period_train"]==cind)&
                                                                                          (baseline_vocabulary_scores["time_period_apply"]==dind)&
                                                                                          (baseline_vocabulary_scores["fold"]==sample)]
                    sample_cd_df["weight"] = baseline_sample_c_df.set_index("feature").reindex(sample_cd_df.index)["weight"].values
                    sample_cd_df = pd.merge(sample_cd_df,
                                            baseline_vocabulary_scores_sample_df.set_index("feature").drop(["fold","frequency","sample_frequency"],axis=1),
                                            left_index=True,
                                            right_index=True,
                                            how="left")
                ## Cumultive Baseline                
                cumulative_baseline = sample_cd_df.loc[sample_cd_df["cumulative_frequency"] >= analysis_config["min_shift_freq"]].index.tolist()
                ## Align Starting Point
                sample_cd_df = sample_cd_df.dropna()
                ## Baselines
                intersection_baseline = sample_cd_df.loc[(sample_cd_df["cumulative_frequency"] >= analysis_config["min_shift_freq"])&(sample_cd_df["discrete_frequency"] >= analysis_config["min_shift_freq"])].index.tolist()
                vocabulary_selection[(cind, dind)]["cumulative"].append(sorted(f_vocab2ind(cumulative_baseline)))
                vocabulary_selection[(cind, dind)]["intersection"].append(sorted(f_vocab2ind(intersection_baseline)))
                ## Isolate Intersection
                sample_cd_df = sample_cd_df.loc[intersection_baseline].copy()
                ## Ranking
                sample_cd_df["weight_rank"] = sample_cd_df["weight"].map(abs).rank(method="min").astype(int)
                sample_cd_df["overlap_rank"] = sample_cd_df["overlap"].rank(method="min").astype(int)
                sample_cd_df["random_rank"] = sample_cd_df["random"].rank(method="min").astype(int)
                score_selectors = []
                for sel in selectors:
                    if "pmi" not in sel and "chi2" not in sel:
                        continue
                    score_selectors.append(sel)
                    sample_cd_df[f"{sel}_rank"] = sample_cd_df[sel].rank(method="min").astype(int)
                ## Weighted Ranking (Overlap + Coefficient)
                if not isinstance(analysis_config["selection_beta"], list):
                    selector_weighted = [("weighted", "weight_overlap_rank")]
                    sample_cd_df["weight_overlap_rank"] = (analysis_config["selection_beta"] * sample_cd_df["overlap_rank"] + (1 - analysis_config["selection_beta"]) * sample_cd_df["weight_rank"]).rank(method="dense").astype(int)
                else:
                    selector_weighted = []
                    for b, beta in enumerate(analysis_config["selection_beta"]):
                        sample_cd_df[f"weight_overlap_rank_{b}"] = (beta * sample_cd_df["overlap_rank"] + (1 - beta) * sample_cd_df["weight_rank"]).rank(method="dense").astype(int)
                        selector_weighted.append((f"weighted-{beta}", f"weight_overlap_rank_{b}"))
                ## Sizes
                sample_sizes = list(map(int,np.percentile(list(range(sample_cd_df.shape[0])), analysis_config["vocab_percentiles"])))
                vocabulary_sizes[(cind, dind)].append([len(cumulative_baseline),len(intersection_baseline)]  + sample_sizes)
                ## Iterate Through Sizes
                for percentile, size in zip(analysis_config["vocab_percentiles"], sample_sizes):
                    ## Compute and Store
                    vocabulary_selection[(cind, dind)][f"random_{percentile}"].append(sorted(f_vocab2ind(sample_cd_df["random_rank"].nlargest(size).index)))
                    vocabulary_selection[(cind, dind)][f"frequency_{percentile}"].append(sorted(f_vocab2ind(sample_cd_df["cumulative_frequency"].nlargest(size).index)))
                    vocabulary_selection[(cind, dind)][f"overlap_{percentile}"].append(sorted(f_vocab2ind(sample_cd_df["overlap_rank"].nlargest(size).index)))
                    vocabulary_selection[(cind, dind)][f"coefficient_{percentile}"].append(sorted(f_vocab2ind(sample_cd_df["weight_rank"].nlargest(size).index)))
                    for prefix, col in selector_weighted:
                        vocabulary_selection[(cind, dind)][f"{prefix}_{percentile}"].append(sorted(f_vocab2ind(sample_cd_df[col].nlargest(size).index)))
                    for scoresel in score_selectors:
                        vocabulary_selection[(cind, dind)][f"{scoresel}_{percentile}"].append(sorted(f_vocab2ind(sample_cd_df[f"{scoresel}_rank"].nlargest(size).index)))
    ## Format Sizes
    vocabulary_sizes = {x:np.array(y) for x, y in vocabulary_sizes.items()}
    vocabulary_sizes = pd.concat({x:pd.DataFrame(y,columns=["cumulative","intersection"]+analysis_config["vocab_percentiles"]) for x, y in vocabulary_sizes.items()})
    ## Return
    return vocabulary_selection, vocabulary_sizes

def run_classification_reduced_vocabulary(sample,
                                          X,
                                          y,
                                          tau,
                                          groups,
                                          vocabulary,
                                          selected_vocabularies,
                                          analysis_config,
                                          time_period_configs,
                                          njobs=1,
                                          random_state=42):
    """
    
    """
    ## Discrete vs. Cumulative Periods
    cumulative_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("cumulative/")]
    discrete_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("discrete/")]
    ## Score Cache
    scores = []
    ## Cycle Through Combinations
    for i, cind in tqdm(enumerate(cumulative_time_period_configs), desc="[Cumulative Period]", file=sys.stdout, total=len(cumulative_time_period_configs), position=1, leave=False):
        ## Isolate Relevant Data Indices
        cumulative_train_mask = [m for m, (t, g) in enumerate(zip(tau, groups)) if t == cind and g == 0]
        cumulative_test_mask = [m for m, (t, g) in enumerate(zip(tau, groups)) if t == cind and g == 1]
        ## Isolate Relevant Data
        X_train_cumulative, y_train_cumulative = X[cumulative_train_mask], y[cumulative_train_mask]
        X_test_cumulative, y_test_cumulative = X[cumulative_test_mask], y[cumulative_test_mask]
        ## Cycle Through Discrete Periods
        for j, dind in tqdm(enumerate(discrete_time_period_configs), desc="[Discrete Period]", file=sys.stdout, total=len(discrete_time_period_configs), position=2, leave=False):
            ## Skip Historical Data
            if j < i:
                continue
            ## Isolate Relevant Data Indices
            discrete_test_mask = [m for m, (t, g) in enumerate(zip(tau, groups)) if t == dind and g == 1]
            ## Isolate Relevant Data
            X_test_discrete, y_test_discrete = X[discrete_test_mask], y[discrete_test_mask]
            ## Set of Vocabularies
            cd_selected_vocabularies = {p:q[sample] for p,q in selected_vocabularies[(cind, dind)].items()}
            ## Iterate Through Vocabularies
            for vocabulary_id, vocabulary_indices in tqdm(cd_selected_vocabularies.items(), desc="[Vocabulary ID]", file=sys.stdout, total=len(cd_selected_vocabularies), position=3, leave=False):
                ## Get Desired Terms
                cd_vocabulary = format_vocabulary([vocabulary[ind] for ind in vocabulary_indices])
                ## Fit Model and Evaluation
                cd_preprocessor, cd_model, cd_cumulative_scores_train = fit_classifier(X_train_cumulative[:, vocabulary_indices], y_train_cumulative, cd_vocabulary, analysis_config, njobs, random_state)
                cd_cumulative_scores_test = evaluate_classifier(X_test_cumulative[:, vocabulary_indices], y_test_cumulative, cd_preprocessor, cd_model)
                cd_discrete_scores_test = evaluate_classifier(X_test_discrete[:, vocabulary_indices], y_test_discrete, cd_preprocessor, cd_model)
                ## Format Scores
                cd_scores = [cd_cumulative_scores_train, cd_cumulative_scores_test, cd_discrete_scores_test]
                for cs, group, t_type, y_true in zip(cd_scores,
                                                     ["train","test","test"],
                                                     ["cumulative","cumulative","discrete"],
                                                     [y_train_cumulative, y_test_cumulative, y_test_discrete]):
                    cs["time_period_train"] = cind
                    cs["time_period_apply"] = cind if t_type == "cumulative" else dind
                    cs["group"] = group
                    cs["support"] = y_true.shape[0]
                    cs["class_balance"] = y_true.mean()
                    cs["time_period_train_type"] = "cumulative"
                    cs["time_period_apply_type"] = t_type
                    cs["vocabulary_id"] = vocabulary_id
                    cs["vocabulary_reference_train"] = cind
                    cs["vocabulary_reference_apply"] = dind
                    cs["fold"] = sample
                ## Cache Scores
                scores.extend(cd_scores)
    ## Format Scores
    scores = pd.DataFrame(scores)
    return scores

def describe_shift(global_vocabulary,
                   neighbors,
                   similarity,
                   frequency,
                   time_period_configs,
                   min_shift_freq=100,
                   top_k=50,
                   top_k_neighbors=20):
    """

    """
    ## Discrete vs. Cumulative Periods
    cumulative_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("cumulative/")]
    discrete_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("discrete/")]
    ## Output String
    output_str = []
    ## Iterate Through Combinations
    ind2term = lambda inds: ", ".join([global_vocabulary[ind] for ind in inds]) if np.min(inds) != -1 else None
    for i, cind in enumerate(cumulative_time_period_configs):
        ## Get Neighbors and Frequency
        cneighbors = [ind2term(row) for row in neighbors[cind][:,:top_k_neighbors]]
        cfreq = frequency[cind]
        for j, dind in enumerate(discrete_time_period_configs):
            ## Skip Historical
            if j < i:
                continue
            ## Update Output String
            output_str.append("~"*100 + f"\nCumulative {cind} | Discrete {dind}\n" + "~"*100)
            ## Get Neighbors and Frequency
            dneighbors = [ind2term(row) for row in neighbors[dind][:,:top_k_neighbors]]
            dfreq = frequency[dind]
            ## Get Similarity
            cd_similarity = similarity[i,j]
            ## Format
            cd_df = pd.DataFrame(columns=global_vocabulary,
                                 index=["neighbors_cumulative","neighbors_discrete","frequency_cumulative","frequency_discrete","overlap"],
                                 data=[cneighbors, dneighbors, cfreq, dfreq, cd_similarity]).T
            cd_df = cd_df.dropna(subset=["overlap"])
            cd_df = cd_df.loc[(cd_df["frequency_cumulative"]>=min_shift_freq)&(cd_df["frequency_discrete"]>=min_shift_freq)]
            ## Sort
            cd_df = cd_df.sort_values("overlap",ascending=False)
            ## Output
            for t, (term, term_data) in enumerate(cd_df.head(top_k).iterrows()):
                tstring = "{}) {} [Score = {}]\n\t[{}] {}\n\t[{}] {}".format(t+1, term, term_data["overlap"], cind, "\n\t".join(wrap(term_data["neighbors_cumulative"], 140)), dind, "\n\t".join(wrap(term_data["neighbors_discrete"], 140)))
                output_str.append(tstring)
            output_str.append("..." * 20)
            for t, (term, term_data) in enumerate(cd_df.tail(top_k).iloc[::-1].iterrows()):
                tstring = "{}) {} [Score = {}]\n\t[{}] {}\n\t[{}] {}".format(t+1, term, term_data["overlap"], cind, "\n\t".join(wrap(term_data["neighbors_cumulative"], 140)), dind, "\n\t".join(wrap(term_data["neighbors_discrete"], 140)))
                output_str.append(tstring)
    output_str = "\n".join(output_str)
    return output_str
            
def merge_coefficient_semantic_stability_data(vocabulary,
                                              similarity,
                                              frequency,
                                              baseline_classification_weights,
                                              analysis_config,
                                              time_period_configs,
                                              min_shift_freq=100):
    """

    """
    ## Discrete vs. Cumulative Periods
    cumulative_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("cumulative/")]
    discrete_time_period_configs = [i for i, j in enumerate(time_period_configs) if j["experiment_name"].startswith("discrete/")]
    ## Initialize Cache
    stability_df = []
    ## Iterate Through Time Periods
    for c, cind in enumerate(cumulative_time_period_configs):
        ## Iterate Through Discrete Time Periods
        for d, dind in enumerate(discrete_time_period_configs):
            if d < c:
                continue
            ## Get Relevant Weights
            cweights = baseline_classification_weights.loc[(baseline_classification_weights["time_period_train"] == cind)&
                                                           (baseline_classification_weights["time_period_apply"] == dind)]
            cweights_agg = cweights.groupby(["feature"]).agg({"sample_support":np.mean, "support":np.mean, "weight":[np.mean, np.std, len]})
            cweights_agg = cweights_agg.loc[cweights_agg[("weight","len")] >= (analysis_config["classifier_k_models"] * analysis_config["selection_sample_support"])]
            ## Initialize DataFrame and Filter
            cd_df = pd.DataFrame(data=[vocabulary,
                                       similarity[c, d],
                                       frequency[cind],
                                       frequency[dind]],
                                 index=["feature","overlap","frequency_cumulative","frequency_discrete"]).T.dropna().set_index("feature")
            cd_df = cd_df.loc[(cd_df.index.isin(cweights_agg.index))&
                              (cd_df["frequency_cumulative"]>=0)&
                              (cd_df["frequency_discrete"]>=0)].copy()
            ## Merge Weight Data
            cd_df["coefficient_mean"] = cweights_agg.loc[cd_df.index]["weight","mean"]
            cd_df["coefficient_std"] = cweights_agg.loc[cd_df.index]["weight","std"]
            cd_df["sample_support_mean"] = cweights_agg.loc[cd_df.index]["sample_support","mean"]
            cd_df["support_mean"] = cweights_agg.loc[cd_df.index]["support","mean"]
            ## Final Filter
            cd_df = cd_df.loc[(cd_df[["frequency_cumulative","frequency_discrete"]] >= min_shift_freq).all(axis=1)].copy()
            ## Meta
            cd_df["time_period_train"] = cind
            cd_df["time_period_apply"] = dind
            ## Cache
            stability_df.append(cd_df.reset_index())
    ## Concatenate
    stability_df = pd.concat(stability_df, axis=0, sort=False)
    return stability_df

def plot_stability_coefficient_relationship(coefficient_stability_summary,
                                            n_percentiles=20):
    """

    """
    ## Train/Apply Time Periods
    time_periods = coefficient_stability_summary[["time_period_train","time_period_apply"]].drop_duplicates().values
    ntrain, napply = len(set(time_periods[:,0])), len(set(time_periods[:,1]))
    ## Format Input Data
    coefficient_stability_summary = coefficient_stability_summary.copy()
    coefficient_stability_summary["coefficient_mean_abs"] = coefficient_stability_summary["coefficient_mean"].map(abs)
    ## Generate Plot
    fig = plt.figure(figsize=(ntrain * 5, napply * 3))
    gs = GridSpec(4 * ntrain, 4 * napply)
    for c, cind in enumerate(sorted(set(time_periods[:,0]))):
        for d, dind in enumerate(sorted(set(time_periods[:,1]))):
            if d < c:
                continue
            ## Get Data
            df = coefficient_stability_summary.set_index(["time_period_train","time_period_apply"]).sort_index().loc[(cind, dind)].copy()
            ## Data Prep
            pbins = np.percentile(df["overlap"].values,
                                  np.linspace(0, 1, n_percentiles+1) * 100)
            pbins = np.array(sorted(set(pbins)))
            pbins_loc = (pbins[:-1] + pbins[1:]) / 2
            df["percentile_overlap"] = pd.cut(df["overlap"],
                                            pbins,
                                            right=False,
                                            labels=pbins_loc)
            df_agg = bootstrap_tuple_to_df(df.groupby(["percentile_overlap"]).agg({"coefficient_mean_abs":bootstrap_ci})["coefficient_mean_abs"])
            df_agg["lower"] = df_agg["median"] - df_agg["lower"]
            df_agg["upper"] = df_agg["upper"] - df_agg["median"]
            ## Add Subplots
            ax_scatter = fig.add_subplot(gs[c*4 + 1: c*4 + 4, d*4 + 0:d*4 + 3])
            ax_hist_x = fig.add_subplot(gs[c*4 + 0, d*4 + 0: d*4 + 3], sharex=ax_scatter)
            ax_hist_y = fig.add_subplot(gs[c*4 + 1: c*4 + 4, d*4 + 3], sharey=ax_scatter)
            ## Add Plots
            ax_scatter.scatter(df['overlap'],
                               df['coefficient_mean_abs'],
                               alpha=0.2)
            ax_scatter.errorbar(df_agg.index,
                                df_agg["median"],
                                yerr=df_agg[["lower","upper"]].values.T,
                                color="darkred",
                                alpha=0.8)
            ax_hist_x.hist(df['overlap'],
                           orientation='vertical',
                           bins=100,
                           density=False,
                           weights=np.ones(df.shape[0]) / df.shape[0],
                           alpha=0.5)
            ax_hist_y.hist(df['coefficient_mean_abs'],
                           orientation="horizontal",
                           bins=100,
                           density=False,
                           weights=np.ones(df.shape[0]) / df.shape[0],
                           alpha=0.5)
            ax_scatter.set_ylim((df_agg["lower"] + df_agg["median"]).min() * 0.5,
                                (df_agg["upper"] + df_agg["median"]).max() * 1.25)
            for a in [ax_scatter, ax_hist_x, ax_hist_y]:
                a.spines["right"].set_visible(False)
                a.spines["top"].set_visible(False)
                _ = plt.setp(a.get_yticklabels(), visible=False)
                _ = plt.setp(a.get_xticklabels(), visible=False)
    fig.tight_layout()
    fig.text(0.5, 0.025, "Semantic Stability", fontweight="bold", ha="center", va="center", fontsize=12)
    fig.text(0.025, 0.5, "Absolute Coefficient Weight", fontweight="bold", ha="center", va="center", fontsize=12, rotation=90)
    fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, bottom=0.05)
    return fig

def run_classification(n,
                       analysis_directory,
                       analysis_config,
                       time_period_configs,
                       skip_existing=False,
                       enforce_frequency_filter=False,
                       enforce_classwise_frequency_filter=False,
                       enforce_min_shift_frequency_filter=False,
                       compute_cumulative=False,
                       compute_discrete=False,
                       njobs=1):
    """
    
    """
    ## Let User Know What Filters Are Being Used
    filter_str = f"Frequency: {enforce_frequency_filter}\nClasswise Frequency: {enforce_classwise_frequency_filter}\nMin Shift Frequency: {enforce_min_shift_frequency_filter}"
    LOGGER.info(f"[Frequency-Based Filters Being Used:]\n~~~~~~~~~~~~~~~~~~~~\n{filter_str}\n~~~~~~~~~~~~~~~~~~~~")
    ## Cache Directory
    result_cache_dir = f"{analysis_directory}/result-cache/{n}/"
    ## Check for Filenames
    all_exists = os.path.exists(result_cache_dir)
    if os.path.exists(result_cache_dir):
        for filename in ["baseline_classification_scores","baseline_classification_weights","reduced_vocabulary_classification_scores"]:
            if not os.path.exists(f"{result_cache_dir}/{filename}.csv"):
                all_exists = False
    ## Early Exit
    if all_exists and skip_existing:
        LOGGER.info("[Sample Classification Results Completed Already. Skipping.]")
        return None
    ## Initialize Directory if Necessary
    if not os.path.exists(result_cache_dir):
        _ = os.makedirs(result_cache_dir)
    ## Similarity Calculations
    LOGGER.info("[Aligning Vocabulary and Computing Semantic Similarity]")
    n_vocabulary, n_neighbors, n_similarity, n_frequency = align_and_compute_similarity(n,
                                                                                        analysis_directory,
                                                                                        analysis_config,
                                                                                        time_period_configs)
    ## Qualitative Similarity Analysis
    LOGGER.info("[Gathering Qualitative Shift Summary]")
    shift = describe_shift(global_vocabulary=n_vocabulary,
                           neighbors=n_neighbors,
                           similarity=n_similarity,
                           frequency=n_frequency,
                           time_period_configs=time_period_configs,
                           min_shift_freq=analysis_config["min_shift_freq"],
                           top_k=100,
                           top_k_neighbors=10)
    with open(f"{result_cache_dir}/shift.txt","w") as the_file:
        the_file.write(shift)
    ## Load Document Term Data
    LOGGER.info("[Loading Document Term Data]")
    n_X, n_y, n_tau, n_group, n_users = load_document_term_data(n=n,
                                                                vocabulary=n_vocabulary,
                                                                analysis_directory=analysis_directory,
                                                                time_period_configs=time_period_configs)
    ## Users in Each Sample
    LOGGER.info(f"[Loading Users in Sample: {n}]")
    n_sample_users = load_sample_users(n, analysis_directory, time_period_configs)
    ## Execute Multiple Samples
    LOGGER.info("[Running Baseline Classification Procedure]")
    baseline_classification_scores, baseline_classification_weights, baseline_vocabulary_scores = [], [], []
    for sample in tqdm(range(analysis_config["classifier_k_models"]),
                       desc="[Baseline Classification Models]",
                       total=analysis_config["classifier_k_models"],
                       file=sys.stdout):
        ## Generate a Sample of Users Based on Desired Balancing Parameters
        n_sample_split = sample_from_splits(n_sample_users, analysis_config, time_period_configs, random_state=sample)
        ## Filter Data
        n_X_sample, n_y_sample, n_tau_sample, n_group_sample = filter_document_term_data(n_X, n_y, n_tau, n_group, n_users, n_sample_split)
        ## Run Classification Baseline
        sample_baseline_classification_scores, sample_baseline_classification_weights, sample_baseline_vocabulary_scores = run_classification_baseline(X=n_X_sample,
                                                                                                                                                       y=n_y_sample,
                                                                                                                                                       tau=n_tau_sample,
                                                                                                                                                       groups=n_group_sample,
                                                                                                                                                       vocabulary=n_vocabulary,
                                                                                                                                                       frequency=n_frequency,
                                                                                                                                                       analysis_config=analysis_config,
                                                                                                                                                       time_period_configs=time_period_configs,
                                                                                                                                                       enforce_frequency_filter=enforce_frequency_filter,
                                                                                                                                                       enforce_classwise_frequency_filter=enforce_classwise_frequency_filter,
                                                                                                                                                       enforce_min_shift_frequency_filter=enforce_min_shift_frequency_filter,
                                                                                                                                                       compute_cumulative=compute_cumulative,
                                                                                                                                                       compute_discrete=compute_discrete,
                                                                                                                                                       njobs=njobs)
        ## Format
        sample_baseline_classification_scores["fold"] = sample
        sample_baseline_classification_weights["fold"] = sample
        sample_baseline_vocabulary_scores["fold"] = sample
        ## Cache
        baseline_classification_scores.append(sample_baseline_classification_scores)
        baseline_classification_weights.append(sample_baseline_classification_weights)
        baseline_vocabulary_scores.append(sample_baseline_vocabulary_scores)
    ## Format
    baseline_classification_scores = pd.concat(baseline_classification_scores, axis=0, sort=False).reset_index(drop=True)
    baseline_classification_weights = pd.concat(baseline_classification_weights, axis=0, sort=False).reset_index(drop=True)
    baseline_vocabulary_scores = pd.concat(baseline_vocabulary_scores, axis=0, sort=False).reset_index(drop=True)
    ## Build Vocabularies
    LOGGER.info("[Beginning Vocabulary Selection]")
    n_selected_vocabularies, n_selected_vocabulary_sizes = run_vocabulary_selection(vocabulary=n_vocabulary,
                                                                                    frequency=n_frequency,
                                                                                    similarity=n_similarity,
                                                                                    baseline_classification_weights=baseline_classification_weights,
                                                                                    baseline_vocabulary_scores=baseline_vocabulary_scores,
                                                                                    time_period_configs=time_period_configs,
                                                                                    analysis_config=analysis_config)
    LOGGER.info("[Vocabulary Sizes]")
    LOGGER.info(n_selected_vocabulary_sizes)
    ## Cache    
    LOGGER.info("[Caching Baseline Classification Results]")
    _ = baseline_classification_scores.to_csv(f"{result_cache_dir}/baseline_classification_scores.csv",index=False)
    _ = baseline_classification_weights.to_csv(f"{result_cache_dir}/baseline_classification_weights.csv",index=False)
    _ = baseline_vocabulary_scores.to_csv(f"{result_cache_dir}/baseline_vocabulary_scores.csv",index=False)
    _ = n_selected_vocabulary_sizes.to_csv(f"{result_cache_dir}/reduced_vocabulary_sizes.csv",index=True)
    ## Merge Weights and Stability Scores
    LOGGER.info("[Generating Coefficient - Stability Summary]")
    coefficient_stability_summary = merge_coefficient_semantic_stability_data(vocabulary=n_vocabulary,
                                                                              similarity=n_similarity,
                                                                              frequency=n_frequency,
                                                                              baseline_classification_weights=baseline_classification_weights,
                                                                              analysis_config=analysis_config,
                                                                              time_period_configs=time_period_configs,
                                                                              min_shift_freq=analysis_config["min_shift_freq"])
    _ = coefficient_stability_summary.to_csv(f"{result_cache_dir}/coefficient_stability_summary.csv",index=False)
    ## Visualize Stability/Coefficient Relationship
    LOGGER.info("[Plotting Coefficient - Stability Summary]")
    fig = plot_stability_coefficient_relationship(coefficient_stability_summary, 25)
    fig.savefig(f"{result_cache_dir}/coefficient_stability_summary.png",dpi=200)
    plt.close(fig)
    ## Reduced Vocabulary Experiments
    LOGGER.info("[Running Reduced Vocabulary Classification Procedure]")
    reduced_vocabulary_classification_scores = []
    for sample in tqdm(range(analysis_config["classifier_k_models"]),
                       desc="[Reduced Vocabulary Classification Models]",
                       total=analysis_config["classifier_k_models"],
                       file=sys.stdout,
                       position=0,
                       leave=True):
        ## Generate a Sample of Users Based on Desired Balancing Parameters
        n_sample_split = sample_from_splits(n_sample_users, analysis_config, time_period_configs, random_state=sample)
        ## Filter Data
        n_X_sample, n_y_sample, n_tau_sample, n_group_sample = filter_document_term_data(n_X, n_y, n_tau, n_group, n_users, n_sample_split)
        ## Run Classification with All Vocabularies
        sample_reduced_vocabulary_classification_scores = run_classification_reduced_vocabulary(sample=sample,
                                                                                                X=n_X_sample,
                                                                                                y=n_y_sample,
                                                                                                tau=n_tau_sample,
                                                                                                groups=n_group_sample,
                                                                                                vocabulary=n_vocabulary,
                                                                                                selected_vocabularies=n_selected_vocabularies,
                                                                                                analysis_config=analysis_config,
                                                                                                time_period_configs=time_period_configs,
                                                                                                njobs=njobs)
        ## Cache
        reduced_vocabulary_classification_scores.append(sample_reduced_vocabulary_classification_scores)
    ## Format
    reduced_vocabulary_classification_scores = pd.concat(reduced_vocabulary_classification_scores, sort=True, axis=0).reset_index(drop=True)
    ## Cache
    LOGGER.info("[Caching Reduced Vocabulary Classification Results]")
    _ = reduced_vocabulary_classification_scores.to_csv(f"{result_cache_dir}/reduced_vocabulary_classification_scores.csv",index=False)
    ## Add Metadata
    reduced_vocabulary_classification_scores["sample"] = sample
    reduced_vocabulary_classification_scores["vocabulary_selector"] = reduced_vocabulary_classification_scores["vocabulary_id"].map(lambda i: "_".join(i.split("_")[:-1]) if "_" in i else i)
    reduced_vocabulary_classification_scores["vocabulary_selector_percentile"] = reduced_vocabulary_classification_scores["vocabulary_id"].map(lambda i: int(i.split("_")[-1]) if "_" in i else -1)
    ## Visualization
    LOGGER.info("[Visualizing Reduced Vocabulary Classification Results]")
    for metric_type in analysis_config["metric_types"]:
        for w, domain_lbl in enumerate(["between-domain","within-domain"]):
            ## Plot Figure
            fig = plot_reduced_vocabulary_classification_performance(reduced_vocabulary_classification_scores,
                                                                     analysis_config=analysis_config,
                                                                     metric=metric_type,
                                                                     group="test",
                                                                     evaluate_within=w,
                                                                     include_samples=False,
                                                                     sharey=True)
            fig.savefig(f"{result_cache_dir}/scores.{domain_lbl}.{metric_type}.png")
            plt.close(fig)
    return None

def load_classification_scores(n_sample,
                               analysis_directory):
    """

    """
    ## Cache
    classification_scores = []
    classification_scores_missing = []
    ## Iterate Through Samples
    for sample in range(n_sample):
        ## Check for File
        sample_score_file = f"{analysis_directory}/result-cache/{sample}/reduced_vocabulary_classification_scores.csv"
        if not os.path.exists(sample_score_file):
            classification_scores_missing.append(sample)
            continue
        ## Load and Append Metadata
        sample_classification_scores = pd.read_csv(sample_score_file)
        sample_classification_scores["sample"] = sample
        ## store
        classification_scores.append(sample_classification_scores)
    ## Concatenate
    classification_scores = pd.concat(classification_scores, axis=0, sort=True).reset_index(drop=True)
    if len(classification_scores_missing) > 0:
        LOGGER.warning(f"[Warning: Score file not found for following samples: {classification_scores_missing}]")
    ## Format
    classification_scores["vocabulary_selector"] = classification_scores["vocabulary_id"].map(lambda i: "_".join(i.split("_")[:-1]) if "_" in i else i)
    classification_scores["vocabulary_selector_percentile"] = classification_scores["vocabulary_id"].map(lambda i: int(i.split("_")[-1]) if "_" in i else -1)
    return classification_scores

def load_stability_coefficient_relationship(n_sample,
                                            analysis_directory):
    """

    """
    ## Cache
    relationships = []
    relationships_missing = []
    ## Iterate Through Samples
    for sample in range(n_sample):
        ## Check for File
        sample_relationship_file = f"{analysis_directory}/result-cache/{sample}/coefficient_stability_summary.csv"
        if not os.path.exists(sample_relationship_file):
            relationships_missing.append(sample)
            continue
        ## Load and Append Metadata
        sample_relationships = pd.read_csv(sample_relationship_file)
        sample_relationships["sample"] = sample
        ## store
        relationships.append(sample_relationships)
    ## Concatenate
    relationships = pd.concat(relationships, axis=0, sort=True).reset_index(drop=True)
    if len(relationships_missing) > 0:
        LOGGER.warning(f"[Warning: Stability relationship file not found for following samples: {relationships_missing}]")
    return relationships

def _plot_reduced_vocabulary_classification_performance(subplot_reduced_vocabulary_classification_scores,
                                                        analysis_config,
                                                        metric="accuracy",
                                                        figure=None,
                                                        include_samples=False):
    """

    """
    ## Aggregate Data
    if include_samples:
        if "sample" not in subplot_reduced_vocabulary_classification_scores:
            subplot_reduced_vocabulary_classification_scores["sample"] = 1
        sample_aggs = bootstrap_tuple_to_df(subplot_reduced_vocabulary_classification_scores.groupby(["sample","vocabulary_selector","vocabulary_selector_percentile"]).agg({metric:bootstrap_ci})[metric])
    aggs = bootstrap_tuple_to_df(subplot_reduced_vocabulary_classification_scores.groupby(["vocabulary_selector","vocabulary_selector_percentile"]).agg({metric:bootstrap_ci})[metric])
    ## Selectors
    selectors, selectors2col = get_selectors(analysis_config)
    ## Figure
    if figure is None:
        fig, ax = plt.subplots(figsize=(10,5.8))
    else:
        fig, ax = figure
    ## Add Performance Visualization
    minx, maxx = 100, 0
    for selector in selectors:
        ## Sample Specific
        if include_samples:
            for sample in sample_aggs.index.levels[0]:
                sample_selector = sample_aggs.loc[sample, selector]
                if sample_selector.index.min() == -1:
                    ax.fill_between([0, 100],
                                    sample_selector["lower"].min(),
                                    sample_selector["upper"].max(),
                                    color=selectors2col[selector],
                                    alpha=0.05,
                                    zorder=-1)
                    ax.axhline(sample_selector["median"].min(),
                            color=selectors2col[selector],
                            alpha=0.1,
                            linestyle=":")
                else:
                    ax.fill_between(sample_selector.index.values,
                                    sample_selector["lower"].values,
                                    sample_selector["upper"].values,
                                    color=selectors2col[selector],
                                    alpha=0.05,
                                    zorder=-1)
                    ax.plot(sample_selector.index.values,
                            sample_selector["median"].values,
                            color=selectors2col[selector],
                            alpha=0.1,
                            linestyle=":",
                            marker="o")
        ## Global
        global_selector = aggs.loc[selector]
        if global_selector.index.min() == -1:
            ax.fill_between([0, 100],
                            global_selector["lower"].min(),
                            global_selector["upper"].max(),
                            color=selectors2col[selector],
                            alpha=0.25 if not include_samples else 0.5,
                            zorder=10)
            ax.axhline(global_selector["median"].min(),
                       color=selectors2col[selector],
                       alpha=0.5 if not include_samples else 0.8,
                       linestyle="-",
                       label=selector.title(),
                       zorder=10,
                       linewidth=2)
        else:
            minx = min(global_selector.index.min(), minx)
            maxx = max(global_selector.index.max(), maxx)
            ax.fill_between(global_selector.index.values,
                            global_selector["lower"].values,
                            global_selector["upper"].values,
                            color=selectors2col[selector],
                            alpha=0.25 if not include_samples else 0.5,
                            zorder=10)
            ax.plot(global_selector.index.values,
                    global_selector["median"].values,
                    color=selectors2col[selector],
                    alpha=0.5 if not include_samples else 0.8,
                    linestyle="-",
                    marker="o",
                    label=selector.title(),
                    zorder=10,
                    linewidth=2)
    ax.set_xlim(minx, maxx)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig, ax

def plot_reduced_vocabulary_classification_performance(reduced_vocabulary_classification_scores,
                                                       analysis_config,
                                                       metric="accuracy",
                                                       group="test",
                                                       evaluate_within=False,
                                                       include_samples=False,
                                                       **kwargs):
    """

    """
    ## Isolate Appropriate Group
    group_scores = reduced_vocabulary_classification_scores.loc[reduced_vocabulary_classification_scores["group"]==group]
    ## Ignore Within-Period Tests
    if evaluate_within:
        group_scores = group_scores.loc[group_scores["time_period_train"] == group_scores["time_period_apply"]]
        time_period_combinations = group_scores[["vocabulary_reference_train","vocabulary_reference_apply"]].drop_duplicates().values
    else:
        group_scores = group_scores.loc[group_scores["time_period_train"] != group_scores["time_period_apply"]]
        time_period_combinations = group_scores[["time_period_train","time_period_apply"]].drop_duplicates().values
    ## Grid Arangement
    tp_train, tp_apply = sorted(set(time_period_combinations[:,0])), sorted(set(time_period_combinations[:,1]))
    tp_train_r, tp_apply_r = dict(zip(tp_train, range(len(tp_train)))), dict(zip(tp_apply, range(len(tp_apply))))
    ## Selectors
    selectors, _ = get_selectors(analysis_config)
    ## Initialize Figure
    fig, axes = plt.subplots(len(tp_train), len(tp_apply), figsize=(max(5 * len(tp_train), 10), max(3 * len(tp_apply),5.8)), sharex=True, **kwargs)
    completed = set()
    ## Cycle Through Combinations
    for jj, (tpt, tpa) in enumerate(time_period_combinations):
        ## Identify Appropriate Row and Column
        row, col = tp_train_r[tpt], tp_apply_r[tpa]
        if len(tp_train) == 1 and len(tp_apply) == 1:
            pax = axes
        elif len(tp_train) == 1 and len(tp_apply) != 1:
            pax = axes[col]
        elif len(tp_train) != 1 and len(tp_apply) != 1:
            pax = axes[row, col]
        completed.add((row, col))
        ## Axes Labeling
        if row == col:
            pax.set_xlabel("Vocabulary Size\n(Proportion of Total)", fontweight="bold", fontsize=12, labelpad=15)
            pax.set_ylabel(metric.title() if metric != "auc" else "AUC", fontweight="bold", fontsize=12, labelpad=15)
        ## Get Relevant Data
        if evaluate_within:
            pdata = group_scores.loc[(group_scores["vocabulary_reference_train"]==tpt)&(group_scores["vocabulary_reference_apply"]==tpa)]   
        else:
            pdata = group_scores.loc[(group_scores["time_period_train"]==tpt)&(group_scores["time_period_apply"]==tpa)]   
        ## Add Data to Plot
        fig, pax = _plot_reduced_vocabulary_classification_performance(pdata,
                                                                       analysis_config=analysis_config,
                                                                       metric=metric,
                                                                       figure=(fig, pax),
                                                                       include_samples=include_samples)
        ## Add Legend
        if jj == 0:
            pax.legend(loc="lower left",
                       ncol=min(len(selectors), 5),
                       fontsize=10,
                       bbox_to_anchor=(0.0125, 1),
                       handlelength=0.25)
    ## Remove Unused Axes
    for r in range(len(tp_train)):
        for c in range(len(tp_apply)):
            if (r, c) not in completed:
                axes[r, c].axis("off")
    ## Format Figure
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.25)
    return fig

def compute_relative_performance_difference(reduced_vocabulary_classification_scores,
                                            analysis_config,
                                            group="test",
                                            metric="accuracy"):
    """

    """
    ## Isolate Group
    group_results = reduced_vocabulary_classification_scores.loc[reduced_vocabulary_classification_scores["group"]==group]
    ## Duplicate Baselines
    baseline_group_results = group_results.loc[group_results["vocabulary_id"].isin(["cumulative","intersection"])].copy()
    group_results = group_results.loc[~group_results["vocabulary_id"].isin(["cumulative","intersection"])]
    baseline_group_results_duplicated = []
    for percentile in sorted(group_results["vocabulary_selector_percentile"].unique()):
        percentile_baseline_group_results = baseline_group_results.copy()
        percentile_baseline_group_results["vocabulary_selector_percentile"] = percentile
        baseline_group_results_duplicated.append(percentile_baseline_group_results)
    baseline_group_results_duplicated = pd.concat(baseline_group_results_duplicated,axis=0,sort=True)
    group_results = group_results.append(baseline_group_results_duplicated).reset_index(drop=True)
    ## Selectors
    selectors, _ = get_selectors(analysis_config)
    ## Format Results
    pivot_results = pd.pivot_table(group_results,
                                   index=["time_period_train","time_period_apply","vocabulary_reference_train","vocabulary_reference_apply","vocabulary_selector_percentile","fold","sample"],
                                   columns=["vocabulary_selector"],
                                   values=metric,
                                   aggfunc=max)[selectors].reset_index()
    ## Compute Difference from Intersection
    for selector in selectors:
        pivot_results[f"delta_{selector}_intersection"] = pivot_results[selector] - pivot_results["intersection"]
        pivot_results[f"delta_{selector}_cumulative"] = pivot_results[selector] - pivot_results["cumulative"]
    return pivot_results

def _generate_summary_file(agg_performance_df,
                           best_performer_df,
                           rank_df):
    """

    """
    output_str = []
    for (dft, dfall), dfname in zip([agg_performance_df, best_performer_df, rank_df],["Average Performance","Win Rate","Relative Performance Rank"]):
        dft = dft.copy()
        dfall = dfall.copy()
        dft.index.name = ""
        dfall.index.name = ""
        dft.columns.name = ""
        dfall.columns.name = ""
        dfall["time_period_train"] = ""
        dfall["time_period_apply"] = "average"
        dfall = dfall.set_index(["time_period_train","time_period_apply"])
        dfc = pd.concat([dft, dfall],axis=0)
        output_str.append("~"*25 + f"\n{dfname}\n" + "~"*25)
        output_str.append(dfc.to_string())
    output_str = "\n".join(output_str)
    return output_str

def generate_relative_performance_difference_summary(relative_performance_difference,
                                                     analysis_config,
                                                     within_prefer_larger=False):
    """

    """
    ## Get Selectors
    selectors, _ = get_selectors(analysis_config)
    ## Sort 
    relative_performance_difference = relative_performance_difference.sort_values(["time_period_train","time_period_apply","vocabulary_reference_train","vocabulary_reference_apply","sample","fold","vocabulary_selector_percentile"]).reset_index(drop=True).copy()
    ## Separate and Copy Input
    within_tp_difference = relative_performance_difference.loc[relative_performance_difference["time_period_train"]==relative_performance_difference["time_period_apply"]].copy()
    between_tp_difference = relative_performance_difference.loc[relative_performance_difference["time_period_train"]!=relative_performance_difference["time_period_apply"]].copy()    
    ## Melt and Sort the Data
    within_melted = pd.melt(within_tp_difference,
                            id_vars=["vocabulary_reference_train","vocabulary_reference_apply","sample","fold","vocabulary_selector_percentile"],
                            value_vars=selectors,
                            value_name="score")
    within_melted = within_melted.sort_values(["score","vocabulary_selector_percentile"],
                                              ascending=[False,not within_prefer_larger]).reset_index(drop=True)
    between_melted = pd.melt(between_tp_difference,
                             id_vars=["time_period_train","time_period_apply","vocabulary_reference_train","vocabulary_reference_apply","sample","fold","vocabulary_selector_percentile"],
                             value_vars=selectors,
                             value_name="score")
    ## Isolate Optimal Scores (Based on Within-Domain Selection)
    within_optimal = within_melted.groupby(["vocabulary_reference_train","vocabulary_reference_apply","sample","fold","vocabulary_selector"]).apply(
        lambda i: i.iloc[0]["vocabulary_selector_percentile"]
    )
    within_optimal = list(map(tuple, within_optimal.astype(int).reset_index().values))
    opt_performance_within = between_melted.set_index(["vocabulary_reference_train","vocabulary_reference_apply","sample","fold","vocabulary_selector","vocabulary_selector_percentile"]).loc[within_optimal,"score"]
    opt_performance_within = opt_performance_within.reset_index().rename(columns={"vocabulary_reference_train":"time_period_train","vocabulary_reference_apply":"time_period_apply"})
    opt_performance_within = pd.pivot_table(opt_performance_within, index=["time_period_train","time_period_apply","sample","fold"], columns="vocabulary_selector", values="score")[selectors].reset_index()
    ## Isolate Optimal Scores (Based on an Oracle)
    opt_performance_between = between_tp_difference.groupby(["time_period_train","time_period_apply","sample","fold"]).agg({sel:max for sel in selectors})[selectors].reset_index()
    ## Isolate Optimal Scores (on a Vocabulary Selector Level in the Test Data)
    avg_opt_performance = pd.pivot_table(between_melted,
                                         index=["vocabulary_reference_train","vocabulary_reference_apply","vocabulary_selector"],
                                         columns="vocabulary_selector_percentile",
                                         values="score",
                                         aggfunc=np.mean)
    avg_opt_performance = avg_opt_performance.idxmax(axis=1).to_frame("vocabulary_selector_percentile").reset_index().apply(tuple, axis=1)
    avg_opt_performance = between_melted.set_index(["vocabulary_reference_train","vocabulary_reference_apply","vocabulary_selector","vocabulary_selector_percentile"]).loc[avg_opt_performance.tolist(),["score","sample","fold"]]
    avg_opt_performance = avg_opt_performance.reset_index().rename(columns={"vocabulary_reference_train":"time_period_train","vocabulary_reference_apply":"time_period_apply"})
    avg_opt_performance = pd.pivot_table(avg_opt_performance, index=["time_period_train","time_period_apply","sample","fold"], columns="vocabulary_selector", values="score")[selectors].reset_index()
    ## Best Performer
    get_top_perf = lambda df: pd.concat([df[["sample","fold","time_period_train","time_period_apply"]],
                                         df[selectors].apply(lambda row: np.array([j == max(row) for j in row]).nonzero()[0], axis=1).map(lambda i: [selectors[j] for j in i])],
                                         axis=1,
                                         sort=False).rename(columns={0:"best_performer"})
    def get_top_perf_df(df, by_time=True):
        """

        """
        df_top = get_top_perf(df)
        df_count = []
        if by_time:
            for group, indices in df_top.groupby(["time_period_train","time_period_apply"]).groups.items():
                group_counts = {sel:0 for sel in selectors}
                for bp in flatten(df_top.loc[indices]["best_performer"]):
                    group_counts[bp] += 1
                group_counts = {g: (c / len(indices)) for g, c in group_counts.items()}
                group_counts.update({"time_period_train":group[0], "time_period_apply":group[1]})
                df_count.append(group_counts)
            df_count = pd.DataFrame(df_count).set_index(["time_period_train","time_period_apply"])
        else:
            group_counts = {sel:0 for sel in selectors}
            for bp in flatten(df_top["best_performer"].values):
                group_counts[bp] += 1
            group_counts = {g: (c / len(df_top)) for g, c in group_counts.items()}
            df_count.append(group_counts)
            df_count = pd.DataFrame(df_count)
        df_count = indicate_ci_max(df_count.applymap(lambda x: "{:.4f}".format(x)), True)
        return df_count
    ## Rank
    get_rank_df = lambda df: pd.concat([df[["sample","fold","time_period_train","time_period_apply"]],
                                        df[selectors].rank(method="min", ascending=False, axis=1).astype(int)],
                                        axis=1,
                                        sort=False)
    opt_avg_rank = get_rank_df(avg_opt_performance)
    opt_within_rank = get_rank_df(opt_performance_within)
    opt_between_rank = get_rank_df(opt_performance_between)
    ## Aggregated Performance (Average over all Vocabularies)
    ci_to_str = lambda x: "{:.3f} ({:.3f},{:.3f})".format(x[1], x[0], x[2])
    def indicate_ci_max(df, prefer_higher):
        if prefer_higher:
            items = df.applymap(lambda i: float(i.split()[0])).idxmax(axis=1).items()
        else:
            items = df.applymap(lambda i: float(i.split()[0])).idxmin(axis=1).items()
        for index, col in items:
            df.loc[index, col] += "*"
        df = df.applymap(lambda i: i + " " if "*" not in i else i)
        return df
    get_agg_performance = lambda df, by_time, pref_higher: indicate_ci_max(df.groupby(["time_period_train","time_period_apply"]).agg({sel:bootstrap_ci for sel in selectors}).applymap(ci_to_str), pref_higher) if by_time else indicate_ci_max(df[selectors].apply(lambda row: bootstrap_ci(row)).apply(ci_to_str).to_frame().T, pref_higher)
    agg_performance_avg = get_agg_performance(avg_opt_performance, True, True); agg_performance_avg_all = get_agg_performance(avg_opt_performance, False, True)
    agg_performance_within_opt = get_agg_performance(opt_performance_within, True, True); agg_performance_within_opt_all = get_agg_performance(opt_performance_within, False, True)
    agg_performance_between_opt = get_agg_performance(opt_performance_between, True, True); agg_performance_between_opt_all = get_agg_performance(opt_performance_between, False, True)
    ## Best Performer Distribution
    avg_best_performer = get_top_perf_df(avg_opt_performance, True); avg_best_performer_all = get_top_perf_df(avg_opt_performance, False)
    opt_within_best_performer = get_top_perf_df(opt_performance_within, True); opt_within_best_performer_all = get_top_perf_df(opt_performance_within, False)
    opt_between_best_performer = get_top_perf_df(opt_performance_between, True); opt_between_best_performer_all = get_top_perf_df(opt_performance_between, False)
    ## Rank Distribution
    avg_rank_dist = get_agg_performance(opt_avg_rank, True, False); avg_rank_dist_all = get_agg_performance(opt_avg_rank, False, False)
    opt_within_rank_dist = get_agg_performance(opt_within_rank, True, False); opt_within_rank_dist_all = get_agg_performance(opt_within_rank, False, False)
    opt_between_rank_dist = get_agg_performance(opt_between_rank, True, False); opt_between_rank_dist_all = get_agg_performance(opt_between_rank, False, False)
    ## Generate Summary Files
    avg_summary = _generate_summary_file([agg_performance_avg, agg_performance_avg_all], [avg_best_performer, avg_best_performer_all], [avg_rank_dist,avg_rank_dist_all])
    opt_within_summary = _generate_summary_file([agg_performance_within_opt,agg_performance_within_opt_all], [opt_within_best_performer, opt_within_best_performer_all], [opt_within_rank_dist,opt_within_rank_dist_all])
    opt_between_summary = _generate_summary_file([agg_performance_between_opt,agg_performance_between_opt_all], [opt_between_best_performer,opt_between_best_performer_all], [opt_between_rank_dist,opt_between_rank_dist_all])
    return avg_summary, opt_within_summary, opt_between_summary

def plot_relative_performance_difference(relative_performance_difference,
                                         analysis_config,
                                         merge_semantics=False,
                                         merge_baselines=False,
                                         ignore=[]):
    """

    """
    ## Ignore Same Period and Copy
    relative_performance_difference = relative_performance_difference.loc[relative_performance_difference["time_period_train"]!=relative_performance_difference["time_period_apply"]].copy()
    ## Get Selectors
    selectors, _ = get_selectors(analysis_config)
    selectors = list(filter(lambda s: s not in ignore, selectors))
    ## Merge Selectors
    selector2group = {}
    selector_order = []
    for selector in selectors:
        if merge_semantics and (selector.startswith("weighted") or selector == "overlap"):
            selector2group[selector] = "semantics"
            if "semantics" not in selector_order:
                selector_order.append("semantics")
        elif merge_baselines and (selector == "intersection" or selector == "cumulative"):
            selector2group[selector] = "baseline"
            if "baseline" not in selector_order:
                selector_order.append("baseline")
        else:
            selector2group[selector] = selector
            selector_order.append(selector)
    group2selector = {}
    for x, y in selector2group.items():
        if y not in group2selector:
            group2selector[y] = []
        group2selector[y].append(x)
    for group, sels in group2selector.items():
        if group not in relative_performance_difference.columns:
            relative_performance_difference[group] = relative_performance_difference[sels].max(axis=1)
    ## Compute Best Performer
    get_top_perf = lambda df: pd.concat([df[["sample","fold","time_period_train","time_period_apply","vocabulary_selector_percentile"]],
                                         df[selector_order].apply(lambda row: np.array([j == max(row) for j in row]).nonzero()[0], axis=1).map(lambda i: [selector_order[j] for j in i])],
                                         axis=1,
                                         sort=False).rename(columns={0:"best_performer"})
    def get_top_perf_df(df):
        """

        """
        df_top = get_top_perf(df)
        df_count = []
        for group, indices in df_top.groupby(["time_period_train","time_period_apply","vocabulary_selector_percentile"]).groups.items():
            group_counts = {sel:0 for sel in selector_order}
            ntot = 0
            for bp in flatten(df_top.loc[indices]["best_performer"]):
                group_counts[bp] += 1
                ntot += 1
            group_counts = {g: (c / ntot) for g, c in group_counts.items()}
            group_counts.update({"time_period_train":group[0],"time_period_apply":group[1],"vocabulary_selector_percentile":group[2]})
            df_count.append(group_counts)
        df_count = pd.DataFrame(df_count).set_index(["time_period_train","time_period_apply","vocabulary_selector_percentile"])
        return df_count
    ## Aggregate Counts
    agg = get_top_perf_df(relative_performance_difference)
    tp_train_vals = agg.index.levels[0]
    tp_apply_vals = agg.index.levels[1]
    ## Generate Figure
    fig, axes = plt.subplots(len(tp_train_vals), len(tp_apply_vals), figsize=(max(5 * len(tp_train_vals), 10), max(3 * len(tp_apply_vals),5.8)), sharex=False, sharey=True)
    for tpt, tp_train in enumerate(tp_train_vals):
        for tpa, tp_apply in enumerate(tp_apply_vals):
            if len(tp_train_vals) == 1 and len(tp_apply_vals) == 1:
                pax = axes
            else:
                pax = axes[tpt, tpa]
            if tpa < tpt:
                pax.axis("off")
                continue
            agg_fmt = agg.loc[tp_train, tp_apply][selector_order]
            lower = np.zeros(agg_fmt.shape[0])
            for s, sel in enumerate(selector_order):
                pax.bar(np.arange(agg_fmt.shape[0]),
                        bottom=lower,
                        height=agg_fmt[sel].values,
                        alpha=0.5,
                        label=sel)
                lower += agg_fmt[sel].values
            for p in pax.patches:
                width, height = p.get_width(), p.get_height()
                if height == 0:
                    continue
                x, y = p.get_xy() 
                pax.text(x+width/2, 
                         y+height/2, 
                         '{:.2f}'.format(height), 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         fontsize=5)
            pax.set_ylim(0, 1)
            pax.set_xlim(-.5, agg_fmt.shape[1])
            pax.set_xticks(np.arange(agg_fmt.shape[0]))
            pax.set_xticklabels(agg_fmt.index.tolist())
            pax.spines["top"].set_visible(False)
            pax.spines["right"].set_visible(False)
            if tpt == 0 and tpa == 0:
                pax.legend(loc="lower left", ncol=min(5, len(selector_order)), bbox_to_anchor=(0.0125, 1), handlelength=1)
            ## Axes Labeling
            if tpt == tpa:
                pax.set_xlabel("Vocabulary Size\n(Proportion of Total)", fontweight="bold", fontsize=12)
                pax.set_ylabel("Win Rate", fontweight="bold", fontsize=12, labelpad=15)
            else:
                pax.set_xlabel("")
                pax.set_ylabel("")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15, wspace=0.05)
    return fig

def plot_performance_summary(reduced_vocabulary_classification_scores,
                             tp_train,
                             tp_apply,
                             analysis_config,
                             evaluate_within=False,
                             within_prefer_larger=True,
                             metric="accuracy"):
    """

    """
    ## Metadata
    selectors, colors = get_selectors(analysis_config)
    hatchs = ["/","/"] + [None for sel in selectors[2:]]
    ## Isolate Test Data
    test = reduced_vocabulary_classification_scores.loc[(reduced_vocabulary_classification_scores["group"] == "test")]
    ## Isolate Time Period Data
    tp_param = test.loc[(test["time_period_train"]==tp_train)&(test["time_period_apply"]==tp_train)&(test["vocabulary_reference_apply"]==tp_apply)]
    if evaluate_within:
        tp_test = test.loc[(test["time_period_train"]==tp_train)&(test["time_period_apply"]==tp_train)&(test["vocabulary_reference_apply"]==tp_apply)]
    else:
        tp_test = test.loc[(test["time_period_train"]==tp_train)&(test["time_period_apply"]==tp_apply)]
    ## Plot
    fig, axes = plt.subplots(1, 3, figsize=(15,5.8), sharey=False)
    bounds = [[np.inf,-np.inf] for _ in range(3)]
    for s, selector in tqdm(enumerate(selectors), total=len(selectors), file=sys.stdout, desc="[Selector]"):
        ## Get Selector Data
        s_tp_test = tp_test.loc[tp_test["vocabulary_selector"]==selector]
        s_tp_param = tp_param.loc[tp_param["vocabulary_selector"]==selector]
        ## Average Performance
        s_tp_test_mean = bootstrap_tuple_to_df(s_tp_test.groupby(["vocabulary_selector_percentile"]).agg({metric:bootstrap_ci})[metric])
        ## "Optimal" Performance (Based on Within-Time Period Estimates)
        s_tp_param = s_tp_param.sort_values([metric,"vocabulary_selector_percentile"],ascending=[False, not within_prefer_larger]).reset_index(drop=True).copy()
        s_tp_param_opt_p = s_tp_param.groupby(["sample","fold"]).apply(lambda i: i.iloc[0]["vocabulary_selector_percentile"])
        s_tp_param_opt_p = list(map(tuple, s_tp_param_opt_p.map(int).reset_index().values))
        s_tp_param_opt_p = s_tp_test.set_index(["sample","fold","vocabulary_selector_percentile"]).loc[s_tp_param_opt_p,metric]
        s_tp_param_opt_p_mean = bootstrap_ci(s_tp_param_opt_p)
        ## Oracle Performance
        s_tp_test_oracle = s_tp_test.groupby(["sample","fold"])[metric].max().to_frame(metric)
        s_tp_test_oracle = bootstrap_ci(s_tp_test_oracle[metric])
        ## Plot Average Performance
        axes[0].errorbar(s_tp_test_mean.index + 0.4 * s,
                         s_tp_test_mean["median"],
                         yerr=np.array([(s_tp_test_mean["median"]-s_tp_test_mean["lower"]).values,
                                        (s_tp_test_mean["upper"]-s_tp_test_mean["median"]).values]), 
                         label=selector, 
                         color=colors[selector],
                         linewidth=2,
                         capsize=2,
                         alpha=0.8)
        if selector in ["cumulative","intersection"]:
            axes[0].axhline(s_tp_test_mean["median"].values[0], color=colors[selector], alpha=0.4, linewidth=1)
        ## Plot Optimized Performance
        axes[1].bar(s, 
                    s_tp_param_opt_p_mean[1],
                    yerr=[[s_tp_param_opt_p_mean[1]-s_tp_param_opt_p_mean[0]], [s_tp_param_opt_p_mean[2]-s_tp_param_opt_p_mean[1]]],
                    color=colors[selector],
                    label=selector,
                    hatch=hatchs[s],
                    capsize=2,
                    alpha=0.5) 
        ## Plot Oracle Performance
        axes[2].bar(s, 
                    s_tp_test_oracle[1],
                    yerr=[[s_tp_test_oracle[1]-s_tp_test_oracle[0]], [s_tp_test_oracle[2]-s_tp_test_oracle[1]]],
                    color=colors[selector],
                    label=selector,
                    hatch=hatchs[s],
                    capsize=2,
                    alpha=0.5)
        for v, val in enumerate([s_tp_param_opt_p_mean, s_tp_test_oracle]):
            axes[v + 1].text(s, val[2] + 0.002, "{:.3f}\n({:.3f},{:.3f})".format(val[1], val[0], val[2]), fontsize=8, ha="center", va="bottom", rotation=90)
        if selector in ["frequency","coefficient"]:
            for i in [1,2]:
                axes[i].axvline(s + 0.5, color="black", linestyle="--", linewidth=0.75, alpha=0.8)
        ## Bounds
        bounds[0][0] = min(bounds[0][0], s_tp_test_mean["lower"].min())
        bounds[0][1] = max(bounds[0][1], s_tp_test_mean["upper"].max())
        bounds[1][0] = min(bounds[1][0], s_tp_param_opt_p_mean[0])
        bounds[1][1] = max(bounds[1][1], s_tp_param_opt_p_mean[1])
        bounds[2][0] = min(bounds[2][0], s_tp_test_oracle[0])
        bounds[2][1] = max(bounds[2][1], s_tp_test_oracle[1])
    ## Final Formatting
    for b, (bl, bu) in enumerate(bounds):
        axes[b].set_ylim(bl - 0.04, bu + 0.04)
    axes[1].legend(loc="lower center", ncol=6, bbox_to_anchor=(0.5,-0.2), fontsize=8)
    axes[0].set_title(f"Average {metric.title()} Score", fontweight="bold")
    axes[1].set_title(f"Optimal {metric.title()} Score", fontweight="bold")
    axes[2].set_title(f"Oracle {metric.title()} Score", fontweight="bold")
    axes[0].set_ylabel(f"{metric.title()} Score", fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_xlabel("Percentile", fontweight="bold")
    for i in [1,2]:
        axes[i].set_xticks([])
        axes[i].set_xlabel("Selector", fontweight="bold")
        axes[i].set_xlim(-0.5, len(selectors)-0.5)
    fig.suptitle(f"Train: {tp_train} | Apply: {tp_apply}", fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    return fig

def main():
    """
    
    """
    ## Parse Command Line
    LOGGER.info("[Parsing Command Line]")
    args = parse_command_line()
    ## Load Configuration
    LOGGER.info("[Loading Configuration File]")
    analysis_config = load_json(args.config)
    ## Filepaths
    experiment_directory = "{}/{}/".format(analysis_config["base_output_dir"],analysis_config["base_output_dir_ext"]).replace("//","/")
    analysis_directory = "{}/analysis/{}/".format(experiment_directory, analysis_config.get("analysis_id"))
    ## Intialize Analysis Directory and Cache Configuration
    if os.path.exists(analysis_directory) and args.rm_existing:
        LOGGER.info("[Remove Existing Analysis Directory: {}]".format(analysis_directory))
        _ = os.system("rm -rf {}".format(analysis_directory))
    if not os.path.exists(analysis_directory):
        _ = os.makedirs(analysis_directory)
    ## Cache Configuration
    LOGGER.info("[Caching/Checking Configuration]")
    _ = cache_configuration(analysis_config, analysis_directory)
    ## Identify Time Period Configurations
    time_period_configs = list(map(load_json, sorted(glob(f"{experiment_directory}/*_*.json"))))
    time_period_configs = sorted(time_period_configs, key = lambda x: (0 if x["experiment_name"].startswith("cumulative") else 1, x["datasets"][0]["date_boundaries"]["max_date"]))
    ## Vectorize Data
    if args.vectorize is not None:
        ## Symbolic Check
        if args.vectorize == "symbolic":
            _ = initialize_symbolic_vector_directory(analysis_directory=analysis_directory,
                                                     symbolic_directory=args.vectorize_symbolic_path,
                                                     analysis_config=analysis_config)
            LOGGER.info("[Initialized Symbolic Vector Directory. Exiting]")
            return None
        ## Sample IDs
        sample_id_list = get_samples(time_period_configs,
                                     experiment_directory)
        ## Scheduling
        if args.vectorize == "parallel" and args.vectorize_id is None:
            ## Run Scheduler
            _ = schedule_vectorization_jobs(sample_id_list,
                                            analysis_directory=analysis_directory,
                                            jobs=args.jobs,
                                            memory_per_job=args.grid_memory_per_job,
                                            max_array_size=args.grid_max_array_size,
                                            skip_existing=args.vectorize_skip_existing)
            LOGGER.info("[Scheduling of Vectorization Complete. Exiting.]")
            return None
        ## Identify Vectorization ID (Parallel Processing)
        if args.vectorize == "parallel" and args.vectorize_id is not None:
            sample_id_list = [sample_id_list[args.vectorize_id - 1]]
        # Initialize Multi-Processor
        with Pool(args.jobs) as mp:
            ## Iterate over Samples
            for tp_ind, sample_ind, sample_split_filename, sample_embedding_directory in sample_id_list:
                ## Vectorize the Sample
                _ = vectorize_time_period_data_sample(sample_ind=sample_ind,
                                                      sample_split_filename=sample_split_filename,
                                                      sample_embedding_directory=sample_embedding_directory,
                                                      analysis_directory=analysis_directory,
                                                      time_period_config=time_period_configs[tp_ind],
                                                      min_n_posts_train=analysis_config["min_train_posts"],
                                                      min_n_posts_test=analysis_config["min_test_posts"],
                                                      n_posts_train=analysis_config["train_posts"],
                                                      n_posts_test=analysis_config["test_posts"],
                                                      random_state=analysis_config["random_seed"],
                                                      mp=mp,
                                                      verbose=True,
                                                      skip_existing=args.vectorize_skip_existing)
        ## Done
        LOGGER.info("[Vectorization Complete. Exiting.]")
        return None
    ## Classification Procedure
    if args.classify is not None:
        ## IDs
        classify_id_list = list(range(time_period_configs[0]["sample_protocol"]["n_sample"]))
        ## Scheduling
        if args.classify == "parallel" and args.classify_id is None:
            ## Schedule
            _ = schedule_classification_jobs(len(classify_id_list),
                                             analysis_directory=analysis_directory,
                                             jobs=args.jobs,
                                             memory_per_job=args.grid_memory_per_job,
                                             max_array_size=args.grid_max_array_size,
                                             skip_existing=args.classify_skip_existing,
                                             enforce_frequency_filter=args.classify_enforce_frequency_filter,
                                             enforce_classwise_frequency_filter=args.classify_enforce_classwise_frequency_filter,
                                             enforce_min_shift_frequency_filter=args.classify_enforce_min_shift_frequency_filter,
                                             compute_cumulative=args.classify_compute_cumulative,
                                             compute_discrete=args.classify_compute_discrete)
            LOGGER.info("[Scheduling of Classification Procedures Complete. Exiting.]")
            return None
        ## Identify Vectorization ID (Parallel Processing)
        if args.classify_id is not None:
            classify_id_list = [classify_id_list[args.classify_id - 1]]
        ## Execute Classification
        LOGGER.info("[Beginning Classification Procedure]")
        for cid, classify_id in enumerate(classify_id_list):
            LOGGER.info("[Beginning Sample {}/{}]".format(cid+1, len(classify_id_list)))
            _ = run_classification(n=classify_id,
                                   analysis_directory=analysis_directory,
                                   analysis_config=analysis_config,
                                   time_period_configs=time_period_configs,
                                   skip_existing=args.classify_skip_existing,
                                   enforce_frequency_filter=args.classify_enforce_frequency_filter,
                                   enforce_classwise_frequency_filter=args.classify_enforce_classwise_frequency_filter,
                                   enforce_min_shift_frequency_filter=args.classify_enforce_min_shift_frequency_filter,
                                   compute_cumulative=args.classify_compute_cumulative,
                                   compute_discrete=args.classify_compute_discrete,
                                   njobs=args.jobs)
        ## Exit
        LOGGER.info("[Classification Complete. Exiting]")
        return None
    ## Analysis of Classification Results
    if args.analyze:
        LOGGER.info("[Beginning Result Analysis]")
        ## Data Loading
        LOGGER.info("[Loading Semantic Stability - Coefficient Relationships]")
        stability_coefficient_relationship = load_stability_coefficient_relationship(time_period_configs[0]["sample_protocol"]["n_sample"],
                                                                                     analysis_directory)
        LOGGER.info("[Loading Reduced Vocabulary Classification Results]")
        reduced_vocabulary_classification_scores = load_classification_scores(time_period_configs[0]["sample_protocol"]["n_sample"],
                                                                              analysis_directory)
        ## Visualiztion (Coefficient - Stability)
        LOGGER.info("[Visualizing Semantic Stability - Coefficient Relationships]")
        fig = plot_stability_coefficient_relationship(stability_coefficient_relationship.groupby(["feature","time_period_train","time_period_apply"]).mean().reset_index(), 25)
        fig.savefig(f"{analysis_directory}/coefficient_stability_summary.png", dpi=300)
        plt.close(fig)
        ## Visualization (Classification Results)
        LOGGER.info("[Visualizing Reduced Vocabulary Classification Results]")
        relative_performance_dfs = {}
        for metric_type in tqdm(analysis_config["metric_types"], desc="[Metric]", file=sys.stdout, leave=True, position=0):
            for w, domain_lbl in enumerate(["between-domain","within-domain"]):
                ## Generate Figure
                fig = plot_reduced_vocabulary_classification_performance(reduced_vocabulary_classification_scores,
                                                                         analysis_config=analysis_config,
                                                                         metric=metric_type,
                                                                         group="test",
                                                                         evaluate_within=w,
                                                                         include_samples=False,
                                                                         sharey=True)
                fig.savefig(f"{analysis_directory}/scores.{domain_lbl}.{metric_type}.png", dpi=300)
                plt.close(fig)
            ## Compute Relative Performance Difference
            metric_relative_performance_difference = compute_relative_performance_difference(reduced_vocabulary_classification_scores,
                                                                                             analysis_config=analysis_config,
                                                                                             metric=metric_type)
            ## Store to Generate Summaries
            relative_performance_dfs[metric_type] = metric_relative_performance_difference
            ## Plot Relative Performance Difference
            for merge_semantics in tqdm([False,True], desc="[Relative Performance - Merge Semantics]", file=sys.stdout, position=1, leave=False):
                for merge_baselines in tqdm([False,True], desc="[Relative Performance - Merge Baselines]", file=sys.stdout, position=2, leave=False):
                    fig = plot_relative_performance_difference(metric_relative_performance_difference,
                                                               analysis_config,
                                                               merge_semantics=merge_semantics,
                                                               merge_baselines=merge_baselines,
                                                               ignore=[])
                    fig.savefig(f"{analysis_directory}/differences.{metric_type}.merge_semantics-{merge_semantics}.merge_baselines-{merge_baselines}.png",dpi=300)
        ## Generate Summary Files
        tp_combos = reduced_vocabulary_classification_scores.loc[reduced_vocabulary_classification_scores["time_period_train"]!=reduced_vocabulary_classification_scores["time_period_apply"]][["time_period_train","time_period_apply"]].drop_duplicates().values
        for metric_type in analysis_config["metric_types"]:
            LOGGER.info(f"[Generating Performance Summaries: {metric_type}]")
            ## Build Summaries
            metric_average_summary, metric_optimal_within_summary, metric_optimal_between_summary = generate_relative_performance_difference_summary(relative_performance_dfs[metric_type],
                                                                                                                                                     analysis_config,
                                                                                                                                                     within_prefer_larger=True)
            ## Cache Summaries
            for stype, s in zip(["average","optimized","oracle"],[metric_average_summary, metric_optimal_within_summary, metric_optimal_between_summary]):
                with open(f"{analysis_directory}/summary.{metric_type}.{stype}.txt", "w") as the_file:
                    _ = the_file.write(s)
            ## Plot Summaries
            for jj, (tp_train, tp_apply) in enumerate(tp_combos):
                LOGGER.info("[Plotting Summary (Time Period {}/{})]".format(jj+1, len(tp_combos)))
                for w, domain_lbl in enumerate(["between-domain","within-domain"]):
                    fig = plot_performance_summary(reduced_vocabulary_classification_scores,
                                                   tp_train=tp_train,
                                                   tp_apply=tp_apply,
                                                   metric=metric_type,
                                                   analysis_config=analysis_config,
                                                   evaluate_within=w,
                                                   within_prefer_larger=True)
                    fig.savefig(f"{analysis_directory}/summary-figure.{domain_lbl}.{metric_type}.{tp_train}-{tp_apply}.png", dpi=150)
                    plt.close(fig)
    ## Entire Script Complete
    LOGGER.info("[Script Complete]")

#####################
### Execute
#####################

if __name__ == "__main__":
    _ = main()