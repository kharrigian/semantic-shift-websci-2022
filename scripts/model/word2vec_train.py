
"""
Learn embeddings based on a preprocessed dataset
"""

################
### Imports
################

## Standard Library
import os
import json
import argparse
import subprocess
from glob import glob
from copy import deepcopy
from textwrap import dedent

## External Libraries
import numpy as np
import pandas as pd

## Project Modules
from semshift.model import train
from semshift.model import embed
from semshift.util.helpers import chunks
from semshift.util.logging import initialize_logger
from semshift.preprocess.preprocess import RawDataLoader
from semshift.model.data_loaders import PostStream, PostStreamPipeline

#################
### Globals
#################

## Create Logger
LOGGER = initialize_logger()

## Identify Root Directory
ROOT_DIR = os.path.abspath(os.path.dirname(__file__) + "/../") + "/"

#################
### Functions
#################

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
                        default=f"{ROOT_DIR}data/resources/embeddings/")
    parser.add_argument("--stream_cache_dir",
                        type=str,
                        default=f"{ROOT_DIR}/")
    parser.add_argument("--rm_existing",
                        action="store_true",
                        default=False,
                        help="If included and output directory already exists, will remove it before moving forward.")
    parser.add_argument("--rerun",
                        action="store_true",
                        default=False,
                        help="If included and sample already exists, will still re-run the training procedure.")
    parser.add_argument("--resample_parallel",
                        action="store_true",
                        default=False,
                        help="If included and using re-sampling, will schedule jobs on the CLSP grid. Otherwise, serial local implementation.")
    parser.add_argument("--resample_parallel_sample",
                        default=None,
                        type=int,
                        help="If performing re-sampling, specifies which sample to isolate.")
    parser.add_argument("--extract_vocabulary",
                        action="store_true",
                        default=False,
                        help="If included, we will learn vocabulary in text stream pre-emptively.")
    parser.add_argument("--phraser_dir",
                        type=str,
                        default=None,
                        help="Optionally, use existing phrasers by passing directory containing saved phrasers.")
    parser.add_argument("--phraser_only",
                        action="store_true",
                        default=False,
                        help="If included, will skip the embedding learning procedure and only learn phrasers.")
    parser.add_argument("--grid_max_array",
                        type=int,
                        default=500,
                        help="Maximum size of a single job array.")
    parser.add_argument("--grid_memory_request_size",
                        type=int,
                        default=16,
                        help="Memory request size.")
    parser.add_argument("--grid_log_dir",
                        type=str,
                        default="./logs/embed/")
    ## Parse Arguments
    args = parser.parse_args()
    return args

def load_config(config_file):
    """
    
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError("Could not find configuration file.")
    with open(config_file,"r") as the_file:
        config = json.load(the_file)
    ## Check Mode
    if config.get("mode") not in [0,1]:
        raise ValueError("Mode must be 0 (sentence-level) or 1 (post-level")
    ## Assign Run Properties To Each Dataset
    for dataset in config.get("datasets",[]):
        dataset["mode"] = config.get("mode",0)
        dataset["vocab_kwargs"] = config.get("vocab_kwargs", {})
        dataset["jobs"] = config.get("jobs",1)
        dataset["random_seed"] = config.get("random_seed", 42)
    ## Assign Properties to Model
    config["model_kwargs"]["jobs"] = config.get("jobs",1)
    config["model_kwargs"]["random_seed"] = config.get("random_seed",42)
    return config

def _resample_files(filename2label,
                    test_size=0.2):
    """
    Resample files while trying to maintain existing balance
    """
    ## Get Filenames by Class
    pos_files = [x for x, y in filename2label.items() if y != "control"]
    neg_files = [x for x, y in filename2label.items() if y == "control"]
    ## Shuffle
    _ = np.random.shuffle(pos_files)
    _ = np.random.shuffle(neg_files)
    ## Targets
    n_pos_test = int(test_size * len(pos_files))
    n_neg_test = int(test_size * len(neg_files))
    ## Slice
    test_pos, train_pos = pos_files[:n_pos_test], pos_files[n_pos_test:]
    test_neg, train_neg = neg_files[:n_neg_test], neg_files[n_neg_test:]
    ## Merge
    split = {
        "train":{x:filename2label[x] for x in train_pos + train_neg},
        "test":{x:filename2label[x] for x in test_pos + test_neg}
    }
    return split

def _get_resampled_splits(filename2label,
                          test_size=0.2,
                          n_sample=10,
                          random_state=42):
    """

    """
    ## Set Random State
    _ = np.random.seed(random_state)
    ## Get Splits
    splits = [_resample_files(filename2label, test_size) for _ in range(n_sample)]
    return splits

def _get_annotated_dataset_files(dataset_config):
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
    if dataset_config.get("include_classes",{}).get("control",True):
        label_filter.add("control")
    if dataset_config.get("include_classes",{}).get(dataset_config.get("target_disorder"), True):
        label_filter.add(dataset_config.get("target_disorder"))
    ## Identify Unique Users Matching Desired Condition
    filtered_filenames = {}
    for user_dict in filenames:
        filtered_filenames.update({u:l for u, l in user_dict.items() if l in label_filter})
    return filtered_filenames, None, None

def _get_custom_dataset_files(dataset_config):
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
    ## Initialize Raw Data Processor
    processor_type = dataset_config["target_disorder"].split("custom-")[1]
    if processor_type not in set(["twitter","reddit"]):
        raise ValueError("Could not identify processor for custom dataset. Input form should be custom-<PLATFORM>")
    processor = RawDataLoader(processor_type,
                              random_state=dataset_config.get("random_seed"),
                              lang=dataset_config.get("lang"),
                              run_pipeline=dataset_config.get("preprocessed") is False)
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
    ## Proxy Format Filenames
    filenames = {f:"control" for f in filenames}
    return filenames, processor, processor_kwargs

def _get_annotated_stream(filenames,
                          dataset_config,
                          cache_dir,
                          baselines=None,
                          sample_ind=None,
                          sample_group=None):
    """

    """
    ## Make a Copy of The Configuration
    dataset_config = deepcopy(dataset_config)
    ## Extract Sample Rate
    n_samples = dataset_config.get("post_sampling",{}).get("n_samples")
    randomized = dataset_config.get("post_sampling",{}).get("randomized")
    ## Update Sampling Arguments
    if baselines is not None and baselines == "post":
        dataset_config["vocab_kwargs"]["random_state"] = dataset_config.get("vocab_kwargs",{}).get("random_state",42) + sample_ind
        n_samples = "discrete-2-{}".format({"a":0,"b":1}[sample_group])
        randomized = True
    ## Initialize Stream
    dataset_stream = PostStream(filenames,
                                loader_kwargs=dataset_config.get("vocab_kwargs",{}),
                                min_date=dataset_config.get("date_boundaries",{}).get("min_date"),
                                max_date=dataset_config.get("date_boundaries",{}).get("max_date"),
                                n_samples=n_samples,
                                randomized=randomized,
                                jobs=dataset_config.get("jobs",1),
                                preserve_order=False,
                                verbose=True,
                                metadata=[],
                                mode=dataset_config.get("mode",0),
                                check_filename_size=dataset_config.get("check_filesize",False),
                                cache_data=True,
                                cache_dir=cache_dir)
    ## Add Type of Stream Attribute
    return dataset_stream

def _get_custom_stream(filenames,
                       dataset_config,
                       processor,
                       processor_kwargs,
                       cache_dir,
                       baselines=None,
                       sample_ind=None,
                       sample_group=None):
    """

    """
    ## Make a Copy of the Processor
    processor = deepcopy(processor)
    processor_kwargs = deepcopy(processor_kwargs)
    ## Update Processor and Arguments
    if baselines is not None and baselines == "post":
        processor.random_state = dataset_config.get("vocab_kwargs",{}).get("random_state",42) + sample_ind
        processor_kwargs["sample_rate"] = "discrete-2-{}".format({"a":0,"b":1}[sample_group])
    ## Initialize Stream
    dataset_stream = PostStream(filenames,
                                loader_kwargs=dataset_config.get("vocab_kwargs",{}),
                                processor=processor,
                                processor_kwargs=processor_kwargs,
                                min_date=None,
                                max_date=None,
                                n_samples=None,
                                randomized=True,
                                jobs=dataset_config.get("jobs",1),
                                preserve_order=False,
                                verbose=True,
                                metadata=[],
                                mode=dataset_config.get("mode",0),
                                check_filename_size=dataset_config.get("check_filesize",False),
                                cache_data=True,
                                cache_dir=cache_dir)
    return dataset_stream

def get_filenames(dataset_config):
    """

    """
    if dataset_config["target_disorder"].startswith("custom-"):
        return _get_custom_dataset_files(dataset_config)
    else:
        return _get_annotated_dataset_files(dataset_config)

def get_stream(filenames,
               dataset_config,
               cache_dir,
               baselines=None,
               sample_ind=None,
               sample_group=None,
               **kwargs):
    """
    Create stream of files for a given dataset configuration.

    Args:
        dataset_config (dict)
    
    Returns:
        dataset_stream (PostStream)
    """
    if dataset_config["target_disorder"].startswith("custom-"):
        dataset_stream = _get_custom_stream(filenames=filenames,
                                            dataset_config=dataset_config,
                                            cache_dir=cache_dir,
                                            baselines=baselines,
                                            sample_ind=sample_ind,
                                            sample_group=sample_group,
                                            **kwargs)
    else:
        dataset_stream = _get_annotated_stream(filenames=filenames,
                                               dataset_config=dataset_config,
                                               cache_dir=cache_dir,
                                               baselines=baselines,
                                               sample_ind=sample_ind,
                                               sample_group=sample_group)
    return dataset_stream

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
    #$ -N embedding
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

def make_sh_file(lower_bound,
                 upper_bound,
                 output_dir,
                 experiment_output_dir,
                 stream_cache_dir,
                 phraser_dir,
                 phraser_only,
                 log_dir,
                 memory=8,
                 num_jobs=1):
    """

    """
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
    python scripts/model/word2vec_train.py {}/config.json --resample_parallel --resample_parallel_sample $SGE_TASK_ID --output_dir {} --stream_cache_dir {} {} {}
    """.format(header,
               init,
               experiment_output_dir,
               output_dir,
               stream_cache_dir,
               f"--phraser_dir {phraser_dir}" if phraser_dir is not None else "",
               "--phraser_only" if phraser_only else "")
    script = dedent(script)
    script = script.replace("//","/")
    ## Write Script File
    script_file = f"{experiment_output_dir}/_jobs/preprocess_{lower_bound}_{upper_bound}.sh".replace("//","/")
    with open(script_file,"w") as the_file:
        the_file.write(script)
    return script_file

def schedule_jobs(config,
                  output_dir,
                  stream_cache_dir,
                  phraser_dir,
                  phraser_only,
                  grid_log_dir,
                  grid_max_array=500,
                  grid_memory_request_size=8,
                  jobs=1):
    """
    
    """
    ## Array Bounds
    n_sample = config["sample_protocol"]["n_sample"]
    n_sample_bounds = [(i[0], i[-1]) for i in chunks(list(range(1, n_sample+1)), grid_max_array)]
    ## Initialize Directories
    experiment_output_dir = "{}/{}/".format(output_dir, config.get("experiment_name")).replace("//","/")
    if not os.path.exists(f"{experiment_output_dir}/_jobs/"):
        _ = os.makedirs(f"{experiment_output_dir}/_jobs/")
    if not os.path.exists(grid_log_dir):
        _ = os.makedirs(grid_log_dir)
    ## Make Job Files
    job_files = []
    for lower, upper in n_sample_bounds:
        job_files.append(make_sh_file(lower_bound=lower,
                                      upper_bound=upper,
                                      output_dir=output_dir,
                                      experiment_output_dir=experiment_output_dir,
                                      stream_cache_dir=stream_cache_dir,
                                      phraser_dir=phraser_dir,
                                      phraser_only=phraser_only,
                                      log_dir=grid_log_dir,
                                      memory=grid_memory_request_size,
                                      num_jobs=jobs))
    ## Schedule
    LOGGER.info(f"[Scheduling {len(job_files)} Job Arrays for {n_sample} Samples]")
    for job_file in job_files:
        command = f"qsub {job_file}"
        job_id = subprocess.check_output(command, shell=True)
        LOGGER.info(job_id)

def _split_files_for_baseline(filename_dict,
                              random_state=None):
    """
    
    """
    ## Separate Files
    neg_files = sorted([f for f, y in filename_dict.items() if y == "control"])
    pos_files = sorted([f for f, y in filename_dict.items() if y != "control"])
    ## Initialize Random Seed
    sampler = np.random.RandomState(random_state)
    ## Suffle The Files
    _ = sampler.shuffle(neg_files)
    _ = sampler.shuffle(pos_files)
    ## Split into two groups
    neg_a, neg_b = neg_files[:len(neg_files)//2], neg_files[len(neg_files)//2:] if len(neg_files) > 0 else ([], [])
    pos_a, pos_b = pos_files[:len(pos_files)//2], pos_files[len(pos_files)//2:] if len(pos_files) > 0 else ([], [])
    ## Merge
    groups = {
        "a":{filename:filename_dict[filename] for filename in neg_a + pos_a},
        "b":{filename:filename_dict[filename] for filename in neg_b + pos_b}
    }
    return groups

def main():
    """
    
    """
    ## Parse Command Line
    LOGGER.info("[Parsing Command Line]")
    args = parse_arguments()
    ## Load Configuration
    LOGGER.info("[Loading Configuration]")
    config = load_config(config_file=args.config_file)
    ## Initialize Output Directory
    experiment_output_dir = "{}/{}".format(args.output_dir, config.get("experiment_name")).replace("//","/")
    if os.path.exists(experiment_output_dir) and not args.rm_existing and not (args.resample_parallel and args.resample_parallel_sample is not None):
        LOGGER.warning(f"[Warning: Output directory '{experiment_output_dir}' already exists. Consider renaming your experiment to avoid overwriting.]")
    elif os.path.exists(experiment_output_dir) and args.rm_existing:
        LOGGER.info(f"[Removing previous output directory and re-initializing at: {experiment_output_dir}]")
        _ = os.system("rm -rf {}".format(experiment_output_dir))
        _ = os.makedirs(experiment_output_dir)
    elif not os.path.exists(experiment_output_dir):
        LOGGER.info(f"[Initializing output directory at: {experiment_output_dir}]")
        _ = os.makedirs(experiment_output_dir)
    ## Cache Config
    if args.resample_parallel_sample is None or (args.resample_parallel_sample is not None and not os.path.exists(f"{experiment_output_dir}/config.json")):
        LOGGER.info("[Caching Config]")
        with open(f"{experiment_output_dir}/config.json","w") as the_file:
            json.dump(config, the_file, indent=4, sort_keys=False)
    ## Important Training Arguments
    resample = config.get("sample_protocol",{}).get("resample",False)
    baselines = config.get("sample_protocol",{}).get("baselines",None)
    ## Check Parameterization (Baseline Testing Configured for File-Based Sampling)
    if baselines is not None and baselines not in ["file","post"]:
        raise ValueError("If running baseline test, configuration parameter should be one of 'file' or 'post'.")
    ## Initialization (Scheduling) for Parallel Training on Grid
    if resample and args.resample_parallel and args.resample_parallel_sample is None:
        LOGGER.info("[Preparing Scheduler]")
        ## Run Scheduling
        _ = schedule_jobs(config=config,
                          output_dir=args.output_dir,
                          stream_cache_dir=args.stream_cache_dir,
                          phraser_dir=args.phraser_dir,
                          phraser_only=args.phraser_only,
                          grid_max_array=args.grid_max_array,
                          grid_log_dir=args.grid_log_dir,
                          grid_memory_request_size=args.grid_memory_request_size,
                          jobs=config["jobs"])
        ## Early Exit
        LOGGER.info("[Script Complete. All Jobs Scheduled.]")
        return None
    ## Get Sample Indice(s)
    if resample and not args.resample_parallel:
        sample_inds = list(range(config["sample_protocol"]["n_sample"]))
    elif resample and args.resample_parallel and args.resample_parallel_sample is not None:
        sample_inds = [args.resample_parallel_sample - 1] ## Subtract one since CLSP scheduler uses a 1-index
    elif resample and args.resample_parallel and args.resample_parallel_sample is None:
        raise Exception("This should not happen based on the logic above.")
    elif not resample:
        sample_inds = [0]
    ## Get Stream Inputs
    LOGGER.info("[Initializing Stream Inputs]")
    stream_inputs = {si:[] for si in sample_inds}
    for dataset_config in config.get("datasets",[]):
        ## Identify Dataset
        dataset = dataset_config.get("dataset-id")
        ## Initialize Sample Directory
        dataset_sample_dir = f"{experiment_output_dir}/splits/{dataset}/"
        try:
            if not os.path.exists(dataset_sample_dir):
                _ = os.makedirs(dataset_sample_dir)
        except FileExistsError as e:
            pass ## Parallel execution can create race conditions
        ## Extract Filenames and Data Parameters
        dataset_filename2label, processor, processor_kwargs = get_filenames(dataset_config)
        ## Manage Resampling
        if not resample:
            ## Baseline Protocol (Randomly Split Files into Groups)
            if baselines is None:
                dataset_filename2label = {"all":dataset_filename2label}
            if baselines == "file":
                dataset_filename2label = _split_files_for_baseline(dataset_filename2label, random_state=config.get("random_seed"))
            elif baselines == "post":
                dataset_filename2label = {"a":dataset_filename2label, "b":dataset_filename2label}
            else:
                raise ValueError("Unrecognized baseline parameter")
            ## Caching of Training Files
            for group, group_labels in dataset_filename2label.items():
                group_filename = f"{dataset_sample_dir}/0.json" if group == "all" else f"{dataset_sample_dir}/0.{group}.json"
                with open(group_filename,"w") as the_file:
                    json.dump(group_labels, the_file)
            ## Cache Params
            stream_inputs[0].append((
                dataset_filename2label,
                dataset_config,
                processor,
                processor_kwargs)
            )
        else:
            ## Case 1: Resample Files (Note That Posts Within Each File are Fixed)
            if config["sample_protocol"]["resample_files"]:
                ## Splits for Each Resampling
                dataset_sample_splits = _get_resampled_splits(dataset_filename2label,
                                                              test_size=config["sample_protocol"]["test_size"],
                                                              n_sample=config["sample_protocol"]["n_sample"],
                                                              random_state=config["random_seed"])
                ## Cache Relevant Splits
                for s, si in enumerate(sample_inds):
                    ## Baseline Protocol (Randomly Split Files into Groups)
                    if baselines is None:
                        si_splits = {
                            "all":dataset_sample_splits[si]
                        }
                    elif baselines == "post":
                        si_splits = {
                            "a":dataset_sample_splits[si],
                            "b":dataset_sample_splits[si]
                        }
                    elif baselines == "file":
                        ## Separate Train/Test
                        si_train_splits = _split_files_for_baseline(dataset_sample_splits[si]["train"], random_state=si)
                        si_test_splits = _split_files_for_baseline(dataset_sample_splits[si]["test"], random_state=si)
                        ## Merge
                        si_splits = {
                            "a":{"train":si_train_splits["a"], "test":si_test_splits["a"]},
                            "b":{"train":si_train_splits["b"], "test":si_test_splits["b"]}
                        }
                    else:
                        raise ValueError("Baselines Parameter Not Recognized.")
                    ## Cache Training/Test Files
                    for group, group_labels in si_splits.items():
                        group_filename = f"{dataset_sample_dir}/{si}.json" if group == "all" else f"{dataset_sample_dir}/{si}.{group}.json"
                        with open(group_filename, "w") as the_file:
                            json.dump(group_labels, the_file)
                    ## Cache Parameters
                    stream_inputs[si].append((
                        {"all":si_splits["all"]["train"]} if baselines is None else {"a":si_splits["a"]["train"], "b":si_splits["b"]["train"]},
                        dataset_config,
                        processor,
                        processor_kwargs)
                    )
            ## Case 2: Resample Posts, But Keep All Files Fixed
            else:
                ## Check Parameters
                if baselines is not None and baselines == "post":
                    raise ValueError("If resampling posts, cannot specify baselines='post'")
                ## Sample Size (A function of Original Sample Rate and Resampling Sample Rate)
                resample_sample_rate = 1 - config["sample_protocol"]["test_size"]
                resample_sample_rate = resample_sample_rate if dataset_config["post_sampling"]["n_samples"] is None else resample_sample_rate * dataset_config["post_sampling"]["n_samples"]
                ## Update Sample Rate
                if processor_kwargs is not None:
                    processor_kwargs["sample_rate"] = resample_sample_rate
                ## Iterate Over Samples
                for s, si in enumerate(sample_inds):
                    ## Baseline Protocol
                    if baselines is None:
                        si_splits = {"all":dataset_filename2label}
                    elif baselines == "file":
                        si_splits = _split_files_for_baseline(dataset_filename2label, random_state=si)
                    else:
                        raise ValueError("Baselines parameter not supported for this sampling configuration.")
                    ## Cache The Splits
                    for group, group_splits in si_splits.items():
                        group_filename = f"{dataset_sample_dir}/{si}.json" if group == "all" else f"{dataset_sample_dir}/{si}.{group}.json"
                        with open(group_filename, "w") as the_file:
                            json.dump({"train":group_splits, "test":{}}, the_file)
                    ## Copy the Processor and Update Sample Params
                    if processor is not None:
                        si_processor = deepcopy(processor)
                        si_processor.random_state = si
                    else:
                        si_processor = None
                    ## Copy the Dataset Config and Update Sample Params
                    si_dataset_config = deepcopy(dataset_config)
                    si_dataset_config["vocab_kwargs"]["random_state"] = si
                    si_dataset_config["post_sampling"]["randomized"] = True
                    si_dataset_config["post_sampling"]["n_samples"] = resample_sample_rate if si_processor is None else None
                    ## Cache Params
                    stream_inputs[si].append((
                        si_splits,
                        si_dataset_config,
                        si_processor,
                        processor_kwargs)
                    )
    ## Translate Stream Inputs into Streams
    LOGGER.info("[Initializing Streams]")
    streams = {}
    for sample_ind, sample_dataset_params in stream_inputs.items():
        ## Sample Specific Output Directory
        sample_output_dir = experiment_output_dir if not resample else f"{experiment_output_dir}/models/sample-{sample_ind}"
        ## Check for Completion
        embedding_file = glob(f"{sample_output_dir}/embeddings.*.txt")
        if len(embedding_file) > 0 and not args.rerun:
            LOGGER.info("[Sample Already Completed. Moving On.]")
            continue
        ## Generate Individual Streams
        sample_ind_streams = {}
        for filenames, dataset_config, processor, processor_kwargs in sample_dataset_params:
            for group, group_filenames in filenames.items():
                if group not in sample_ind_streams:
                    sample_ind_streams[group] = []
                group_stream = get_stream(filenames=sorted(group_filenames.keys()),
                                          dataset_config=dataset_config,
                                          processor=processor,
                                          processor_kwargs=processor_kwargs,
                                          cache_dir=args.stream_cache_dir,
                                          baselines=baselines,
                                          sample_ind=sample_ind,
                                          sample_group=group)
                sample_ind_streams[group].append(group_stream)
        ## Merge and Cache
        streams[sample_ind] = {group:PostStreamPipeline(*sample_ind_stream) for group, sample_ind_stream in sample_ind_streams.items()}
    ## Phrasers
    if args.phraser_dir is not None:
        if not os.path.exists(args.phraser_dir):
            raise FileNotFoundError("Phraser directory passed as input does not exist.")
        phrasers = embed.Word2Vec.load_phrasers(f"{args.phraser_dir}/")
    ## Cycle Through Streams
    if len(streams) > 0:
        LOGGER.info("[Beginning Training Procedure]")
    for s, (sample_ind, sample_streams) in enumerate(streams.items()):
        LOGGER.info("[Beginning Training for Stream Sample {}/{}: Sample {}]".format(s+1, len(streams), sample_ind))
        ## Iterate Through Groups (Either One or Two depending on Whether Running Baselines)
        for group, group_sample_stream in sample_streams.items():
            ## Sample Specific Output Directory
            sample_output_dir = experiment_output_dir if not resample else f"{experiment_output_dir}/models/sample-{sample_ind}"
            if group != "all":
                sample_output_dir = f"{sample_output_dir}/{group}"
                LOGGER.info("[Beginning Group {}]".format(group.title()))
            ## Initialize Model
            model = embed.Word2Vec(verbose=True, **config.get("model_kwargs",{}))
            ## Add Phrasers (Optionally)
            if args.phraser_dir is not None:
                model = model.assign_phrasers(phrasers)
            ## Fit Model
            model = model.fit(group_sample_stream,
                              phrase_learning_only=args.phraser_only,
                              extract_vocabulary=args.extract_vocabulary)
            ## Cache Model and Embeddings (and Vocabulary if --extract_vocabulary)
            _ = model.save(f"{sample_output_dir}/")
            _ = model.write(f"{sample_output_dir}/embeddings")
    ## Done
    LOGGER.info("[Script Complete!]")

###################
### Execution
###################

## Main Program
if __name__ == "__main__":
    _ = main()