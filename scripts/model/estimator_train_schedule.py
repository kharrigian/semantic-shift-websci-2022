
"""
Schedule multiple training scripts across pairs of labeled and unlabeled datasets
"""

######################
### Configuration
######################

## Base Output Directory
OUTPUT_DIR = "./data/results/estimator/train-adaptation/"

## Root Project Directory
BASE_DIR = "/export/c01/kharrigian/semantic-shift/websci-2022/"

## Dataset Paths
LABELED_DATASETS = {
    "twitter":{
        "clpsych":f"{BASE_DIR}/data/results/estimator/embeddings/clpsych/",
        "multitask":f"{BASE_DIR}/data/results/estimator/embeddings/multitask/"
    },
    "reddit":{
        "smhd":f"{BASE_DIR}/data/results/estimator/embeddings/smhd/",
        "wolohan":f"{BASE_DIR}/data/results/estimator/embeddings/wolohan/"
    }
}
UNLABELED_DATASETS = {
    "twitter":{
        "gardenhose":f"{BASE_DIR}/data/results/estimator/embeddings/gardenhose/"
    },
    "reddit":{
        "active":f"{BASE_DIR}/data/results/estimator/embeddings/active/"
    }
}

## Training Configuration Templates
CONFIGURATIONS = {
    "twitter":"./configs/experiments/estimator/parameterized/train/twitter.json",
    "reddit":"./configs/experiments/estimator/parameterized/train/reddit.json"
}

## Training Script Flags (Comment Out Those Not Desired)
ADDED_FLAGS = [
    # "--classify_enforce_frequency_filter",
    # "--classify_enforce_classwise_frequency_filter",
    "--classify_enforce_min_shift_frequency_filter",
    # "--classify_enforce_target"
]

## Scheduling Parameters
GRID_JOBS = 8
GRID_MEMORY_PER_JOB = 32
GRID_LOG_DIR = f"{BASE_DIR}/logs/estimator/train/"
GRID_MAX_ARRAY_SIZE = 500
GRID_MAX_CONCURRENT_TASKS = 8

######################
### Imports
######################

## Standard Library
import os
import json
import subprocess
from glob import glob
from textwrap import dedent
from itertools import product
from copy import deepcopy

## Private
from semshift.util.helpers import chunks
from semshift.util.logging import initialize_logger

#####################
### Globals
#####################

## Create Logger
LOGGER = initialize_logger()

#####################
### Functions
#####################

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
    #$ -N estimator_train
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

def format_parallel_script_train(lower_bound,
                                 upper_bound,
                                 output_dir,
                                 config_dir,
                                 memory=8,
                                 num_jobs=1,
                                 log_dir="./logs/",
                                 flags=[],
                                 max_tasks_concurrent=8):
    """

    """

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
    python scripts/experiments/estimator_train.py {}/$SGE_TASK_ID.json --output_dir {} --jobs {} {}
    """.format(header,
               init,
               config_dir,
               output_dir,
               num_jobs,
               " ".join(flags))
    script = dedent(script)
    script = script.replace("//","/")
    return script

def schedule_train(schedule_dir,
                   output_dir,
                   flags=[],
                   jobs=1,
                   memory_per_job=8,
                   max_array_size=500,
                   max_tasks_concurrent=8):
    """
    
    """
    ## Identify Job Array Bounds
    n_configs = len(glob(f"{schedule_dir}/configs/*.json"))
    filenames_bounds = [(i[0], i[-1]) for i in chunks(list(range(1, n_configs+1)), max_array_size)]
    ## Create Job Directory and Log Directory
    job_dir = f"{schedule_dir}/jobs/"
    log_dir = f"{schedule_dir}/logs/"
    if not os.path.exists(job_dir):
        _ = os.makedirs(job_dir)
    if not os.path.exists(log_dir):
        _ = os.makedirs(log_dir)
    ## Create Scripts
    job_files = []
    for lower, upper in filenames_bounds:
        ## Generate Script
        script = format_parallel_script_train(lower_bound=lower,
                                              upper_bound=upper,
                                              output_dir=output_dir,
                                              config_dir=f"{schedule_dir}/configs/",
                                              memory=memory_per_job,
                                              num_jobs=jobs,
                                              log_dir=log_dir,
                                              flags=flags,
                                              max_tasks_concurrent=max_tasks_concurrent)
        ## Write Script File
        script_file = f"{job_dir}/train_{lower}_{upper}.sh".replace("//","/")
        with open(script_file,"w") as the_file:
            the_file.write(script)
        ## Cache
        job_files.append(script_file)
    ## Schedule Jobs
    LOGGER.info(f"[Scheduling {len(job_files)} Job Arrays for {n_configs} Samples]")
    for job_file in job_files:
        command = f"qsub {job_file}"
        job_id = subprocess.check_output(command, shell=True)
        LOGGER.info(job_id)

def create_configurations():
    """

    """
    ## Initialize Cache of Scheduling Directories
    schedule_directories = []
    ## Iterate Through Platforms and Datasets
    for platform, platform_datasets in LABELED_DATASETS.items():
        ## Identify Config File
        platform_config_file = CONFIGURATIONS[platform]
        ## Load Base Platform Configuration
        with open(platform_config_file,"r") as the_file:
            platform_config = json.load(the_file)
        ## Iterate Over Pairs
        for labeled_dataset, labeled_dataset_path in platform_datasets.items():
            for unlabeled_dataset, unlabeled_dataset_path in UNLABELED_DATASETS[platform].items():
                ## Pair ID
                pair_id = f"{labeled_dataset}-{unlabeled_dataset}"
                ## Initialize Schedule Directory
                schedule_dir = f"{OUTPUT_DIR}/{pair_id}/scheduler/"
                schedule_directories.append(schedule_dir)
                for filetype in ["configs","jobs"]:
                    schedule_subdir = f"{schedule_dir}/{filetype}/"
                    if not os.path.exists(schedule_subdir):
                        _ = os.makedirs(schedule_subdir)
                ## Get Samples from Labeled and Unlabeled)
                labeled_dataset_sample_splits = sorted(map(lambda x: int(x.split("/")[-1].split(".")[0]), glob(f"{labeled_dataset_path}/subsets/splits/{labeled_dataset}/*.json")))
                unlabeled_dataset_sample_splits = sorted(map(lambda x: int(x.split("/")[-1].split(".")[0]), glob(f"{unlabeled_dataset_path}/subsets/splits/{unlabeled_dataset}/*.json")))
                ## Create Pairs
                sample_pairs = list(product(labeled_dataset_sample_splits, unlabeled_dataset_sample_splits))
                ## Iterate Through Pairs
                for sample_id, (labeled_sample, unlabeled_sample) in enumerate(sample_pairs):
                    ## 1-Index for Writing
                    sample_id_sge = sample_id + 1
                    ## Copy Base Configuration
                    pair_config = deepcopy(platform_config)
                    ## Update ID
                    pair_config["experiment_id"] = f"{pair_id}/{labeled_sample}-{unlabeled_sample}"
                    ## Update Source
                    pair_config["source"]["config"] = f"{labeled_dataset_path}/subsets/config.json"
                    pair_config["source"]["splits"] = f"{labeled_dataset_path}/subsets/splits/{labeled_dataset}/{labeled_sample}.json"
                    pair_config["source"]["embeddings"] = f"{labeled_dataset_path}/subsets/models/sample-{labeled_sample}/"
                    ## Update Target
                    pair_config["target"]["config"] = f"{unlabeled_dataset_path}/subsets/config.json"
                    pair_config["target"]["splits"] = f"{unlabeled_dataset_path}/subsets/splits/{unlabeled_dataset}/{unlabeled_sample}.json"
                    pair_config["target"]["embeddings"] = f"{unlabeled_dataset_path}/subsets/models/sample-{unlabeled_sample}/"
                    ## Update Evaluation
                    pair_config["evaluation"]["config"] = f"{labeled_dataset_path}/subsets/config.json"
                    pair_config["evaluation"]["splits"] = f"{labeled_dataset_path}/subsets/splits/{labeled_dataset}/{labeled_sample}.json"
                    ## Write Configuration
                    sample_config_file = f"{schedule_dir}/configs/{sample_id_sge}.json"
                    with open(sample_config_file,"w") as the_file:
                        _ = json.dump(pair_config, the_file)
    return schedule_directories

def main():
    """

    """
    ## Create Configurations
    schedule_directories = create_configurations()
    ## Schedule Jobs
    for schedule_dir in schedule_directories:
        _ = schedule_train(schedule_dir,
                           output_dir=OUTPUT_DIR,
                           flags=ADDED_FLAGS,
                           jobs=GRID_JOBS,
                           memory_per_job=GRID_MEMORY_PER_JOB,
                           max_array_size=GRID_MAX_ARRAY_SIZE,
                           max_tasks_concurrent=GRID_MAX_CONCURRENT_TASKS)
    LOGGER.info("[Scheduling Complete]")

#######################
### Execution
#######################

if __name__ == "__main__":
    _ = main()