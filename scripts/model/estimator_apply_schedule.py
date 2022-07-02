
"""
Apply multiple train-adaptation runs to an unlabeled, vectorized sample
"""

#########################
### Configuration
#########################

## Output Directory
BASE_DIR = "/export/c01/kharrigian/semantic-shift-websci-2022/"
BASE_OUTPUT_DIR = f"{BASE_DIR}/data/results/estimator/apply/"

## Configs
BASE_CONFIG_FILES = {
    "twitter":"configs/estimator/parameterized/apply/twitter.json",
    "reddit":"configs/estimator/parameterized/apply/reddit.json"
}

## Model Paths
MODEL_PATHS = {
    "twitter":{
        "clpsych":"./data/results/estimator/train-adaptation/clpsych-gardenhose/",
        "multitask":"./data/results/estimator/train-adaptation/multitask-gardenhose/",
    },
    "reddit":{
        "smhd":"./data/results/estimator/train-adaptation/smhd-active/",
        "wolohan":"./data/results/estimator/train-adaptation/wolohan-active/",
    }
}

## Vector Paths
VECTOR_PATHS = {
    "gardenhose":"./data/results/estimator/datasets/gardenhose/vectors/",
    "active":"./data/results/estimator/datasets/active/vectors/"
}

## Model to Vector Pairs
MODEL_TO_VECTOR_PAIRS = {
    "clpsych":["gardenhose"],
    "multitask":["gardenhose"],
    "smhd":["active"],
    "wolohan":["active"]
}

## Application Parameters
APPLY_PARAMS = {
    "--models_per_chunk":10,
    "--vector_files_per_chunk":250,
    "--skip_existing":""
}

## Whether to Use Scheduling (SGE Access)
USE_SCHEDULER = False

## Scheduling Parameters
GRID_JOBS = 8
GRID_MEMORY_PER_JOB = 64
GRID_MAX_ARRAY_SIZE = 500
GRID_MAX_CONCURRENT_TASKS = 2
GRID_LOG_DIR = f"{BASE_DIR}/logs/estimator/apply/"

######################
### Imports
######################

## Standard Library
import os
import json
import subprocess
from glob import glob
from textwrap import dedent
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

def format_parallel_script_apply(lower_bound,
                                 upper_bound,
                                 config_dir,
                                 memory=8,
                                 num_jobs=1,
                                 log_dir="./logs/",
                                 flags={},
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
    python scripts/model/estimator_apply.py {}/$SGE_TASK_ID.json {} --run_predict --run_analyze
    """.format(header,
               init,
               config_dir,
               " ".join([f"{x} {y}" for x, y in flags.items()]))
    script = dedent(script)
    script = script.replace("//","/")
    return script

def schedule_apply(schedule_dir,
                   flags={},
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
        script = format_parallel_script_apply(lower_bound=lower,
                                              upper_bound=upper,
                                              config_dir=f"{schedule_dir}/configs/",
                                              memory=memory_per_job,
                                              num_jobs=jobs,
                                              log_dir=log_dir,
                                              flags=flags,
                                              max_tasks_concurrent=max_tasks_concurrent)
        ## Write Script File
        script_file = f"{job_dir}/apply_{lower}_{upper}.sh".replace("//","/")
        with open(script_file,"w") as the_file:
            the_file.write(script)
        ## Cache
        job_files.append(script_file)
    ## Schedule Jobs
    if USE_SCHEDULER:
        LOGGER.info(f"[Scheduling {len(job_files)} Job Arrays for {n_configs} Samples]")
        for job_file in job_files:
            command = f"qsub {job_file}"
            job_id = subprocess.check_output(command, shell=True)
            LOGGER.info(job_id)

def get_model_dirs(trained_dataset_path):
    """

    """
    trained_model_dirs = {}
    for directory in sorted(glob(f"{trained_dataset_path}/*-*/models/")):
        directory_id = os.path.dirname(directory).split("/")[-2]
        trained_model_dirs[directory_id] = directory
    return trained_model_dirs

def create_configurations():
    """

    """
    ## Holder of Schedule Directories (Where Configs Will End Up)
    schedule_dirs = []
    ## Iterate Over Platforms of Trained Models
    for platform, platform_datasets in MODEL_PATHS.items():
        ## Get Base Config for the Platform
        with open(BASE_CONFIG_FILES[platform],"r") as the_file:
            platform_config = json.load(the_file)
        ## Iterate Over Pairs
        for trained_dataset, trained_dataset_path in platform_datasets.items():
            ## Identify Models
            trained_dataset_models = get_model_dirs(trained_dataset_path)
            ## Identify Apply Datasets
            apply_datasets = MODEL_TO_VECTOR_PAIRS[trained_dataset]
            ## Iterate Over Apply Datasets
            for apply_dataset in apply_datasets:
                ## Create Output Directory (And Scheduler)
                pair_output_dir = f"{BASE_OUTPUT_DIR}/{platform}/{trained_dataset}-{apply_dataset}/"
                for subname in ["configs","jobs"]:
                    pair_schedule_subdir = f"{pair_output_dir}/scheduler/{subname}/".replace("//","/")
                    if not os.path.exists(pair_schedule_subdir):
                        _ = os.makedirs(pair_schedule_subdir)
                ## Add Schedule Directory
                schedule_dirs.append(f"{pair_output_dir}/scheduler/")
                ## Vector Path
                apply_vector_path = VECTOR_PATHS[apply_dataset]
                ## Iterate Over Dataset Model Directories
                for mind, (model_id, model_dir_path) in enumerate(trained_dataset_models.items()):
                    ## SGE ID
                    mind_sge = mind + 1
                    ## Copy Base Platform Config
                    model_apply_config = deepcopy(platform_config)
                    ## Update Parameters
                    model_apply_config["experiment_id"] = f"{platform}/{trained_dataset}-{apply_dataset}/{model_id}"
                    model_apply_config["output_dir"] = BASE_OUTPUT_DIR
                    model_apply_config["models_dir"] = model_dir_path
                    model_apply_config["vector_dir"] = apply_vector_path
                    ## Write Config
                    with open(f"{pair_output_dir}/scheduler/configs/{mind_sge}.json","w") as the_file:
                        json.dump(model_apply_config, the_file)
    ## Output Directories
    return schedule_dirs

def main():
    """

    """
    ## Create Configurations
    schedule_directories = create_configurations()
    ## Update Flags
    flags = deepcopy(APPLY_PARAMS)
    flags["--jobs"] = GRID_JOBS
    ## Schedule Jobs
    for schedule_dir in schedule_directories:
        _ = schedule_apply(schedule_dir,
                           flags=flags,
                           jobs=GRID_JOBS,
                           memory_per_job=GRID_MEMORY_PER_JOB,
                           max_array_size=GRID_MAX_ARRAY_SIZE,
                           max_tasks_concurrent=GRID_MAX_CONCURRENT_TASKS)
    if USE_SCHEDULER:
        LOGGER.info("[Scheduling Complete]")
    else:
        LOGGER.info("[Script Writing Complete]")

##########################
### Execution
##########################

if __name__ == "__main__":
    _ = main()