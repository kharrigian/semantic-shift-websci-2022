
"""
Parallelize tracking script across CLSP grid. Note that this is better
for a small number of large files than a large number of small files. For 
the latter, recommend just calling the track_apply.py script directly
and using multiprocessing within the same job.
"""

#################
### Configuration
#################

## Experiment Info
EXPERIMENT_ID = "gardenhose"
EXPERIMENT_OUTDIR = "./data/results/track/gardenhose/"
EXPERIMENT_BASE_CONFIG = "./configs/track/gardenhose.json"
EXPERIMENT_DATASETS = [
    "/export/c01/kharrigian/semantic-shift-websci-2022/data/processed/twitter/gardenhose/*.gz"
]

## CLSP Grid Parameters
GRID_USERNAME = "kharrigian"
GRID_MEMORY_REQUEST_SIZE = 8
GRID_MAX_ARRAY = 500
LOG_DIR = "./logs/track/"

#################
### Imports
#################

## Standard Library
import os
import json
import subprocess
from glob import glob
from textwrap import dedent

## Project Specific
from semshift.util.helpers import chunks
from semshift.util.logging import initialize_logger

#################
### Globals
#################

## Logger
LOGGER = initialize_logger()

#################
### Functions
#################

def get_header(nstart,
               nend,
               log_dir,
               memory=8):
    """

    """
    header=f"""
    #$ -cwd
    #$ -S /bin/bash
    #$ -m eas
    #$ -N tracker
    #$ -t {nstart}-{nend}
    #$ -e {log_dir}
    #$ -o {log_dir}
    #$ -pe smp 1
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
                 log_dir,
                 memory=8):
    """

    """
    ## Construct Script
    header = get_header(lower_bound,
                        upper_bound,
                        log_dir=log_dir,
                        memory=memory)
    init = get_init_env()
    script="""
    #!/bin/bash
    {}
    {}
    python scripts/model/track_apply.py {}/{} --output_dir {}
    """.format(header, init, output_dir, "configs/${SGE_TASK_ID}.json", output_dir)
    script = dedent(script)
    script = script.replace("//","/")
    ## Write Script File
    script_file = f"{output_dir}/jobs/track_{lower_bound}_{upper_bound}.sh".replace("//","/")
    with open(script_file,"w") as the_file:
        the_file.write(script)
    return script_file

def main():
    """

    """
    ## Output Directory
    LOGGER.info("Initializing Directories")
    for d in [EXPERIMENT_OUTDIR,
            f"{EXPERIMENT_OUTDIR}/configs/",
            f"{EXPERIMENT_OUTDIR}/jobs/",
            f"{EXPERIMENT_OUTDIR}/results/",
            LOG_DIR]:
        if not os.path.exists(d):
            _ = os.makedirs(d)
    ## Load Base Config
    LOGGER.info("Loading Base Configuration")
    with open(EXPERIMENT_BASE_CONFIG,"r") as the_file:
        config = json.load(the_file)
    config["jobs"] = 1
    config["chunksize"] = 1
    ## Identify Filenames
    LOGGER.info("Finding Files")
    filenames = []
    for dataset in EXPERIMENT_DATASETS:
        filenames.extend(glob(dataset))
    filenames = sorted(filenames)
    ## Generate Configuration Files
    LOGGER.info("Writing Configuration Files")
    config_files = []
    for f, filename in enumerate(filenames):
        config["datasets"][0]["dataset"] = filename
        config["experiment_name"] = f"results/{f+1}/"
        config_file = f"{EXPERIMENT_OUTDIR}/configs/{f+1}.json"
        with open(config_file,"w") as the_file:
            json.dump(config, the_file)
        config_files.append(config_file)
    ## Generate Scripts
    LOGGER.info("Writing Job Files")
    array_bounds = [(i[0], i[-1]) for i in chunks(list(range(1, len(config_files)+1)), GRID_MAX_ARRAY)]
    array_files = []
    for lower, upper in array_bounds:
        array_file = make_sh_file(lower_bound=lower,
                                  upper_bound=upper,
                                  output_dir=EXPERIMENT_OUTDIR,
                                  log_dir=LOG_DIR,
                                  memory=GRID_MEMORY_REQUEST_SIZE)
        array_files.append(array_file)
    ## Schedule Jobs
    LOGGER.info(f"Scheduling {len(array_files)} Job Arrays for {len(filenames)} Files")
    for array_file in array_files:
        command = f"qsub {array_file}"
        job_id = subprocess.check_output(command, shell=True)
        LOGGER.info(job_id)
    LOGGER.info("Scheduler Complete!")

#################
### Execute
#################

if __name__ == "__main__":
    _ = main()

