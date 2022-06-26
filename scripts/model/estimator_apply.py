
"""
Apply trained models to vectorized data (e.g. predictions of depression as a function of vocabulary)
"""

###################
### Imports
###################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from glob import glob
from functools import partial
from multiprocessing import Pool

## External Library
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import matplotlib.pyplot as plt

## Private
from semshift.util.helpers import chunks, flatten
from semshift.util.logging import initialize_logger

## Project-Specific
_ = sys.path.append(os.path.abspath(os.path.dirname(__file__))+"/")
from helpers import align_vocab

###################
### Globals
###################

## Logging
LOGGER = initialize_logger()

###################
### Helpers
###################

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

###################
### Functions
###################

def parse_command_line():
    """
    
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("config",
                            type=str)
    _ = parser.add_argument("--run_predict",
                            default=False,
                            action="store_true")
    _ = parser.add_argument("--run_analyze",
                            default=False,
                            action="store_true")
    _ = parser.add_argument("--run_analyze_merged",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--models_per_chunk",
                            type=int,
                            default=10)
    _ = parser.add_argument("--vector_files_per_chunk",
                            type=int,
                            default=100)
    _ = parser.add_argument("--skip_existing",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--jobs",
                            type=int,
                            default=1)
    args = parser.parse_args()
    return args

def load_configuration(filename):
    """
    
    """
    ## Check for File
    if not os.path.exists(filename):
        raise FileNotFoundError("Configuration File Not Found: {}".format(filename))
    ## Load
    with open(filename,"r") as the_file:
        config = json.load(the_file)
    ## Initialize Output Directory
    output_dir = "{}/{}/".format(config["output_dir"], config["experiment_id"]).replace("//","/")
    if not os.path.exists(output_dir):
        LOGGER.info("Initializing Output Directory: {}".format(output_dir))
        _ = os.makedirs(output_dir)
    if not os.path.exists("{}/predictions/".format(output_dir)):
        _ = os.makedirs("{}/predictions/".format(output_dir))
    ## Store Configuration
    with open("{}/config.json".format(output_dir),"w") as the_file:
        _ = json.dump(config, the_file, indent=4, sort_keys=False)
    ## Return
    return config, output_dir

def load_txt(filename):
    """
    
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("File not found: {}".format(filename))
    data = []
    with open(filename,"r") as the_file:
        for line in the_file:
            data.append(line.strip())
    return data

def _get_model_meta(model_path):
    """

    """
    selector, percentile, sample = os.path.dirname(model_path).split("/")[-3:]
    percentile, sample = int(percentile), int(sample)
    metadata = {"selector":selector,"percentile":percentile,"sample":sample}
    return metadata

def _load_model(model_path):
    """
    
    """
    ## Identify Directory
    model_dir = os.path.dirname(model_path) + "/"
    ## Load Model, Preprocessor, and Vocabulary
    vocabulary = load_txt(f"{model_dir}/../vocabulary.txt")
    preprocessor = joblib.load(f"{model_dir}/preprocessor.joblib")
    model = joblib.load(model_path)
    ## Model Information (From Filepath)
    metadata = _get_model_meta(model_path)
    ## Return
    return vocabulary, preprocessor, model, metadata

def load_models(model_paths,
                jobs=1):
    """
    
    """
    ## Load In Parallel
    with Pool(jobs) as mp:
        models = list(mp.imap_unordered(_load_model, model_paths))
    return models

def load_resources(config):
    """
    
    """
    model_files = sorted(glob("{}/*/*/*/model.joblib".format(config["models_dir"])))
    vector_files = sorted(glob("{}/data.*.npz".format(config["vector_dir"])))
    vector_vocabulary = load_txt("{}/vocabulary.txt".format(config["vector_dir"]))
    return model_files, vector_files, vector_vocabulary

def _load_vectors(vector_data_file):
    """
    
    
    """
    ## Load Each Resource
    X = sparse.load_npz(vector_data_file)
    n_posts = np.load(vector_data_file.replace("data.","n_posts.").replace(".npz",".npy"))
    tau = np.load(vector_data_file.replace("data.","tau.").replace(".npz",".npy"))
    user_ids = load_txt(vector_data_file.replace("data.","users.").replace(".npz",".txt"))
    ## Return
    return X, n_posts, tau, user_ids

def load_vectors(vector_data_files,
                 jobs=1):
    """
    
    """
    ## Load using Multiprocessing
    with Pool(jobs) as mp:
        results = list(mp.imap_unordered(_load_vectors, vector_data_files))
    ## Parse
    X = sparse.vstack([r[0] for r in results])
    n_posts = np.hstack([r[1] for r in results])
    tau = np.hstack([r[2] for r in results])
    user_ids = flatten([r[3] for r in results])
    return X, n_posts, tau, user_ids

def _generate_model_predictions(model,
                                vectors,
                                vectors_vocabulary):
    """
    
    """
    ## Parse Model
    model_vocabulary = model[0]
    model_preprocessor = model[1]
    model_estimator = model[2]
    model_metadata = model[3]
    ## Align Vocabulary
    X_aligned = align_vocab(X=vectors,
                            xs_vocab=vectors_vocabulary,
                            xt_vocab=model_vocabulary)
    ## Transform
    X_aligned = model_preprocessor.transform(X_aligned)
    ## Make Predictions
    y_pred = model_estimator.predict_proba(X_aligned)[:,1]
    ## Return
    return y_pred, model_metadata

def _generate_predictions(models,
                          vectors,
                          vectors_vocabulary,
                          jobs=1):
    """
    
    """
    ## Parameterize Helper
    helper = partial(_generate_model_predictions, vectors=vectors, vectors_vocabulary=vectors_vocabulary)
    ## Generate Predictions
    with Pool(jobs) as mp:
        predictions = list(mp.imap_unordered(helper, models))
    return predictions

def generate_predictions(models,
                         vectors,
                         vectors_vocabulary,
                         output_dir,
                         min_n_posts,
                         models_per_chunk=10,
                         vector_files_per_chunk=100,
                         skip_existing=True,
                         jobs=1):
    """
    
    """
    ## Model ID Helper
    model_id = lambda metadata: (metadata["selector"], metadata["percentile"], metadata["sample"])
    ## Check for Completion
    if skip_existing:
        models = list(filter(lambda mfile: not os.path.exists("{}/predictions/{}.{}.{}.json.gz".format(output_dir, *model_id(_get_model_meta(mfile)))), models))
        if len(models) == 0:
            LOGGER.info("[Predictions for All Models Already Completed]")
            return None
    ## Get Chunks
    model_chunks = list(chunks(models, models_per_chunk))
    vector_chunks = list(chunks(vectors, vector_files_per_chunk))
    ## Iterate
    for ii, mc in tqdm(enumerate(model_chunks), total=len(model_chunks), file=sys.stdout, position=0, leave=True, desc="[Model Chunks]"):
        ## Load Models
        models_mc = load_models(model_paths=mc, jobs=jobs)
        ## Initialize Model Prediction Cache
        models_mc_predictions = {}
        ## Iterate Over Vector Chunks
        for vc in tqdm(vector_chunks, total=len(vector_chunks), file=sys.stdout, position=1, leave=ii==len(model_chunks)-1, desc="[Vector Chunks]"):
            ## Load Vectors
            X_vc, n_vc, tau_vc, users_vc = load_vectors(vector_data_files=vc, jobs=jobs)
            ## Post Mask
            if min_n_posts is not None:
                ## Get Mask
                vc_mask = (n_vc >= min_n_posts)
                ## Skip Missing
                if vc_mask.sum() == 0:
                    continue
                vc_mask = vc_mask.nonzero()[0]
                ## Apply Mask
                X_vc = X_vc[vc_mask]
                n_vc = n_vc[vc_mask]
                tau_vc = tau_vc[vc_mask]
                users_vc = [users_vc[vcm] for vcm in vc_mask]
            ## Make Predictions
            mc_vc_predictions = _generate_predictions(models=models_mc,
                                                      vectors=X_vc,
                                                      vectors_vocabulary=vectors_vocabulary,
                                                      jobs=jobs)
            ## Cache Predictions
            for mpred, mmeta in mc_vc_predictions:
                ## Get ID
                mid = model_id(mmeta)
                ## Check ID
                if mid not in models_mc_predictions:
                    models_mc_predictions[mid] = {}
                ## Format Predictions
                mpreds_fmt = {(u, t):p for u, t, p in zip(users_vc, tau_vc, mpred)}
                ## Cache
                models_mc_predictions[mid].update(mpreds_fmt)
        ## Cache Model Predictions
        for mid, mpreds in models_mc_predictions.items():
            mfile = "{}/predictions/{}.{}.{}.json.gz".format(output_dir, *mid).replace("//","/")
            with gzip.open(mfile, "wt") as the_file:
                for m in sorted(mpreds.items()):
                    mj = {"data":[m[0][0], int(m[0][1]), m[1]]}
                    the_file.write("{}\n".format(json.dumps(mj)))

def file2att(f):
    """

    """
    fsplit = os.path.basename(f).replace(".json.gz","").split(".")
    if len(fsplit) == 4:
        fsplit = [".".join(fsplit[:2]), fsplit[2], fsplit[3]]
    fsplit = [fsplit[0], int(fsplit[1]), int(fsplit[2])]
    return fsplit

def get_predictions(output_dir):
    """

    """
    ## Check Directory
    prediction_dir = f"{output_dir}/predictions/"
    if not os.path.exists(prediction_dir):
        raise FileNotFoundError("Prediction directory does not exist, suggesting prediction has not been run.")
    ## Check/Identify Files
    prediction_files = glob(f"{prediction_dir}/*.json.gz")
    if len(prediction_files) == 0:
        raise FileNotFoundError("No prediction files were found.")
    ## Attribute Files
    prediction_files = {f:file2att(f) for f in prediction_files}
    ## Group Files
    prediction_groups = {}
    for fname, fatts in prediction_files.items():
        if fatts[0] not in prediction_groups:
            prediction_groups[fatts[0]] = {}
        if fatts[1] not in prediction_groups[fatts[0]]:
            prediction_groups[fatts[0]][fatts[1]] = []
        prediction_groups[fatts[0]][fatts[1]].append((fatts[2], fname))
    ## Sort Groups
    for sel, sel_group in prediction_groups.items():
        for sel_per, sel_per_files in sel_group.items():
            prediction_groups[sel][sel_per] = [j[1] for j in sorted(sel_per_files, key=lambda i: i[0])]
    ## Return
    return prediction_files, prediction_groups

def measure_prevalence_change(filename,
                              thresholds=[0.5]):
    """

    """
    ## Load File Data
    data = []
    with gzip.open(filename,"r") as the_file:
        for line in the_file:
            data.append(json.loads(line)["data"])
    data = pd.DataFrame(data, columns=["user_id_str","time_period","y_pred"])
    ## Within Subjects
    within_subjects_users = data["user_id_str"].value_counts() == 2
    within_subjects_users = set(within_subjects_users.loc[within_subjects_users].index)
    ## Compute Estimates
    estimates = []
    for within_subject in [False, True]:
        ## Subjects
        if within_subject:
            wdata = data.loc[data["user_id_str"].isin(within_subjects_users)]
        else:
            wdata = data
        ## Confidence Threshold
        for threshold in thresholds:
            threshold_dist = pd.concat([wdata["time_period"], 
                                       (wdata["y_pred"] > threshold).astype(int)],axis=1).groupby(["time_period"]).agg({"y_pred":[len,sum]})["y_pred"]
            threshold_dist["rate"] = threshold_dist["sum"] / threshold_dist["len"]
            change = threshold_dist.loc[1,"rate"] - threshold_dist.loc[0,"rate"]
            rel_change = change / threshold_dist.loc[0,"rate"]
            estimates.append({
                "measure":"threshold",
                "measure_value":threshold,
                "within_subject":within_subject,
                "value_start":threshold_dist.loc[0,"rate"] * 100,
                "value_end":threshold_dist.loc[1,"rate"] * 100,
                "delta":change * 100,
                "support":threshold_dist.loc[[0,1],"len"].to_dict()
            })
            estimates.append({
                "measure":"threshold-relative",
                "measure_value":threshold,
                "within_subject":within_subject,
                "value_start":threshold_dist.loc[0,"rate"] * 100,
                "value_end":threshold_dist.loc[1,"rate"] * 100,
                "delta":rel_change * 100,
                "support":threshold_dist.loc[[0,1],"len"].to_dict()
            })
        ## Average Score
        mean = wdata.groupby(["time_period"])["y_pred"].mean()
        estimates.append({
            "measure":"statistic",
            "measure_value":"mean",
            "within_subject":within_subject,
            "value_start":mean.loc[0],
            "value_end":mean.loc[1],
            "delta":mean.loc[1] - mean.loc[0],
            "support":wdata.groupby(["time_period"]).size().to_dict()
        })
        estimates.append({
            "measure":"statistic-relative",
            "measure_value":"mean",
            "within_subject":within_subject,
            "value_start":mean.loc[0],
            "value_end":mean.loc[1],
            "delta":(mean.loc[1] - mean.loc[0])/mean.loc[0] * 100,
            "support":wdata.groupby(["time_period"]).size().to_dict()
        })
    return filename, estimates

def estimate_prevalence_change(prediction_files,
                               jobs=1):
    """

    """
    ## Compute Estimates
    with Pool(jobs) as mp:
        estimates = dict(tqdm(mp.imap_unordered(measure_prevalence_change, prediction_files),
                              total=len(prediction_files),
                              desc="[Estimating Prevalence Change]",
                              file=sys.stdout))
    ## Format Estimates
    estimates_df = []
    for prediction_file, file_estimates in estimates.items():
        file_meta = dict(zip(["vocabulary_selector","vocabulary_selector_percentile","fold"],
                              prediction_files[prediction_file]))
        file_estimates = [{**est, **file_meta} for est in file_estimates]
        estimates_df.extend(file_estimates)
    estimates_df = pd.DataFrame(estimates_df)
    ## Sort
    estimates_df = estimates_df.sort_values(["measure","measure_value","within_subject","vocabulary_selector","vocabulary_selector_percentile","fold"])
    estimates_df.reset_index(drop=True, inplace=True)
    return estimates_df

def plot_prevalence_change(prevalence_estimates,
                           measure,
                           measure_value,
                           within_subjects,
                           title=None):
    """

    """
    ## Isolate Estimates
    mestimates = prevalence_estimates.loc[(prevalence_estimates["measure"]==measure)&
                                          (prevalence_estimates["measure_value"]==measure_value)&
                                          (prevalence_estimates["within_subject"]==within_subjects)]
    ## Group Estimates
    mestimates_agg = mestimates.groupby(["vocabulary_selector","vocabulary_selector_percentile"]).agg({
        "delta":bootstrap_ci
    })["delta"]
    mestimates_agg = bootstrap_tuple_to_df(mestimates_agg)
    ## Plot By Groups
    selector_order = [
        "cumulative",
        "intersection",
        "random",
        "frequency",
        "chi2",
        "coefficient",
        "overlap",
        "overlap-coefficient_0.5",
    ]
    selectors = list(filter(lambda i: i in mestimates_agg.index.levels[0], selector_order))
    ## Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    for s, selector in enumerate(selectors):
        ## Select
        smestimates_agg = mestimates_agg.loc[selector]
        ## Interpolation
        if selector not in ["cumulative","intersection"]:
            ss = smestimates_agg.reindex(np.arange(smestimates_agg.index.min(), smestimates_agg.index.max()+1))
            ss = ss.interpolate("quadratic", limit_direction="both")
        ## Plot
        if selector in ["cumulative","intersection"]:
            ax.fill_between([0, 100],
                            (smestimates_agg["lower"]).values[0],
                            (smestimates_agg["upper"]).values[0],
                            alpha=0.2,
                            color=f"C{s}",
                            linewidth=2)
            ax.axhline(smestimates_agg.loc[100,"median"],
                       linewidth=2,
                       alpha=0.8,
                       color=f"C{s}",
                       label=selector)
        else:
            ax.fill_between(ss.index + s * 0.5,
                            ss["lower"],
                            ss["upper"],
                            color=f"C{s}",
                            linewidth=2,
                            alpha=0.2)
            ax.errorbar(smestimates_agg.index + s * 0.5,
                        smestimates_agg["median"],
                        yerr=np.vstack([(smestimates_agg["median"]-smestimates_agg["lower"]).values,
                                        (smestimates_agg["upper"]-smestimates_agg["median"]).values]),
                        color=f"C{s}",
                        label=selector,
                        linewidth=0,
                        marker="o",
                        alpha=0.8,
                        elinewidth=2,
                        capsize=5)
    if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 0:
        ax.axhline(0, color="black", zorder=-1, alpha=0.5, linewidth=1.5, linestyle="--")
    ax.legend(loc="best",ncol=2)
    ax.set_xlim(0,100)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_ylabel("Estimated Change",fontweight="bold",fontsize=16)
    ax.set_xlabel("Vocabulary Size", fontweight="bold",fontsize=16)
    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold", fontsize=16)
    fig.tight_layout()
    return fig

def main():
    """
    
    """
    ## Parse Command Line
    LOGGER.info("[Parsing Command Line]")
    args = parse_command_line()
    ## Check Inputs
    if not args.run_predict and not args.run_analyze and not args.run_analyze_merged:
        raise ValueError("Must run either prediction (--run_predict) or analysis (--run_analyze) or analysis merged (--run_analyze_merged).")
    if (args.run_predict or args.run_analyze) and args.run_analyze_merged:
        raise ValueError("Merged analysis should be done independent of other analysis types.")
    ## Load Configuration
    LOGGER.info("[Loading Configuration]")
    if not os.path.isdir(args.config):
        config, output_dir = load_configuration(args.config)
    else:
        LOGGER.info("Passed a directory, assuming this contains multiple results to merge.")
        if not args.run_analyze_merged:
            raise ValueError("Passing a directory is used to run a combined analysis over multiple samples.")
        config = None
        output_dir = args.config
    ## Prediction Stage
    if args.run_predict:
        LOGGER.info("[Running Prediction Stage]")
        ## Identify Model and Vector Files
        LOGGER.info("[Identifying Data Resources]")
        models, vectors, vectors_vocabulary = load_resources(config)
        ## Generate Predictions
        LOGGER.info("[Generating Predictions]")
        _ = generate_predictions(models=models,
                                 vectors=vectors,
                                 vectors_vocabulary=vectors_vocabulary,
                                 output_dir=output_dir,
                                 min_n_posts=config["min_n_posts"],
                                 models_per_chunk=args.models_per_chunk,
                                 vector_files_per_chunk=args.vector_files_per_chunk,
                                 skip_existing=args.skip_existing,
                                 jobs=args.jobs)
    ## Generate Summaries
    if args.run_analyze:
        LOGGER.info("[Running Analysis Stage]")
        ## Identify Prediction Groups
        prediction_files, prediction_groups = get_predictions(output_dir=output_dir)
        ## Compute Estimates
        LOGGER.info("[Computing Estimates]")
        prevalence_estimates = estimate_prevalence_change(prediction_files,
                                                          jobs=args.jobs)
        ## Save Estimates
        LOGGER.info("[Caching Estimates]")
        _ = prevalence_estimates.to_csv(f"{output_dir}/estimates.csv",index=False)
        ## Plot Results
        LOGGER.info("[Visualizing Estimates]")
        measure_groups = prevalence_estimates[["measure","measure_value","within_subject"]].drop_duplicates().values
        for msr, msrv, within in measure_groups:
            fig = plot_prevalence_change(prevalence_estimates,
                                         measure=msr,
                                         measure_value=msrv,
                                         within_subjects=within)
            _ = fig.savefig(f"{output_dir}/estimates.{msr}.{msrv}.{within}.png", dpi=150)
            plt.close(fig)
    ## Merged Summary
    if args.run_analyze_merged:
        ## Get Estimate Files
        estimate_files = glob(f"{output_dir}/*/estimates.csv")
        if len(estimate_files) == 0:
            raise FileNotFoundError("No Estimates Were Found.")
        ## Load Estimates
        LOGGER.info("[Loading Prevalence Estimates]")
        prevalence_estimates = []
        for estimate_file in estimate_files:
            estimate_file_df = pd.read_csv(estimate_file)
            estimate_file_df["sample_id"] = os.path.dirname(estimate_file).split("/")[-1]
            prevalence_estimates.append(estimate_file_df)
        prevalence_estimates = pd.concat(prevalence_estimates).reset_index(drop=True)
        ## Visualize Merged Estimates
        LOGGER.info("[Visualizing Estimates]")
        measure_groups = prevalence_estimates[["measure","measure_value","within_subject"]].drop_duplicates().values
        for msr, msrv, within in measure_groups:
            fig = plot_prevalence_change(prevalence_estimates,
                                         measure=msr,
                                         measure_value=msrv,
                                         within_subjects=within)
            _ = fig.savefig(f"{output_dir}/estimates.{msr}.{msrv}.{within}.png", dpi=150)
            plt.close(fig)
    ## Done
    LOGGER.info("[Script Complete]")

###################
### Execute
###################

if __name__ == "__main__":
    _ = main()