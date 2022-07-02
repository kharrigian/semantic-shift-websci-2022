
"""
Analyze model coefficients (baseline), their semantic stability, and changes in usage. Also 
"""

####################
### Imports
####################

## Standard Library
import os
import sys
import argparse
from glob import glob
from multiprocessing import Pool

## External Libary
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
import matplotlib.pyplot as plt

## Project-Specific
from semshift.util.logging import initialize_logger

## Non-Library
_ = sys.path.append(os.path.abspath(os.path.dirname(__file__))+"/")
from estimator_train import plot_performance_summary, plot_shift_influence_summary
from helpers import replace_emojis

####################
### Globals
####################

## Logging
LOGGER = initialize_logger()

## Mapping
LBL_MAP = {
    "cf_ratio":"Log Ratio (Pre/Post)",
    "df_ratio":"User Log Ratio (Pre/Post)",
    "overlap":"Semantic Similarity",
    "weight":"Cofficient Weight",
    "weight_absolute":"Coefficient Weight (Absolute)",
    "cf_ratio_absolute":"Log Ratio (Pre/Post, Absolute)",
    "df_ratio_absolute":"Log Ratio (Pre/Post, Absolute)",
}

####################
### Functions
####################

def parse_command_line():
    """

    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--models_dir",
                            type=str,
                            default=None)
    _ = parser.add_argument("--vector_dir",
                            type=str,
                            default=None)
    _ = parser.add_argument("--min_cf",
                            type=int,
                            default=None)
    _ = parser.add_argument("--min_df",
                            type=int,
                            default=None)
    _ = parser.add_argument("--jobs",
                            type=int,
                            default=1)
    args = parser.parse_args()
    if args.models_dir is None or args.vector_dir is None:
        raise ValueError("--models_dir and --vector_dir must be specified.")
    return args

def load_text(filename):
    """

    """
    data = []
    with open(filename,"r") as the_file:
        for line in the_file:
            data.append(line.strip())
    return data

def _load_usage(data_file):
    """

    """
    X = sparse.load_npz(data_file)
    tau = np.load(data_file.replace("data.","tau.").replace(".npz",".npy"))
    counts = np.vstack([X[(tau == tp).nonzero()[0]].sum(axis=0).A[0] for tp in [0, 1]])
    support = np.vstack([(X[(tau == tp).nonzero()[0]]!=0).sum(axis=0).A[0] for tp in [0, 1]])
    return counts, support

def load_usage(vector_dir,
               jobs=1):
    """

    """
    ## Load Vocabulary
    vocabulary = load_text(f"{vector_dir}/vocabulary.txt")
    ## Identify and Load Usage Counts by Time Period
    data_files = glob(f"{vector_dir}/data.*.npz")
    with Pool(jobs) as mp:
        results = list(tqdm(mp.imap_unordered(_load_usage, data_files), total=len(data_files), desc="[Vector File]", file=sys.stdout))
    ## Parse + Aggregate Results
    counts = np.vstack([
        np.sum([r[0][0] for r in results],axis=0),
        np.sum([r[0][1] for r in results],axis=0),
        np.sum([r[1][0] for r in results],axis=0),
        np.sum([r[1][1] for r in results],axis=0),
    ]).T
    counts = pd.DataFrame(counts,
                          index=vocabulary,
                          columns=["cf_start","cf_end","df_start","df_end"]).astype(int)
    return counts

def load_model_resources(models_dir):
    """

    """
    ## Filenames
    similarity_files = glob(f"{models_dir}/*-*/shift.summary.csv")
    ## Load Weights
    weight_files = sorted(glob(f"{models_dir}/*-*/baseline_classification_weights.csv"))
    weights = []
    for wf in weight_files:
        wf_df = pd.read_csv(wf)
        wf_df["sample"] = os.path.dirname(wf).split("/")[-1]
        weights.append(wf_df)
    weights = pd.concat(weights, axis=0, sort=False).reset_index(drop=True)
    ## Load Scores
    score_files = sorted(glob(f"{models_dir}/*-*/reduced_vocabulary_classification_scores.csv"))
    scores = []
    for sf in score_files:
        sf_df = pd.read_csv(sf)
        sf_df["sample"] = os.path.dirname(sf).split("/")[-1]
        sf_df = sf_df.loc[(sf_df["data_id"]=="evaluation")&(sf_df["group"]=="test")]
        scores.append(sf_df)
    scores = pd.concat(scores, axis=0, sort=False).reset_index(drop=True)
    ## Load Similarity
    similarity_files = sorted(glob(f"{models_dir}/*-*/shift.summary.csv"))
    similarity = []
    for sf in similarity_files:
        sf_df = pd.read_csv(sf, usecols=[0,1,4,5]).dropna()
        sf_df["sample"] = os.path.dirname(sf).split("/")[-1]
        similarity.append(sf_df)
    similarity = pd.concat(similarity, sort=False, axis=0).reset_index(drop=True)
    return scores, weights, similarity

def compute_usage_change(vocabulary_frequency,
                         alpha=1,
                         min_cf=None,
                         min_df=None):
    """

    """
    ## Compute Probability
    vocabulary_probability = vocabulary_frequency.apply(lambda row: (row + 1) / (row + 1).sum(), axis=0)
    ## Filter
    min_cf = 1 if min_cf is None else min_cf
    min_df = 1 if min_df is None else min_df
    mask = (vocabulary_frequency[["cf_start","cf_end"]]>min_cf).all(axis=1) & (vocabulary_frequency[["df_start","df_end"]]>min_df).all(axis=1)
    mask = set(mask.loc[mask].index)
    ## Apply Filter
    vocabulary_frequency = vocabulary_frequency.loc[vocabulary_frequency.index.isin(mask)].copy()
    vocabulary_probability = vocabulary_probability.loc[vocabulary_probability.index.isin(mask)].copy()
    vocabulary_probability["cf_ratio"] = np.log(vocabulary_probability["cf_end"] / vocabulary_probability["cf_start"])
    vocabulary_probability["df_ratio"] = np.log(vocabulary_probability["df_end"] / vocabulary_probability["df_start"])
    return vocabulary_frequency, vocabulary_probability

def quadrant(x, y):
    """
    
    """
    if x >= 0 and y >= 0:
        return 1
    elif x >= 0 and y < 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    else:
        return 4

def visualize_shift(vocabulary_frequency,
                    vocabulary_change,
                    weights,
                    similarity,
                    min_support=0.1):
    """

    """
    ## Copy the Vocabulary Statistics
    vocabulary_frequency = vocabulary_frequency.copy()
    vocabulary_change = vocabulary_change.copy()
    ## Aggregate
    weights_agg = weights.groupby(["feature"]).agg({"weight":[np.mean,np.std,len]})["weight"]
    similarity_agg = similarity.groupby(["feature"]).agg({"overlap":[np.mean,np.std,len]})["overlap"]
    ## Filter
    weights_agg = weights_agg.loc[weights_agg["len"] >= weights_agg["len"].max() * min_support]
    similarity_agg = similarity_agg.loc[similarity_agg["len"] >= similarity_agg["len"].max() * min_support]
    ## Isolate Intersection
    intersection = set(vocabulary_frequency.index) & set(weights_agg.index) & set(similarity_agg.index)
    weights_agg = weights_agg.loc[weights_agg.index.isin(intersection)]
    similarity_agg = similarity_agg.loc[similarity_agg.index.isin(intersection)]
    vocabulary_frequency = vocabulary_frequency.loc[vocabulary_frequency.index.isin(intersection)]
    vocabulary_change = vocabulary_change.loc[vocabulary_change.index.isin(intersection)]
    ## Merge Insights
    merged = pd.concat([weights_agg["mean"].to_frame("weight"),
                        similarity_agg["mean"].to_frame("overlap"),
                        vocabulary_change[["cf_ratio","df_ratio"]]], axis=1)
    merged.index = replace_emojis(merged.index)
    ## Quadrant    
    merged["quadrant"] = merged.apply(lambda row: quadrant(row["weight"],row["cf_ratio"]), axis=1)
    merged["weight_std"] = (merged["weight"] - merged["weight"].mean())/merged["weight"].std()
    merged["cf_std"] = (merged["cf_ratio"] - merged["cf_ratio"].mean())/merged["cf_ratio"].std()
    merged["outlier_score"] = np.sqrt(merged["weight_std"] ** 2 + merged["cf_std"] ** 2)
    merged["overlap_rank"] = merged["overlap"].rank(pct=True)
    ## Get Outliers
    all_outliers = []
    for q in range(1, 5):
        outliers = merged.loc[merged["quadrant"]==q]["outlier_score"].nlargest(25).index.tolist()
        all_outliers.extend(outliers)
    all_outliers = merged.loc[all_outliers].copy()
    all_outliers["weight_norm"] = all_outliers["weight"].rank(pct=True)
    all_outliers["cf_norm"] = all_outliers["cf_ratio"].rank(pct=True)
    ## Plot
    fig, axes = plt.subplots(1, 2, figsize=(12,4.6))
    ax, ax_out = axes
    mm = ax.scatter(merged["weight"].values,
                    merged["cf_ratio"].values,
                    alpha=0.2,
                    c=merged["overlap_rank"].values,
                    cmap=plt.cm.coolwarm,
                    vmin=0,
                    vmax=1)
    mm = ax.scatter(all_outliers["weight"].values,
                    all_outliers["cf_ratio"].values,
                    alpha=0.9,
                    c=all_outliers["overlap_rank"].values,
                    cmap=plt.cm.coolwarm,
                    vmin=0,
                    vmax=1,
                    edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", alpha=0.4)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    mm = ax_out.scatter(all_outliers["weight_norm"].values,
                        all_outliers["cf_norm"].values,
                        alpha=0.8,
                        c=all_outliers["overlap_rank"],
                        cmap=plt.cm.coolwarm,
                        vmin=0,
                        vmax=1)
    ax_out.axvline(0.5, color="black", linestyle="--", alpha=0.4)
    ax_out.axhline(0.5, color="black", linestyle="--", alpha=0.4)
    txts = []
    for out in all_outliers.index:
        txts.append(ax_out.text(all_outliers.loc[out,"weight_norm"],
                                all_outliers.loc[out,"cf_norm"],
                                out,
                                fontsize=4,
                                ha="left",
                                va="center"))
    for ii, a in enumerate([ax, ax_out]):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        if ii != 0:
            a.set_xticklabels([])
            a.set_yticklabels([])
        else:
            a.tick_params(labelsize=14)
    ax.set_title("Predictive Influence vs. Frequency Change", fontweight="bold", loc="left")
    ax_out.set_title("Outliers by Quadrant", fontweight="bold", loc="left")
    fig.text(0.5, 0.03, "Coefficient Weight", fontweight="bold", fontsize=16, va="center", ha="center")
    ax.set_ylabel("Log Ratio\n(Frequency Change)", fontweight="bold", fontsize=16)
    cbar = fig.colorbar(mm, ax=ax_out)
    cbar.set_label("Semantic Overlap", fontweight="bold", fontsize=16, labelpad=15)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.125)
    return fig

def merge_data(vocabulary_change,
               weights,
               similarity,
               min_support=0.8):
    """

    """
    ## Aggregate
    weights_agg = weights.groupby(["feature"]).agg({"weight":[np.mean, len]})["weight"]
    similarity_agg = similarity.groupby(["feature"]).agg({"overlap":[np.mean,len]})["overlap"]
    ## Filter
    weights_agg = weights_agg.loc[weights_agg["len"] >= weights_agg["len"].max() * min_support]
    similarity_agg = similarity_agg.loc[similarity_agg["len"] >= similarity_agg["len"].max() * min_support]
    intersection = set(weights_agg.index) & set(similarity_agg.index) & set(vocabulary_change.index)
    ## Merge
    merged = pd.concat([
        weights_agg["mean"].loc[intersection].to_frame("weight"),
        similarity_agg["mean"].loc[intersection].to_frame("overlap"),
        vocabulary_change.loc[intersection]
    ], axis=1)
    ## Add Absolutes
    merged["weight_absolute"] = np.abs(merged["weight"])
    merged["cf_ratio_absolute"] = np.abs(merged["cf_ratio"])
    merged["df_ratio_absolute"] = np.abs(merged["df_ratio"])
    return merged

def plot_bar_scatter(merged,
                     primary_metric="cf_ratio",
                     secondary_metric="overlap",
                     ktop=50,
                     kbot=0,
                     primary_label=None,
                     secondary_label=None):
    """

    """
    ## Get Terms to Plot
    terms = merged[primary_metric].nlargest(ktop).index.tolist() + \
            merged[primary_metric].nsmallest(kbot).index.tolist()
    terms = list(set(terms))
    terms_merged = merged.loc[terms].sort_values(primary_metric, ascending=True)
    ## Make Plot
    fig, ax = plt.subplots(1, 2, figsize=(10,5.8))
    ax[1].scatter(merged[primary_metric],
                  merged[secondary_metric],
                  alpha=0.01,
                  color="navy")
    ax[1].scatter(terms_merged[primary_metric],
                  terms_merged[secondary_metric],
                  alpha=0.4,
                  color="navy",
                  edgecolor="black")
    ax[0].barh(np.arange(terms_merged.shape[0]),
               terms_merged[primary_metric].values,
               color="black",
               height=0.1,
               alpha=0.2,
               zorder=-1)
    mm = ax[0].scatter(terms_merged[primary_metric].values,
                       np.arange(terms_merged.shape[0]),
                       c=terms_merged[secondary_metric].values,
                       alpha=0.7,
                       cmap=plt.cm.coolwarm,
                       vmin=merged[secondary_metric].min(),
                       vmax=merged[secondary_metric].max())
    if ax[0].get_xlim()[0] < 0 and ax[0].get_ylim()[1] > 0:
        ax[0].axvline(0, color="black", alpha=0.2, zorder=-1)
    cbar = fig.colorbar(mm, ax=ax[0])
    ax[0].set_yticks(np.arange(terms_merged.shape[0]))
    ax[0].set_yticklabels(replace_emojis(terms_merged.index.tolist()))
    ax[0].set_ylim(-.5, len(terms)-.5)
    ax[0].tick_params(axis="y", labelsize=8)
    for a in ax:
        a.spines["right"].set_visible(False)
        a.spines["top"].set_visible(False)
    ax[0].set_xlabel(primary_metric.replace("_"," ").title() if primary_label is None else primary_label, fontweight="bold")
    ax[1].set_xlabel(primary_metric.replace("_"," ").title() if primary_label is None else primary_label, fontweight="bold")
    ax[1].set_ylabel(secondary_metric.replace("_", " ").title() if secondary_label is None else secondary_label, fontweight="bold")
    fig.tight_layout()
    return fig

def assign_pbin(x,
                nbins=25):
    """

    """
    pbins = np.percentile(x, np.linspace(0, 100, nbins+1))
    xprime = np.array(pd.cut(x, pbins, include_lowest=True, labels=list(range(nbins))))
    return xprime, pbins

def bootstrap_ci(x,
                 alpha=0.05,
                 n_samples=100,
                 aggfunc=np.nanmean,
                 random_state=42):
    """

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

    """
    vals = {}
    for i, name in enumerate(["lower","median","upper"]):
        vals[name] = series.map(lambda j: j[i])
    vals = pd.DataFrame(vals)
    return vals

def _get_summary(merged,
                 n_sim_bins=10,
                 n_weight_bins=10):
    """

    """
    ## Summarize Similarity
    similarity_bins = np.linspace(0, 1, 101)
    similarity_summary, _ = np.histogram(merged["overlap"].values, bins=similarity_bins)
    similarity_summary = similarity_summary / similarity_summary.sum()
    similarity_summary = np.vstack([similarity_bins[:-1], similarity_summary])
    ## Group by Semantic Similarity and Get Weight
    assignments, pbins = assign_pbin(merged["overlap"], nbins=n_sim_bins)
    sim_weight_summary = bootstrap_tuple_to_df(pd.DataFrame(np.vstack([merged["weight"].values, assignments]).T).groupby(1)[0].apply(bootstrap_ci))
    sim_weight_summary["similarity_midpoint"] = (pbins[:-1] + pbins[1:]) / 2
    sim_weight_summary["similarity_bin"] = list(zip(pbins[:-1], pbins[1:]))
    sim_weight_summary.index = np.linspace(0, 100, n_sim_bins+1)[:-1]
    ## Group by Semantic Similarity and Get Weight Absolute Weight
    abs_sim_weight_summary = bootstrap_tuple_to_df(pd.DataFrame(np.vstack([merged["weight_absolute"], assignments]).T).groupby(1)[0].apply(bootstrap_ci))
    abs_sim_weight_summary["similarity_midpoint"] = (pbins[:-1] + pbins[1:]) / 2
    abs_sim_weight_summary["similarity_bin"] = list(zip(pbins[:-1], pbins[1:]))
    abs_sim_weight_summary.index = np.linspace(0, 100, n_sim_bins+1)[:-1]
    ## Group by Weight and Compute Similarity
    assignments, pbins = assign_pbin(merged["weight"].values, nbins=n_weight_bins)
    weight_sim_summary = bootstrap_tuple_to_df(pd.DataFrame(np.vstack([merged["overlap"].values, assignments]).T).groupby(1)[0].apply(bootstrap_ci))
    weight_sim_summary["weight_midpoint"] = (pbins[:-1] + pbins[1:]) / 2
    weight_sim_summary["weight_bin"] = list(zip(pbins[:-1], pbins[1:]))
    weight_sim_summary.index = np.linspace(0, 100, n_weight_bins+1)[:-1]
    ## Group Absolute Weight and Compute Similarity
    assignments_abs, pbins_abs = assign_pbin(merged["weight_absolute"].values, nbins=n_weight_bins)
    abs_weight_sim_summary = bootstrap_tuple_to_df(pd.DataFrame(np.vstack([merged["overlap"].values, assignments_abs]).T).groupby(1)[0].apply(bootstrap_ci))
    abs_weight_sim_summary["weight_midpoint"] = (pbins_abs[:-1] + pbins_abs[1:]) / 2
    abs_weight_sim_summary["weight_bin"] = list(zip(pbins_abs[:-1], pbins_abs[1:]))
    abs_weight_sim_summary.index = np.linspace(0, 100, n_weight_bins+1)[:-1]
    return similarity_summary, sim_weight_summary, abs_sim_weight_summary, weight_sim_summary, abs_weight_sim_summary

def _set_bounds(x, y, xp, yp, ax):
    """

    """
    xstep = (np.max(x) - np.min(x)) * 0.025
    ystep = (np.max(y) - np.min(y)) * 0.025
    xlim = np.percentile(x, [xp, 100-xp])
    ylim = np.percentile(y, [yp, 100-yp])
    ax.set_xlim(xlim[0]-xstep, xlim[1]+xstep)
    ax.set_ylim(ylim[0]-ystep, ylim[1]+ystep)
    return ax

def plot_percentile_summary(merged,
                            n_sim_bins=20,
                            n_weight_bins=20):
    """

    """
    ## Compute Summary
    sim_sum, sim_weight, sim_abs_weight, weight_sim, abs_weight_sim = _get_summary(merged, n_sim_bins=n_sim_bins, n_weight_bins=n_weight_bins)
    ## Make Plot
    fig, ax = plt.subplots(1, 5, figsize=(15, 4.6))
    ax[0].plot(sim_sum[0],
               sim_sum[1],
               color="black",
               alpha=0.7,
               marker="o")
    ax[1].scatter(merged["overlap"].values,
                  merged["weight"].values,
                  alpha=.05,
                  color="navy",
                  zorder=-1,
                  s=2.5)
    ax[1].errorbar(sim_weight["similarity_midpoint"],
                   sim_weight["median"].values,
                   yerr=np.vstack([(sim_weight["median"]-sim_weight["lower"]).values,
                                   (sim_weight["upper"]-sim_weight["median"]).values]),
                   color="black",
                   marker="o",
                   markersize=2.5,
                   linewidth=2,
                   alpha=0.6,
                   elinewidth=2,
                   capsize=2)
    ax[1] = _set_bounds(sim_weight["similarity_midpoint"].values, merged["weight"].values, 0, 5, ax=ax[1])
    ax[2].scatter(merged["overlap"].values,
                  merged["weight_absolute"].values,
                  alpha=.05,
                  color="navy",
                  zorder=-1,
                  s=2.5)
    ax[2].errorbar(sim_abs_weight["similarity_midpoint"],
                   sim_abs_weight["median"].values,
                   yerr=np.vstack([(sim_abs_weight["median"]-sim_abs_weight["lower"]).values,
                                   (sim_abs_weight["upper"]-sim_abs_weight["median"]).values]),
                   color="black",
                   marker="o",
                   markersize=2.5,
                   linewidth=2,
                   alpha=0.6,
                   elinewidth=2,
                   capsize=2)
    ax[2] = _set_bounds(sim_abs_weight["similarity_midpoint"].values, merged["weight_absolute"].values, 0, 5, ax=ax[2])
    ax[2].set_ylim(bottom=0)
    ax[3].scatter(merged["weight"].values,
                  merged["overlap"].values,
                  alpha=.05,
                  color="navy",
                  zorder=-1,
                  s=2.5)
    ax[3].errorbar(weight_sim["weight_midpoint"],
                   weight_sim["median"].values,
                   yerr=np.vstack([(weight_sim["median"]-weight_sim["lower"]).values,
                                   (weight_sim["upper"]-weight_sim["median"]).values]),
                   color="black",
                   marker="o",
                   markersize=2.5,
                   linewidth=2,
                   alpha=0.6,
                   elinewidth=2,
                   capsize=2)
    ax[3] = _set_bounds(weight_sim["weight_midpoint"].values, merged["overlap"].values, 0, 5, ax=ax[3])
    ax[4].scatter(merged["weight_absolute"].values,
                  merged["overlap"].values,
                  alpha=.05,
                  color="navy",
                  zorder=-1,
                  s=2.5)
    ax[4].errorbar(abs_weight_sim["weight_midpoint"],
                   abs_weight_sim["median"].values,
                   yerr=np.vstack([(abs_weight_sim["median"]-abs_weight_sim["lower"]).values,
                                   (abs_weight_sim["upper"]-abs_weight_sim["median"]).values]),
                   color="black",
                   marker="o",
                   markersize=2.5,
                   linewidth=2,
                   alpha=0.6,
                   elinewidth=2,
                   capsize=2)
    ax[4] = _set_bounds(abs_weight_sim["weight_midpoint"].values, merged["overlap"].values, 0, 5, ax=ax[4])
    ax[4].set_xlim(left=0)
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(labelsize=12)
    ax[0].set_xlabel("Similarity", fontweight="bold"); ax[0].set_ylabel("Frequency", fontweight="bold")
    ax[1].set_xlabel("Similarity", fontweight="bold"); ax[1].set_ylabel("Weight", fontweight="bold")
    ax[2].set_xlabel("Similarity", fontweight="bold"); ax[2].set_ylabel("Absolute Weight", fontweight="bold")
    ax[3].set_xlabel("Weight", fontweight="bold"); ax[3].set_ylabel("Similarity", fontweight="bold")
    ax[4].set_xlabel("Absolute Weight", fontweight="bold"); ax[4].set_ylabel("Similarity", fontweight="bold")
    fig.tight_layout()
    return fig

def main():
    """

    """
    ## Parse Command Line
    args = parse_command_line()
    ## Load Frequency of Usage by Time Period
    LOGGER.info("[Loading Vocabulary Usage]")
    vocabulary_frequency = load_usage(vector_dir=args.vector_dir,
                                      jobs=args.jobs)
    ## Compute Shift (and Filter)
    vocabulary_frequency, vocabulary_change = compute_usage_change(vocabulary_frequency,
                                                                   alpha=1,
                                                                   min_cf=args.min_cf,
                                                                   min_df=args.min_df)
    ## Load Model Information
    LOGGER.info("[Loading Model Information]")
    scores, weights, similarity = load_model_resources(models_dir=args.models_dir)
    ## Merged DataFrame
    merged = merge_data(vocabulary_change=vocabulary_change,
                        weights=weights,
                        similarity=similarity,
                        min_support=0.8)
    ## Visualize Model Performance
    LOGGER.info("[Visualizing Model Performance]")
    for metric in ["accuracy","f1","auc"]:
        fig = plot_performance_summary(scores, metric)
        _ = fig.savefig(f"{args.models_dir}/scores.summary.evaluation.{metric}.png", dpi=300)
        plt.close(fig)
    ## Visualize Model vs. Data
    LOGGER.info("[Visualizing Model Coefficient vs. Data Shift]")
    fig = visualize_shift(vocabulary_frequency,
                          vocabulary_change,
                          weights,
                          similarity,
                          min_support=0.8)
    fig.savefig(f"{args.models_dir}/shift.data.png", dpi=300)
    plt.close(fig)
    ## Visualize Model vs. Semantic Shift
    LOGGER.info("[Visualizing Model Coefficient vs. Semantic Shift]")
    fig = plot_shift_influence_summary(baseline_classification_weights=weights,
                                       shift_csv=similarity.groupby("feature").mean(),
                                       min_shift_freq_source=50,
                                       min_shift_freq_target=50,
                                       enforce_target=False,
                                       dropna=True,
                                       npercentile=25)
    fig.savefig(f"{args.models_dir}/shift.semantics.png", dpi=300)
    plt.close(fig)
    ## Plotting Summary
    LOGGER.info("[Visualizing Percentile Groups]")
    fig = plot_percentile_summary(merged, n_weight_bins=10, n_sim_bins=10)
    fig.savefig(f"{args.models_dir}/scatter.summary.png", dpi=300)
    plt.close(fig)
    ## Bar/Scatter
    LOGGER.info("[Visualizing Comparisons]")
    comparisons = [
        ("weight","overlap",25,25),
        ("weight_absolute","overlap",50,0),
        ("cf_ratio","overlap",25,25),
        ("cf_ratio_absolute","overlap",50,0),
        ("cf_ratio","weight",25,25),
        ("cf_ratio_absolute","weight",50,0),
        ("weight","cf_ratio",25,25),
        ("weight_absolute","cf_ratio",50,0)
    ]
    for primary, secondary, topk, botk in comparisons:
        fig = plot_bar_scatter(merged,
                               primary_metric=primary,
                               secondary_metric=secondary,
                               ktop=topk,
                               kbot=botk,
                               primary_label=LBL_MAP.get(primary, primary),
                               secondary_label=LBL_MAP.get(secondary, secondary))
        fig.savefig(f"{args.models_dir}/comparison.{primary}.{secondary}.png", dpi=300)
        plt.close(fig)
    LOGGER.info("[Script Complete]")

#################
### Execute
#################

if __name__ == "__main__":
    _ = main()