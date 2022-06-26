
"""
One-off Visualizations
"""

PLOT_DIR = "./plots/v2/"

#######################
### Imports
#######################

## Standard Library
import os
from glob import glob
from multiprocessing import Pool

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, PercentFormatter

#######################
### Globals
#######################

DATASETS = {
    "clpsych":"CLPsych",
    "multitask":"Multi-Task Learning",
    "wolohan":"Topic-Restricted Text",
    "smhd":"SMHD"
}

## Plot Directory
if not os.path.exists(PLOT_DIR):
    _ = os.makedirs(PLOT_DIR)

## Colors and Markers
COLORS = [
    "navy",
    "darkred",
    "darkgreen",
    "darkviolet",
    "darkorange",
    "darkturquoise",
    "black",
    "gray",
]
MARKERS = ["o","x","^","*","s","d",">","<"]

## Selectors
SELECTORS = {
    "cumulative":"Cumulative",
    "intersection":"Intersection",
    "frequency":"Frequency",
    "random":"Random",
    "chi2":"Chi-Squared",
    "coefficient":"Coefficient",
    "overlap":"Overlap",
    "overlap-coefficient_0.5":"Weighted"
}

#######################
### Helpers
#######################

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

######################################################
### Performance (Within) vs. Prevance Estimates
######################################################

## Directories
model_dirs = {
    "clpsych":"./data/results/estimator/train-adaptation/clpsych-gardenhose/",
    "multitask":"./data/results/estimator/train-adaptation/multitask-gardenhose/",
    "wolohan":"./data/results/estimator/train-adaptation/wolohan-active/",
    "smhd":"./data/results/estimator/train-adaptation/smhd-active/",
}
estimate_dirs = {
    "clpsych":"data/results/estimator/apply/twitter/clpsych-gardenhose/",
    "multitask":"data/results/estimator/apply/twitter/multitask-gardenhose/",
    "wolohan":"data/results/estimator/apply/reddit/wolohan-active/",
    "smhd":"data/results/estimator/apply/reddit/smhd-active/",
}

## Load Classification Scores
scores = []
for dataset, dataset_path in model_dirs.items():
    dataset_score_files = glob(f"{dataset_path}/*-*/reduced_vocabulary_classification_scores.csv")
    for sf in dataset_score_files:
        ## Read
        sf_data = pd.read_csv(sf)
        ## Isolate
        sf_data = sf_data.loc[(sf_data["data_id"]=="evaluation")&(sf_data["group"]=="test")].copy()
        ## Format
        sf_data["vocabulary_selector"] = sf_data["vocabulary_id"].map(lambda i: i.split("@")[0])
        sf_data["vocabulary_selector_percentile"] = sf_data["vocabulary_id"].map(lambda i: int(i.split("@")[1]) if "@" in i else -1)
        sf_data["sample"] = os.path.dirname(sf).split("/")[-1]
        sf_data["dataset"] = dataset
        ## Cache
        scores.append(sf_data)
scores = pd.concat(scores, axis=0, sort=False).reset_index(drop=True)

## Load Estimates
estimates = []
for dataset, dataset_path in estimate_dirs.items():
    dataset_estimate_files = glob(f"{dataset_path}/*-*/estimates.csv")
    for ef in dataset_estimate_files:
        ## Read
        ef_data = pd.read_csv(ef)
        ## Format
        ef_data["sample"] = os.path.dirname(ef).split("/")[-1]
        ef_data["dataset"] = dataset
        ## Cache
        estimates.append(ef_data)
estimates = pd.concat(estimates, axis=0, sort=False).reset_index(drop=True)

## What to Plot
metric = "f1"
measure = "threshold"
measure_value = "0.5"
within = False

## Isolate Scores
scores_summary = scores.loc[scores["vocabulary_selector"].isin(["cumulative","intersection","overlap"])]
estimates_summary = estimates.loc[estimates["vocabulary_selector"].isin(["cumulative","intersection","overlap"])]

## Make Plot
fig = plt.figure(constrained_layout=False, figsize=(8,3))
gs = GridSpec(4, 2, figure=fig)
ax_score = [fig.add_subplot(gs[i, 0]) for i in range(4)]
ax_estimate = fig.add_subplot(gs[:, 1])
for d, dataset in enumerate(DATASETS.keys()):
    ## Isolate
    d_scores = scores_summary.loc[scores_summary["dataset"] == dataset]
    d_estimates = estimates_summary.loc[(estimates_summary["dataset"] == dataset)&
                                        (estimates_summary["measure"]==measure)&
                                        (estimates_summary["measure_value"]==measure_value)&
                                        (estimates_summary["within_subject"]==within)]
    ## Aggregate
    d_scores_ci = bootstrap_tuple_to_df(d_scores.groupby(["vocabulary_selector","vocabulary_selector_percentile"])[metric].apply(bootstrap_ci))
    d_estimates_ci = bootstrap_tuple_to_df(d_estimates.groupby(["vocabulary_selector","vocabulary_selector_percentile"])["delta"].apply(bootstrap_ci))
    ## Plot
    for s, selector in enumerate(["intersection","overlap"]):
        sel_scores = d_scores_ci.loc[selector]
        sel_scoress_int = sel_scores.reindex(np.arange(sel_scores.index.min(), sel_scores.index.max()+1))
        sel_scoress_int = sel_scoress_int.interpolate("quadratic")
        sel_estimates = d_estimates_ci.loc[selector]
        sel_estimates_int = sel_estimates.reindex(np.arange(sel_estimates.index.min(), sel_estimates.index.max()+1))
        sel_estimates_int = sel_estimates_int.interpolate("quadratic")
        if selector == "overlap":
            ax_score[d].fill_between(sel_scoress_int.index + 0.5 * s,
                                     sel_scoress_int["lower"],
                                     sel_scoress_int["upper"],
                                     alpha=0.2,
                                     color=COLORS[d],
                                     linewidth=0)
            ax_score[d].errorbar(sel_scores.index + 0.5 * s,
                                 sel_scores["median"],
                                 yerr=np.array([(sel_scores["median"]-sel_scores["lower"]).values,
                                                (sel_scores["upper"]-sel_scores["median"]).values]), 
                                 label=dataset, 
                                 color=COLORS[d],
                                 linewidth=0,
                                 marker=MARKERS[d],
                                 capsize=2,
                                 elinewidth=2,
                                 alpha=0.8)
            ax_estimate.fill_between(sel_estimates_int.index + 0.5 * s,
                                     sel_estimates_int["lower"],
                                     sel_estimates_int["upper"],
                                     alpha=0.2,
                                     color=COLORS[d],
                                     linewidth=0)
            ax_estimate.errorbar(sel_estimates.index + 0.5 * s,
                                 sel_estimates["median"],
                                 yerr=np.array([(sel_estimates["median"]-sel_estimates["lower"]).values,
                                                (sel_estimates["upper"]-sel_estimates["median"]).values]), 
                                 label=DATASETS[dataset], 
                                 color=COLORS[d],
                                 linewidth=0,
                                 marker=MARKERS[d],
                                 capsize=2,
                                 elinewidth=2,
                                 alpha=0.8)
        else:
            ax_score[d].fill_between([0, 100],
                               sel_scores["lower"].values[0],
                               sel_scores["upper"].values[0],
                               alpha=0.2,
                               color=COLORS[d],
                               hatch="/")
            ax_score[d].axhline(sel_scores["median"].values[0],
                          alpha=0.8,
                          color=COLORS[d],
                          linewidth=2)
            ax_estimate.fill_between([0, 100],
                               sel_estimates["lower"].values[0],
                               sel_estimates["upper"].values[0],
                               alpha=0.2,
                               color=COLORS[d],
                               hatch="/")
            ax_estimate.axhline(sel_estimates["median"].values[0],
                                alpha=0.8,
                                color=COLORS[d],
                                linewidth=2)
ax_estimate.axhline(0, color="black", linewidth=1, alpha=0.2, zorder=-1, linestyle="--")
ax_score[-1].set_xticks(np.arange(0, 120, 20))
ax_estimate.set_xticks(np.arange(0, 120, 20))
for a in ax_score + [ax_estimate]:
    a.set_xlim(0, 100)
    a.spines["right"].set_visible(False)
    a.spines["top"].set_visible(False)
    a.tick_params(labelsize=12)
    a.xaxis.set_major_formatter(PercentFormatter())
for a in ax_score[:-1]:
    a.set_xticklabels([])
for a in ax_score:
    a.yaxis.set_major_locator(MaxNLocator(2)) 
ax_estimate.legend(ncol=4, bbox_to_anchor=(1.1, 0.975), loc="lower right", frameon=False, columnspacing=0.25, fontsize=14, handletextpad=0.05)
ax_estimate.set_ylabel("Change in\nPrevalence", fontweight="bold", fontsize=18)
fig.text(0.5, 0.05, "Vocabulary Size", fontweight="bold", fontsize=18, ha="center", va="center")
fig.text(0.025, 0.5, "{} Score".format(metric.title()), fontweight="bold", fontsize=18, ha="center", va="center", rotation=90)
fig.tight_layout()
fig.subplots_adjust(left=0.13, bottom=0.2, wspace=0.47, hspace=0.2, top=0.875)
fig.savefig(f"{PLOT_DIR}/performance-prevalence.pdf", dpi=300)
plt.close(fig)

## Dataset Specific Summaries
COLORS = COLORS[-2:] + COLORS[:-2]
fig, axes = plt.subplots(4, 2, figsize=(8,8), sharex=True)
for d, dataset in enumerate(DATASETS.keys()):
    ## Initialize Figure
    ax = axes[d]
    ## Iterate Through Selectors
    for s, selector in enumerate(SELECTORS):
        ## Get Data
        s_scores = scores.loc[(scores["dataset"]==dataset)&(scores["vocabulary_selector"]==selector)]
        s_estimates = estimates.loc[(estimates["dataset"] == dataset)&
                                    (estimates["measure"]==measure)&
                                    (estimates["measure_value"]==measure_value)&
                                    (estimates["within_subject"]==within)&
                                    (estimates["vocabulary_selector"]==selector)]
        ## Aggregate
        s_scores_ci = bootstrap_tuple_to_df(s_scores.groupby(["vocabulary_selector_percentile"])[metric].apply(bootstrap_ci))
        s_estimates_ci = bootstrap_tuple_to_df(s_estimates.groupby(["vocabulary_selector_percentile"])["delta"].apply(bootstrap_ci))
        ## Interpolate
        s_scores_ci_int = s_scores_ci.reindex(np.arange(s_scores_ci.index.min(), s_scores_ci.index.max() + 1)).interpolate("quadratic")
        s_estimates_ci_int = s_estimates_ci.reindex(np.arange(s_estimates_ci.index.min(), s_estimates_ci.index.max() + 1)).interpolate("quadratic")
        ## Plot
        if selector in ["cumulative","intersection"]:
            ax[0].fill_between([0, 100],
                                s_scores_ci["lower"].values[0],
                                s_scores_ci["upper"].values[0],
                                alpha=0.2,
                                color=COLORS[s],
                                zorder=-1)
            ax[0].axhline(s_scores_ci["median"].values[0],
                          color=COLORS[s],
                          linewidth=2,
                          label=SELECTORS[selector],
                          alpha=0.8)
            ax[1].fill_between([0, 100],
                                s_estimates_ci_int["lower"].values[0],
                                s_estimates_ci_int["upper"].values[0],
                                alpha=0.2,
                                color=COLORS[s],
                                zorder=-1)
            ax[1].axhline(s_estimates_ci_int["median"].values[0],
                          color=COLORS[s],
                          linewidth=2,
                          label=SELECTORS[selector],
                          alpha=0.8)
        else:
            ax[0].fill_between(s_scores_ci_int.index,
                               s_scores_ci_int["lower"].values,
                               s_scores_ci_int["upper"].values,
                               alpha=0.2,
                               color=COLORS[s],
                               zorder=-1)
            ax[0].errorbar(s_scores_ci.index,
                           s_scores_ci["median"].values,
                           yerr=np.array([(s_scores_ci["median"]-s_scores_ci["lower"]).values,
                                          (s_scores_ci["upper"]-s_scores_ci["median"]).values]), 
                           label=SELECTORS[selector], 
                           color=COLORS[s],
                           linewidth=0,
                           marker=MARKERS[s],
                           capsize=2,
                           elinewidth=2,
                           alpha=0.8)
            ax[1].fill_between(s_estimates_ci_int.index,
                               s_estimates_ci_int["lower"].values,
                               s_estimates_ci_int["upper"].values,
                               alpha=0.2,
                               color=COLORS[s],
                               zorder=-1)
            ax[1].errorbar(s_estimates_ci.index,
                           s_estimates_ci["median"].values,
                           yerr=np.array([(s_estimates_ci["median"]-s_estimates_ci["lower"]).values,
                                          (s_estimates_ci["upper"]-s_estimates_ci["median"]).values]), 
                           label=SELECTORS[selector], 
                           color=COLORS[s],
                           linewidth=0,
                           marker=MARKERS[s],
                           capsize=2,
                           elinewidth=2,
                           alpha=0.8)
    ax[0].set_title(DATASETS[dataset], loc="left", fontsize=14)
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(labelsize=12)
        a.xaxis.set_major_formatter(PercentFormatter())
        a.set_xlim(0,100)
        a.yaxis.set_major_locator(MaxNLocator(3)) 
    if ax[1].get_ylim()[0] < 0 and ax[1].get_ylim()[1] > 0:
        ax[1].axhline(0, color="black", linestyle="--", alpha=0.2, zorder=-1)
axes[0,0].legend(loc="lower left", ncol=4, bbox_to_anchor=(0.1,1.1), fontsize=12, frameon=False, columnspacing=0.25, handletextpad=0.25)
fig.text(0.025, 0.5, "{} Score".format(metric.title()), fontsize=18, fontweight="bold", ha="center", va="center", rotation=90)
fig.text(0.525, 0.5, "Change in Prevalence", fontsize=18, fontweight="bold", ha="center", va="center", rotation=90)
fig.text(0.5, 0.025, "Vocabulary Size", fontweight="bold", fontsize=18, ha="center", va="center")
fig.tight_layout()
fig.subplots_adjust(bottom=0.08, wspace=0.3, left=0.125, top=0.9)
fig.savefig(f"{PLOT_DIR}/performance-prevalence.all.pdf", dpi=300)
plt.close(fig)

######################################################
### Semantic Stability Scores
######################################################

## Identify Files
stability_files = glob(f"./data/results/word2vec-context/*-v3/analysis/*/result-cache/*/coefficient_stability_summary.csv")

## Helper
def summarize_stability(sf):
    """
    
    """
    ## Read File
    sf_scores = pd.read_csv(sf)
    ## Metadata
    sf_dataset = sf.split("word2vec-context/")[1].split("/")[0]
    sf_sample = int(os.path.dirname(sf).split("/")[-1])
    ## Normalize Time Periods
    sf_scores["time_period_apply"] = sf_scores["time_period_apply"] - sf_scores["time_period_apply"].min() + 1
    sf_scores["time_delta"] = sf_scores["time_period_apply"] - sf_scores["time_period_train"]
    ## Aggregate
    sf_scores_agg = sf_scores.groupby(["time_period_train","time_period_apply"]).agg({"overlap":[np.mean, np.min, np.max, np.std, len], "time_delta":max})
    sf_scores_agg["dataset"] = sf_dataset
    sf_scores_agg["sample"] = sf_sample
    return sf_scores_agg

## Compute Average Statistics
with Pool(8) as mp:
    stability_statistics = list(tqdm(mp.imap_unordered(summarize_stability, stability_files), desc="[Stability Files]", total=len(stability_files)))

## Format
stability_statistics = pd.concat(stability_statistics, axis=0, sort=False)
stability_statistics = stability_statistics.sort_values(["dataset","sample","time_period_train","time_period_apply"]).reset_index()
stability_statistics[("overlap","std_error")] = stability_statistics[("overlap","std")] / np.sqrt(stability_statistics[("overlap","len")] - 1)

## Aggregate
stability_statistics_agg = stability_statistics.groupby(["dataset","time_period_train","time_period_apply"]).agg({
    ("time_delta","max"):max,
    ("overlap","mean"):np.mean,
    ("overlap","std"):np.mean
})

## Plot
fig, ax = plt.subplots()
for d, dataset in enumerate(stability_statistics_agg.index.levels[0]):
    dstats = stability_statistics_agg.loc[dataset]
    ax.errorbar(dstats[("time_delta","max")] + np.random.uniform(0,0.5,dstats.shape[0]),
                dstats[("overlap","mean")],
                yerr=dstats[("overlap","std")],
                marker="o",
                linewidth=0,
                elinewidth=2,
                capsize=2,
                color=COLORS[d],
                label=DATASETS[dataset.split("-")[0]])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1))
fig.tight_layout()
plt.show()