
"""

"""

#####################
### Configuration
#####################

## Outputs
OUTPUT_DIR = "./plots/v1/"

## Keyword Parameters
AGG_HASHTAGS = True
DELTA_BOUNDARIES = [("2019-03-01","2019-07-01"),("2020-03-01","2020-07-01")]

## Semantic Shift Parameters
MIN_FREQ_A = 1
MIN_FREQ_B = 1

#####################
### Imports
#####################

## Standard Library
import os
import sys
from glob import glob
from datetime import datetime

## External Libraries
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from adjustText import adjust_text

## Project Specific
from semshift.util.logging import initialize_logger

## Non-Library
_ = sys.path.append(os.path.dirname(__file__))
from helpers import replace_emojis

#####################
### Globals
#####################

## Logging
LOGGER = initialize_logger()

#####################
### Helpers
#####################

def load_txt(filename, func=None):
    """

    """
    data = []
    with open(filename,"r") as the_file:
        for l in the_file:
            data.append(l.strip())
    if func is not None:
        data = [func(d) for d in data]
    return data

def load_post_counts(directory):
    """

    """
    ## Load Data
    counts = sparse.load_npz(f"{directory}/counts.npz")
    time_bins = load_txt(f"{directory}/time_bins.txt", func=lambda x: list(map(int, x.split())))
    users = load_txt(f"{directory}/users.txt", func=None)
    return counts, time_bins, users

def add_months(date, n):
    """
    
    """
    end_month = date[1] + n
    end_year = date[0]
    if end_month > 12:
        end_month = end_month - 12
        end_year += (n // 12) + 1
    return [end_year, end_month, date[2]]
    
def get_monthly_windows(date_boundaries,
                        window_size=4,
                        slide_size=1):
    """

    """
    ## Starting Point
    lower = list(map(int,date_boundaries[0].split("-")))
    upper = add_months(lower, window_size)
    ## Stopping Point
    stop = datetime(*list(map(int,date_boundaries[1].split("-"))))
    ## Initialize Window Cache
    windows = [(lower, upper)]
    while datetime(*upper) < stop:
        lower = add_months(lower, slide_size)
        upper = add_months(upper, slide_size)
        windows.append((lower, upper))
    ## Format
    windows = [(datetime(*i), min(datetime(*j), stop)) for i, j in windows]
    return windows

def load_keyword_tracking(platform,
                          keyword_map):
    """
    
    """
    ## Path to Tracking Results
    if platform == "twitter":
        tracking_dir = "./data/results/tracker/gardenhose/"
    elif platform == "reddit":
        tracking_dir = "./data/results/tracker/reddit-active/"
    else:
        raise ValueError("Platform Not Recognized.")
    ## Load Data
    matches = pd.read_csv(f"{tracking_dir}/match_counts.csv",index_col=0)
    n_posts = pd.read_csv(f"{tracking_dir}n_posts.csv",index_col=0)
    ## Format Data
    matches.index = pd.to_datetime(matches.index)
    n_posts.index = pd.to_datetime(n_posts.index)
    ## Filter by Dates
    matches = matches.loc[matches.index >= pd.to_datetime("2019-01-01")].copy()
    n_posts = n_posts.loc[n_posts.index >= pd.to_datetime("2019-01-01")].copy()
    ## Hashtag Aggregation
    if AGG_HASHTAGS:
        col_aggs = {}
        for col in matches.columns:
            col_s = col.lstrip("#")
            if col_s not in col_aggs:
                col_aggs[col_s] = []
            col_aggs[col_s].append(col)
        matches = pd.concat([matches[cols].sum(axis=1).to_frame(col) for col, cols in col_aggs.items()],axis=1)
        matches.index = pd.to_datetime(matches.index)
    ## Normalization (e.g. Matches per Post)
    matches_normed = pd.concat([(matches[col]/n_posts["n_posts"]).to_frame(col) for col in matches.columns],axis=1)
    ## Match Rate Delta
    delta_boundaries_dt = [pd.to_datetime(db) for db in DELTA_BOUNDARIES]
    matches_delta = [matches[(matches.index >= dt[0])&(matches.index<dt[1])].sum(axis=0) for dt in delta_boundaries_dt]
    n_posts_delta = [n_posts[(n_posts.index >= dt[0])&(n_posts.index<dt[1])].sum(axis=0) for dt in delta_boundaries_dt]
    matches_normed_delta = pd.concat([(m / n["n_posts"]).to_frame(col) for m, n, col in zip(matches_delta, n_posts_delta, ["A","B"])],axis=1)
    matches_normed_delta["change"] = matches_normed_delta["B"] - matches_normed_delta["A"]
    matches_normed_delta["change_rel"] = matches_normed_delta["change"] / matches_normed_delta["A"]
    ## Filter Out Anomalies
    matches_normed_delta = matches_normed_delta.loc[(matches_normed_delta["A"]>0)&(matches_normed_delta["B"]>0)].copy()
    ## Add List Identifiers
    for klist, klist_terms in keyword_map.items():
        matches_normed_delta[klist] = matches_normed_delta.index.isin(klist_terms)
    return matches, matches_normed, matches_normed_delta

def load_overlap_scores(platform,
                        matches_normed_delta):
    """
    
    """
    ## Appropriate File
    if platform == "twitter":
        score_file = "data/results/model/vocabularies/twitter/gardenhose/scores.csv"
    elif platform == "reddit":
        score_file = "data/results/model/vocabularies/reddit/active/scores.csv"
    else:
        raise ValueError("Platform Not Recognized.")
    ## Load Scores and Filter
    overlap_scores_df = pd.read_csv(score_file,index_col=0)
    overlap_scores_df = overlap_scores_df.loc[(overlap_scores_df["freq_a"]>=MIN_FREQ_A)&
                                            (overlap_scores_df["freq_b"]>=MIN_FREQ_B)]
    overlap_scores_df = overlap_scores_df.loc[~overlap_scores_df.index.isnull()].copy()
    ## Measure Frequency and Change
    overlap_scores_df["freq_a_norm"] = overlap_scores_df["freq_a"] / overlap_scores_df["freq_a"].sum()
    overlap_scores_df["freq_b_norm"] = overlap_scores_df["freq_b"] / overlap_scores_df["freq_b"].sum()
    overlap_scores_df["freq_change"] = overlap_scores_df["freq_b_norm"] - overlap_scores_df["freq_a_norm"]
    overlap_scores_df["freq_log_ratio"] = np.log(overlap_scores_df["freq_b_norm"] / overlap_scores_df["freq_a_norm"])
    ## Isolate Mental Health Keyword Subset
    keyword_subset = overlap_scores_df.loc[overlap_scores_df.index.isin(matches_normed_delta.index)] 
    non_keyword_subset = overlap_scores_df.loc[~overlap_scores_df.index.isin(keyword_subset.index)]
    return overlap_scores_df, keyword_subset, non_keyword_subset

def get_users_available_by_threshold(activity_dir,
                                     time_periods,
                                     thresholds = [10, 25, 50, 75, 100, 150, 200],
                                     window_size=4,
                                     slide_size=1):
    """
    
    """
    ## Get Dataset and Time Periods
    dir_to_dataset = lambda path: path.split("/post-temporal-grouped/")[0].split("/")[-1]
    dir_dataset = dir_to_dataset(activity_dir)
    date_boundaries = time_periods.get(dir_dataset)
    ## Load Count Data
    counts, time_bins, _ = load_post_counts(activity_dir)
    ## Get Windows 
    windows = get_monthly_windows(date_boundaries, window_size, slide_size)
    time_bins_dt = np.array([[datetime.utcfromtimestamp(i), datetime.utcfromtimestamp(j)] for i, j in time_bins])
    windows_inds = [np.logical_and(time_bins_dt[:,0] >= start, time_bins_dt[:,0] < end).nonzero()[0] for start, end in windows]
    ## Aggregate Counts Within Windows
    windows_counts = np.vstack([counts[:,inds].sum(axis=1).A.T[0] for inds in windows_inds])
    ## Get Users Per Threshold
    threshold_counts = np.zeros((len(thresholds), windows_counts.shape[0]), dtype=int)
    for t, threshold in enumerate(thresholds):
        threshold_counts[t] = (windows_counts >= threshold).sum(axis=1)
    threshold_counts = pd.DataFrame(threshold_counts,
                                    index=thresholds,
                                    columns=["{} : {}".format(w[0].date().isoformat(),w[1].date().isoformat()) for w in windows]).T
    return threshold_counts

def bootstrap_ci(x,
                 alpha=0.05,
                 n_samples=100,
                 aggfunc=np.nanmean):
    """
    
    """
    n = len(x)
    q = aggfunc(x)
    q_cache = np.zeros(n_samples)
    for sample in range(n_samples):
        x_sample = np.random.choice(x, n, replace=True)
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

#####################
### General Overhead
#####################

## Establish Overhead
if not os.path.exists(OUTPUT_DIR):
    _ = os.makedirs(OUTPUT_DIR)

## Load Keywords
keywords = {}
for keyword_list in ["crisis_level1","crisis_level2","crisis_level3","pmi"]:
    keywords[keyword_list] = set()
    with open(f"./data/resources/keywords/{keyword_list}.keywords","r") as the_file:
        for line in the_file:
            keywords[keyword_list].add(line.strip())

## Naming and Order
dataset_names = {
    "clpsych-gardenhose":"CLPsych",
    "multitask-gardenhose":"Multitask",
    "wolohan-active":"Topic-Restricted Text",
    "smhd-active":"SMHD",
    "gardenhose-filtered-semantic_vocabulary":"Twitter",
    "active-semantic_vocabulary":"Reddit"
}
dataset_order = {x:i for i, x in enumerate(dataset_names)}

#####################
### Keyword Tracking
#####################

## Load Match Data
twitter_matches, twitter_matches_normed, twitter_matches_normed_delta = load_keyword_tracking("twitter", keywords)
reddit_matches, reddit_matches_normed, reddit_matches_normed_delta = load_keyword_tracking("reddit", keywords)

## Visualization
isolate_terms = [
                 "panic",
                 "suicide",
                 "eviction",
                 "vulnerable",
                 "isolated",
]
fig, ax = plt.subplots(len(isolate_terms), 1, figsize=(5, 3.5), sharex=True, sharey=False)
for platform, matches_normed, color in zip(["Twitter","Reddit"],
                                           [twitter_matches_normed, reddit_matches_normed],
                                           ["navy","darkred"]):
    for t, term in enumerate(isolate_terms):
        _ = ax[t].plot(matches_normed.index,
                       matches_normed[term] / matches_normed[term].max(),
                       color=color,
                       linewidth=1.5,
                       alpha=0.7,
                       label=platform)
        ax[t].spines["right"].set_visible(False)
        ax[t].spines["top"].set_visible(False)
        ax[t].tick_params(labelsize=14)
        ax[t].set_ylim(bottom=0)
        ax[t].set_xlim(matches_normed.index.min(), matches_normed.index.max())
        ax[t].set_yticks([])
        ax[t].set_xticks(pd.to_datetime(["2019-03-01","2020-03-01"]))
        ax[t].set_xticklabels(["March\n2019","March\n2020"], fontweight="bold")
for t, term in enumerate(isolate_terms):
    ax[t].text(pd.to_datetime("2019-01-15"), .85, f'"{term}"', fontstyle="italic", fontsize=12)
leg = ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=12)
fig.text(0.05, 0.6, "Match Rate", fontweight="bold", fontsize=16, ha="center", va="center", rotation=90)
fig.tight_layout()
fig.subplots_adjust(hspace=0.5, left=0.1, top=0.88)
fig.savefig(f"{OUTPUT_DIR}/keyword_timeseries.png",dpi=300)
fig.savefig(f"{OUTPUT_DIR}/keyword_timeseries.pdf",dpi=300)
plt.close(fig)

## Change Visualization (Scatter Plot)
fig, axes = plt.subplots(1, 2, figsize=(10,3.5))
for p, (platform, matches_normed_delta) in enumerate(zip(["Twitter","Reddit"],
                                                         [twitter_matches_normed_delta, reddit_matches_normed_delta])):
    largest_changes = matches_normed_delta["change"].map(abs).nlargest(10).index.tolist()
    plot_data = matches_normed_delta[["A","B"]] * 1e3
    ax = axes[p]
    ax.scatter(plot_data["A"],
            plot_data["B"],
            color="navy",
            alpha=0.3,
            s=50)
    ax.plot([0, 1000],
            [0, 1000],
            color="black",
            alpha=0.2,
            linestyle="--")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylim(bottom=0, top=plot_data["B"].max() * 1.05)
    ax.set_xlim(left=0, right=plot_data["A"].max() * 1.05)
    ax.set_xlabel("2019 Rate", fontweight="bold", fontsize=16)
    if p == 0:
        ax.set_ylabel("2020 Rate", fontweight="bold", fontsize=16)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(list(map(lambda x: "{}".format(int(x)) + "$e^{-3}$" if x > 0 else "0", ax.get_xticks())))
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(list(map(lambda x: "{}".format(int(x)) + "$e^{-3}$" if x > 0 else "0", ax.get_yticks())))
    ax.tick_params(labelsize=12)
    txts = []
    for term in largest_changes:
        t = ax.text(plot_data.loc[term,"A"],
                    plot_data.loc[term,"B"],
                    term,
                    fontsize=8,
                    bbox={"facecolor":"white","boxstyle":"round","fc":"white","ec":"black"})
        txts.append(t)
    _ = adjust_text(txts,
                    add_step_numbers=False,
                    ax=ax,
                    force_text=(2,2),
                    precision=1e-3)
    ax.set_title(platform, fontsize=14, fontstyle="italic")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/keyword_delta.png",dpi=300)
fig.savefig(f"{OUTPUT_DIR}/keyword_delta.pdf",dpi=300)
plt.close(fig)

#####################
### Semantic Shift
#####################

## Load Scores
twitter_overlap_scores_df, twitter_keyword_subset, twitter_non_keyword_subset = load_overlap_scores("twitter", twitter_matches_normed_delta)
reddit_overlap_scores_df, reddit_keyword_subset, reddit_non_keyword_subset = load_overlap_scores("reddit", reddit_matches_normed_delta)

## Terms to Annotate
annot_terms = [
    "abortion",
    "opioid",
    "depressive",
    "xanax",
    "impulsive",
    "illness",
    "crisis",
    "isolated",
    "panic",
    "lethal",
    "depressed",
    "strain",
    "eviction",
    "clinic",
    "nervous",
    "lonely"
]

## Generate Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(10,3.5), sharey=True)
for p, (overlap_scores_df, non_keyword_subset, keyword_subset) in enumerate(zip(
                    [twitter_overlap_scores_df, reddit_overlap_scores_df],
                    [twitter_non_keyword_subset, reddit_non_keyword_subset],
                    [twitter_keyword_subset, reddit_keyword_subset])):
    ax = axes[p]
    p_annot_terms = list(filter(lambda a: a in overlap_scores_df.index, annot_terms))
    non_keyword_subset_plot = non_keyword_subset.sample(frac=0.2, random_state=42)
    ax.scatter(non_keyword_subset_plot["freq_log_ratio"],
            non_keyword_subset_plot["overlap"],
            alpha=0.1,
            s=25,
            color="navy")
    ax.scatter(keyword_subset["freq_log_ratio"],
            keyword_subset["overlap"],
            alpha=0.8,
            s=25,
            color="darkred",
            label="Depression\nIndicative Terms")
    txts = []
    for term in p_annot_terms:
        txts.append(ax.text(keyword_subset.loc[term]["freq_log_ratio"],
                            keyword_subset.loc[term]["overlap"],
                            term,
                            fontsize=8,
                            ha="left",
                            va="center",
                            fontweight="bold",
                            bbox={"fc":"white","ec":"black","boxstyle":"round"}))
    _ = adjust_text(txts,
                    add_step_numbers=False,
                    ax=ax,
                    force_text=(2,2),
                    expand_text=(1.5,1.5),
                    expand_objects=(1.5,1.5),
                    only_move={"text":"y"},
                    ha="center",
                    avoid_text=True,
                    avoid_points=False,
                    precision=1e-5)
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Frequency Log Ratio", fontweight="bold", fontsize=14)
    if p == 0:
        ax.set_ylabel("Semantic Similarity", fontweight="bold", fontsize=14)
    if p == 1:
        ax.set_xlim(-1.5,1.5)
    else:
        ax.set_xlim(-3, 3)
    ax.tick_params(labelsize=12)
for p, platform in enumerate(["Twitter","Reddit"]):
    axes[p].set_title(platform, fontsize=14, fontstyle="italic", loc="left")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/semantic_shift_scatter.png",dpi=300)
fig.savefig(f"{OUTPUT_DIR}/semantic_shift_scatter.pdf",dpi=300)
plt.close(fig)

## Largest Increase Bar Plot
ktop = 20
fig, axes = plt.subplots(2, 2, figsize=(10,7))
for p, (overlap_scores_df, non_keyword_subset, keyword_subset) in enumerate(zip(
                [twitter_overlap_scores_df, reddit_overlap_scores_df],
                [twitter_non_keyword_subset, reddit_non_keyword_subset],
                [twitter_keyword_subset, reddit_keyword_subset])):
    for d, df in enumerate([non_keyword_subset,keyword_subset]):
        largest_increase = df["freq_log_ratio"].nlargest(ktop).index.tolist()[::-1]
        ax = axes[p, d]
        for l, li in enumerate(largest_increase):
            m=ax.barh(l,
                    overlap_scores_df.loc[li]["freq_log_ratio"],
                    color="navy",
                    alpha=overlap_scores_df.loc[li]["overlap"],
                    linewidth=0)
            m=ax.barh(l,
                    overlap_scores_df.loc[li]["freq_log_ratio"],
                    color="none",
                    edgecolor="navy",
                    linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_yticks(list(range(len(largest_increase))))
            ax.set_yticklabels(largest_increase)
            ax.tick_params(axis="x",labelsize=12)
            ax.tick_params(axis="y",labelsize=8)
            ax.set_ylim(-.5, len(largest_increase)-.5)
            if p == 1:
                ax.set_xlabel("Match Rate Increase\n(Log Ratio)", fontweight="bold", fontsize=14)
axes[0,0].set_title("General Terms", fontsize=14, fontweight="bold", fontstyle="italic", loc="left")
axes[0,1].set_title("Depression Indicative Terms", fontsize=14, fontweight="bold", fontstyle="italic", loc="left")
fig.text(0.025, 0.75, "Twitter", rotation=90, ha="center", va="center", fontsize=14, fontweight="bold", fontstyle="italic")
fig.text(0.025, 0.3, "Reddit",  rotation=90, ha="center", va="center", fontsize=14, fontweight="bold", fontstyle="italic")
fig.tight_layout()
fig.subplots_adjust(wspace=0.175, left=0.14)
fig.savefig(f"{OUTPUT_DIR}/semantic_shift_barh.png",dpi=300)
fig.savefig(f"{OUTPUT_DIR}/semantic_shift_barh.pdf",dpi=300)
plt.close(fig)
