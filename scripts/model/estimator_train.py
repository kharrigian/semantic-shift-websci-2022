
########################
#### Imports
########################

## Standard Library
import os
import sys
import json
import argparse
from textwrap import wrap
from copy import deepcopy
from functools import partial
from collections import Counter
from multiprocessing import Pool

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from langid import langid
import matplotlib.pyplot as plt
from sklearn import feature_selection
from sklearn.metrics.pairwise import cosine_similarity

## Private
from semshift.model import classifiers
from semshift.model.embed import Word2Vec
from semshift.model.vocab import Vocabulary
from semshift.util.helpers import flatten, chunks
from semshift.model.data_loaders import PostStream
from semshift.util.logging import initialize_logger
from semshift.model.feature_extractors import FeaturePreprocessor
from semshift.preprocess.tokenizer import STOPWORDS, CONTRACTIONS

## Local Project
_ = sys.path.append(os.path.abspath(os.path.dirname(__file__))+"/")
from helpers import get_feature_names, score_predictions, DVec, replace_emojis

########################
#### Globals
########################

## Initialize Logger
LOGGER = initialize_logger()

## Load Language Identification Model ## Needed for Multiprocessing Environment
_ = langid.load_model()

##########################
### General Helpers
##########################

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

def make_dirs(directory,
              allow_overwrite=False):
    """
    
    """
    ## Deal with Existing Directories
    if os.path.exists(directory):
        if not allow_overwrite:
            raise FileExistsError("Directory already exists. Must allow overwrites to continue. [{}]".format(directory))
        else:
            _ = os.system("rm -rf {}".format(directory))
    ## Make New Directory
    _ = os.makedirs(directory)
    return directory

##########################
### Functions
##########################

def parse_command_line():
    """
    
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("config",
                            type=str,
                            help="Path to train-adaptation configuration file.")
    _ = parser.add_argument("--output_dir",
                            type=str,
                            default=None)
    _ = parser.add_argument("--classify_enforce_frequency_filter",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--classify_enforce_classwise_frequency_filter",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--classify_enforce_min_shift_frequency_filter",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--classify_enforce_target",
                            action="store_true",
                            default=False)
    _ = parser.add_argument("--allow_overwrite",
                            default=False,
                            action="store_true")
    _ = parser.add_argument("--jobs",
                            type=int,
                            default=1)
    args = parser.parse_args()
    return args

def load_configuration(config_file,
                       output_dir=None,
                       allow_overwrite=False):
    """
    
    """
    ## Check Configuration Existence
    if not os.path.exists(config_file):
        raise FileNotFoundError("Configuration File not Found: {}".format(config_file))
    ## Load File
    with open(config_file, "r") as the_file:
        config = json.load(the_file)
    ## Check Paths
    for domain in ["source","target","evaluation"]:
        for path in ["config","splits","embeddings"]:
            if domain == "evaluation" and path == "embeddings":
                continue
            if not os.path.exists(config[domain][path]):
                raise FileNotFoundError("{} {} Not Found: {}".format(domain.title(), path.title(), config[domain][path]))
    ## Output Directory
    if output_dir is None:
        output_dir = "./data/results/longitudinal/results/explore/train-adaptation/"
    output_dir = "{}/{}".format(output_dir, config["experiment_id"]).replace("//","/")
    if os.path.exists(output_dir) and not allow_overwrite:
        raise FileExistsError("If overwriting, must include --allow_overwrite flag.")
    output_dir = make_dirs(output_dir, True)
    ## Cache Configuration
    with open(f"{output_dir}/config.training.json","w") as the_file:
        json.dump(config, the_file, indent=4, sort_keys=False)
    return config, output_dir

def load_embeddings(embeddings_directory):
    """
    
    """
    ## Load
    model = Word2Vec.load(f"{embeddings_directory}/")
    ## Parse Embeddings
    vocabulary = model.get_ordered_vocabulary()
    vectors = model.model.wv.vectors
    frequency = np.array([model.model.wv.get_vecattr(v,"count") for v in vocabulary])
    phrasers = model.phrasers
    return vocabulary, vectors, frequency, phrasers

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
    return vocabulary, vectors, frequency

def load_resources(domain_config,
                   evaluation=False):
    """
    
    """
    ## Load Configuration and Splits
    with open(domain_config["config"],"r") as the_file:
        config = json.load(the_file)
    with open(domain_config["splits"],"r") as the_file:
        splits = json.load(the_file)
    ## Ignore Embeddings for Evaluation
    if not evaluation:
        ## Load Embeddings
        vocabulary, vectors, frequency, phrasers = load_embeddings(domain_config["embeddings"])
        ## Filter Embeddings
        vocabulary, vectors, frequency = filter_embeddings(vectors=vectors,
                                                           vocabulary=vocabulary,
                                                           frequency=frequency,
                                                           min_vocab_freq=domain_config["min_vocab_freq"],
                                                           rm_top=domain_config["rm_top"],
                                                           rm_stopwords=domain_config["rm_stopwords"])
    else:
        ## Null
        vocabulary, vectors, frequency, phrasers = None, None, None, None
    ## Combine
    resources = {
        "splits":splits,
        "vocabulary":vocabulary,
        "embeddings":vectors,
        "frequency":frequency,
        "phrasers":phrasers,
        "configuration":config
    }
    return resources

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
    for index_chunk in tqdm(list(chunks(list(range(vectors.shape[0])), chunksize)), desc="[Building Neighborhood]", file=sys.stdout):
        sim = cosine_similarity(vectors[index_chunk], vectors)
        sim = np.apply_along_axis(lambda x: np.argsort(x)[-top_k-1:-1][::-1], 1, sim)
        neighbors.append(sim)
    neighbors = np.vstack(neighbors)
    ## Alignment
    globalvocab2ind = dict(zip(global_vocabulary, range(len(global_vocabulary))))
    neighbors = np.apply_along_axis(lambda x: list(map(lambda i: globalvocab2ind[vocabulary[i]], x)), 1, neighbors)
    return neighbors

def compute_overlap(*neighbors):
    """
    Compute overlap between multiple neighborhoods

    Args:
        neighbors (list of array): Each array is an ordered list of term IDs, aligned globally.
    
    Returns:
        overlap_at_k (float): Percentage of neighboring terms that overlap
        jaccard (float): Jaccard index score between all sets
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

def align_neighbors_and_frequency(neighbors,
                                  frequency,
                                  vocabulary,
                                  global_vocabulary):
    """
    
    """
    vocab2ind = dict(zip(vocabulary, range(len(vocabulary))))
    neighbors = np.vstack(list(map(lambda term: neighbors[vocab2ind[term]] if term in vocab2ind else np.ones(neighbors.shape[1], dtype=int) * -1, global_vocabulary)))
    frequency = np.array(list(map(lambda term: frequency[vocab2ind[term]] if term in vocab2ind else np.nan, global_vocabulary)))
    return neighbors, frequency

def align_and_compute_similarity(source_data,
                                 target_data,
                                 config):
    """
    
    """
    ## Global Vocabulary
    global_vocabulary = sorted(set(source_data["vocabulary"]) | set(target_data["vocabulary"]))
    ## Compute Neighbors
    source_neighbors = compute_neighborhood(vectors=source_data["embeddings"],
                                            vocabulary=source_data["vocabulary"],
                                            global_vocabulary=global_vocabulary,
                                            top_k=config["source"]["top_k_neighbors"],
                                            chunksize=500)
    target_neighbors = compute_neighborhood(vectors=target_data["embeddings"],
                                            vocabulary=target_data["vocabulary"],
                                            global_vocabulary=global_vocabulary,
                                            top_k=config["target"]["top_k_neighbors"],
                                            chunksize=500)
    ## Align Neighbors and Frequency
    source_neighbors, source_frequency = align_neighbors_and_frequency(neighbors=source_neighbors,
                                                                       frequency=source_data["frequency"],
                                                                       vocabulary=source_data["vocabulary"],
                                                                       global_vocabulary=global_vocabulary)
    target_neighbors, target_frequency = align_neighbors_and_frequency(neighbors=target_neighbors,
                                                                       frequency=target_data["frequency"],
                                                                       vocabulary=target_data["vocabulary"],
                                                                       global_vocabulary=global_vocabulary)
    ## Compute Similarity
    source_target_similarity = list(map(lambda x: compute_overlap(*x), tqdm(list(zip(source_neighbors, target_neighbors)), desc="[Computing Overlap]", file=sys.stdout)))
    ## Return
    return global_vocabulary, source_neighbors, target_neighbors, source_frequency, target_frequency, source_target_similarity

def describe_shift(global_vocabulary,
                   source_neighbors,
                   target_neighbors,
                   source_frequency,
                   target_frequency,
                   similarity,
                   min_shift_freq_source=100,
                   min_shift_freq_target=100,
                   top_k=50,
                   top_k_neighbors=20):
    """

    """

    ## Output String
    output_str = []
    ## Helper
    ind2term = lambda inds: ", ".join([global_vocabulary[ind] for ind in inds]) if np.min(inds) != -1 else None
    ## Iterate Through Combinations
    source_neighbors_fmt = [ind2term(row) for row in source_neighbors[:,:top_k_neighbors]]
    target_neighbors_fmt = [ind2term(row) for row in target_neighbors[:,:top_k_neighbors]]
    ## Format
    cd_df = pd.DataFrame({
        "feature":global_vocabulary,
        "overlap":similarity,
        "source_neighbors":source_neighbors_fmt,
        "target_neighbors":target_neighbors_fmt,
        "source_frequency":source_frequency,
        "target_frequency":target_frequency
    }).set_index("feature")
    ## Copy For Return
    cd_df_copy = cd_df.copy()
    ## Format Summary
    cd_df = cd_df.dropna(subset=["overlap"]).sort_values("overlap", ascending=False)
    cd_df = cd_df.loc[(cd_df["source_frequency"]>=min_shift_freq_source)&
                      (cd_df["target_frequency"]>=min_shift_freq_target)]
    ## Generate Output String
    output_str.append("~~~~~~~ Most Similar ~~~~~~~")
    for t, (term, term_data) in enumerate(cd_df.head(top_k).iterrows()):
        tstring = "{}) {} [Score = {}]\n\t[{}] {}\n\t[{}] {}".format(t+1, term, term_data["overlap"], "Source", "\n\t".join(wrap(term_data["source_neighbors"], 140)), "Target", "\n\t".join(wrap(term_data["target_neighbors"], 140)))
        output_str.append(tstring)
    output_str.append("..." * 20)
    output_str.append("~~~~~~~ Least Similar ~~~~~~~")
    for t, (term, term_data) in enumerate(cd_df.tail(top_k).iloc[::-1].iterrows()):
        tstring = "{}) {} [Score = {}]\n\t[{}] {}\n\t[{}] {}".format(t+1, term, term_data["overlap"], "Source", "\n\t".join(wrap(term_data["source_neighbors"], 140)), "Target", "\n\t".join(wrap(term_data["target_neighbors"], 140)))
        output_str.append(tstring)
    ## Merge String
    output_str = "\n".join(output_str)
    return output_str, cd_df_copy

def _get_token_counts(stream_input,
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

def vectorize_data(splits,
                   vocabulary,
                   phrasers,
                   config,
                   min_n_posts=None,
                   n_posts=None,
                   groups=["train","test"],
                   njobs=1,
                   random_state=42):
    """
    
    """
    ## Check Splits
    if not (len(splits) == 2 and set(splits.keys()) == set(["train","test"])):
        splits = {"train":splits, "test":{}}
    ## Check Inputs
    for g in groups:
        if g not in ["train","test"]:
            raise KeyError("Possible values in `groups` are 'train' and 'test'. Got: {}".format(g))
    ## Initialize Vectorizer
    dvec = DVec(vocabulary)
    ## Prepare Loaders
    X = []
    y = []
    users = []
    data_groups = []
    with Pool(njobs) as mp:
        for group in groups:
            ## Group Integer ID
            group_int_id = int(group == "test")
            ## Check Existence
            if len(splits[group].keys()) == 0:
                continue
            ## Initialize Stream
            group_stream = PostStream(filenames=list(splits[group].keys()),
                                      loader_kwargs=config["vocab_kwargs"],
                                      processor=None,
                                      processor_kwargs={},
                                      min_date=config["datasets"][0]["date_boundaries"]["min_date"],
                                      max_date=config["datasets"][0]["date_boundaries"]["max_date"],
                                      randomized=True,
                                      n_samples=None,
                                      mode=2,
                                      phrasers=phrasers,
                                      jobs=1,
                                      check_filename_size=False,
                                      cache_data=False)
            ## Count Tokens
            counter = partial(_get_token_counts, stream=group_stream, min_n_samples=min_n_posts, n_samples=n_posts, random_state=random_state)
            iterable = tqdm(list(enumerate(group_stream.filenames)), file=sys.stdout, desc=f"[Counting Terms ({group.title()})]", total=len(group_stream.filenames))
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
            data_groups.extend([group_int_id for _ in group_counts])
    ## Stack and Format
    X = sparse.vstack(X)
    y = np.array(y)
    data_groups = np.array(data_groups)
    return X, y, data_groups, users

def split_data(y,
               groups,
               ids,
               n_samples=1,
               resample_rate=1,
               random_state=42):
    """
    
    """
    ## Seed
    seed = np.random.RandomState(random_state)
    ## Sample Seeds for Each Iteration
    sample_random_states = seed.randint(0,1e6,n_samples)
    ## Breakdown of Data
    n = np.zeros((3,2), dtype=int)
    masks = [[None, None],
             [None, None],
             [None, None]]
    for row, (g, i) in enumerate(zip([0, 1, 1], [0, 0, 1])):
        for l in [0, 1]:
            masks[row][l] = [k for k, (_y, _g, _i) in enumerate(zip(y, groups, ids)) if _y ==l and _g == g and _i == i]
            n[row][l] = len(masks[row][l])
    assert len(y) == n.sum()
    ## Get Sample Sizes (Balance and Downsampling)
    n = n.min(axis=1,keepdims=True) * np.ones_like(n)
    n = np.floor(n * resample_rate).astype(int)
    ## Generate Splits
    splits = []
    for sn in range(n_samples):
        sample_inds = [[],[],[]]
        for row, (group, i) in enumerate(zip([0, 1, 2], [0, 0, 1])):
            row_sample_seed = np.random.RandomState(sample_random_states[sn])
            for lbl in [0, 1]:
                ## Sample
                gl_sample = list(row_sample_seed.choice(masks[row][lbl], n[row][lbl], replace=resample_rate==1))
                ## Add To Cache
                sample_inds[group].extend(gl_sample)
        sample_inds = list(map(sorted, sample_inds))
        splits.append(sample_inds)
    return splits

def compute_vocabulary_statistics(X,
                                  y,
                                  vocabulary):
    """

    """
    vocabulary_statistics = []
    for lbl in [0, 1]:
        lbl_mask = (y == lbl).nonzero()[0]
        lbl_freq = X[lbl_mask].sum(axis=0).A[0].astype(int)
        lbl_sample_freq = (X[lbl_mask] != 0).sum(axis=0).A[0].astype(int)
        vocabulary_statistics.append(pd.DataFrame(
            index=["term","frequency","sample_frequency","label","support"],
            data=[vocabulary, lbl_freq, lbl_sample_freq, np.ones_like(lbl_sample_freq) * lbl, np.ones_like(lbl_sample_freq) * len(lbl_mask)]
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
    p_1_w_s = (vocabulary_statistics["sample_frequency",1] + alpha) / (vocabulary_statistics["sample_frequency"] + alpha).sum(axis=1) # p(y=1|ws)
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
    
def fit_classifier(X_train,
                   y_train,
                   vocabulary,
                   config,
                   njobs=1,
                   random_state=None):
    """
    
    """
    ## Format Vocabulary
    vocabulary = format_vocabulary(vocabulary)
    ## Initialize Vocabulary
    vocabulary_obj = Vocabulary()
    vocabulary_obj = vocabulary_obj.assign(vocabulary)
    ## Extract Features
    preprocessor = FeaturePreprocessor(vocab=deepcopy(vocabulary_obj),
                                       feature_flags=config["feature_flags"],
                                       feature_kwargs=config["feature_kwargs"],
                                       standardize=config["feature_standardize"],
                                       min_variance=None,
                                       verbose=False)
    X_train_T = preprocessor.fit_transform(X_train)
    ## Fit Model
    model = classifiers.linear_model.LogisticRegressionCV(Cs=10,
                                                          fit_intercept=True,
                                                          cv=10,
                                                          n_jobs=njobs,
                                                          scoring=config["cv_metric_opt"],
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

def get_vocabulary_mask(X,
                        y,
                        source_frequency,
                        target_frequency,
                        config,
                        enforce_frequency_filter=False,
                        enforce_classwise_frequency_filter=False,
                        enforce_min_shift_frequency_filter=False,
                        enforce_target=False):
    """
    
    """
    ## Vocabulary Frequency Filtering (Word2Vec Training Based)
    if enforce_min_shift_frequency_filter:
        if enforce_target:
            vmask_shift = np.logical_and(np.nan_to_num(source_frequency) >= config["source"]["min_shift_freq"],
                                         np.nan_to_num(target_frequency) >= config["target"]["min_shift_freq"])
        else:
            vmask_shift = np.nan_to_num(source_frequency) >= config["source"]["min_shift_freq"]
    else:
        vmask_shift = np.array([True for _ in range(X.shape[1])])
    ## Vocabulary Frequency Filtering (Sample Based)
    if enforce_classwise_frequency_filter:
        ## Require Both Classes to Meet Criteria
        vmask_sample = np.logical_and.reduce([
            X[y==0].sum(axis=0).A[0] >= config["classifier_min_vocab_freq"],
            X[y==1].sum(axis=0).A[0] >= config["classifier_min_vocab_freq"],
            X[y==0].getnnz(axis=0) >= config["classifier_min_vocab_sample_freq"],
            X[y==1].getnnz(axis=0) >= config["classifier_min_vocab_sample_freq"]
        ])
    elif enforce_frequency_filter and not enforce_classwise_frequency_filter:
        ## Require Combined Classes to Meet Criteria
        vmask_sample = np.logical_and(X.sum(axis=0).A[0] >= config["classifier_min_vocab_freq"],
                                      X.getnnz(axis=0) >= config["classifier_min_vocab_sample_freq"])
    elif not enforce_frequency_filter and not enforce_classwise_frequency_filter:
        ## Only Require Non-Null Values
        vmask_sample = X.getnnz(axis=0) > 0
    else:
        raise ValueError("Something Went Wrong with This Logic that Shouldn't.")
    vmask_sample = np.logical_and(vmask_sample, vmask_shift).nonzero()[0]
    return vmask_sample

def run_vocabulary_selection(vocabulary,
                             source_frequency,
                             target_frequency,
                             similarity,
                             baseline_classification_weights,
                             baseline_vocabulary_scores,
                             config):
    """
    
    """
    ## Random Seed
    seed = np.random.RandomState(config["random_state"])
    ## Vocabulary Map
    vocab2ind = dict(zip(vocabulary, range(len(vocabulary))))
    term2ind = lambda terms: sorted([vocab2ind[term] for term in terms])
    ## Aggregate Weights and Scores
    baseline_classification_weights_agg = baseline_classification_weights.groupby(["feature"]).agg({"weight":[np.mean, len]})["weight"]
    baseline_vocabulary_scores_agg = baseline_vocabulary_scores.drop(["fold"],axis=1).groupby(["feature"]).mean()
    ## Support Threshold
    n_folds = len(baseline_vocabulary_scores["fold"].unique())
    support = baseline_classification_weights_agg["len"] >= config["selectors_sample_support"] * n_folds
    support = set(support.loc[support].index)
    ## Initialize Sampling DataFrame
    vocabulary_df = pd.DataFrame({
        "frequency_source":source_frequency,
        "frequency_target":target_frequency,
        "overlap":similarity,
        "feature":vocabulary,
    }).set_index("feature")
    ## Merge Data
    vocabulary_df = pd.concat([vocabulary_df, baseline_classification_weights_agg[["mean"]]],axis=1,sort=False).rename(columns={"mean":"coefficient"})
    vocabulary_df["coefficient"] = np.abs(vocabulary_df["coefficient"])
    vocabulary_df = pd.merge(vocabulary_df,
                             baseline_vocabulary_scores_agg,
                             how="left",
                             left_index=True,
                             right_index=True)
    ## Intial Filter
    vocabulary_df = vocabulary_df.loc[vocabulary_df.index.isin(support)].copy()
    ## Add Randomness
    vocabulary_df["random"] = seed.normal(size=vocabulary_df.shape[0])
    ## Baseline
    vocabularies = {
        "cumulative":term2ind(vocabulary_df.loc[vocabulary_df["frequency_source"] >= config["source"]["min_shift_freq"]].index),
        "intersection":term2ind(vocabulary_df.loc[(vocabulary_df["frequency_source"]>=config["source"]["min_shift_freq"])&(vocabulary_df["frequency_target"]>=config["target"]["min_shift_freq"])].index)
    }
    ## Reset Baseline
    vocabulary_df = vocabulary_df.loc[(vocabulary_df["frequency_source"]>=config["source"]["min_shift_freq"])&
                                      (vocabulary_df["frequency_target"]>=config["target"]["min_shift_freq"])]
    vocabulary_df = vocabulary_df.dropna()
    ## Identify Additional Selectors
    selectors = []
    for selector in config["selectors"]:
        if not selector.startswith("weighted_"):
            selectors.append((selector, (selector, ), 1))
            continue
        sel_weight = float(selector[9:].split("_")[-1])
        sel_pair = tuple("_".join(selector[9:].split("_")[:-1]).split("-"))
        selectors.append((selector[9:], sel_pair, sel_weight))
    ## Sizes
    sizes = [(p, int(np.floor(vocabulary_df.shape[0] * p / 100))) for p in config["selectors_percentiles"]]
    ## Build Vocabularies
    for prefix, selector, weight in selectors:
        ## Skip Baseline
        if selector == ("cumulative",) or selector == ("intersection", ):
            continue
        ## Get Ranks
        if len(selector) == 1:
            ranks = vocabulary_df[selector[0]].rank(method="min").astype(int)
        elif len(selector) == 2:
            ranks = weight * vocabulary_df[selector[0]].rank(method="min") + (1 - weight) * vocabulary_df[selector[1]].rank(method="min")
        else:
            raise ValueError("Only support provided for two sets of scores.")
        ## Build Vocabulary
        for percentile, size in sizes:
            vocabularies[f"{prefix}@{percentile}"] = term2ind(ranks.nlargest(size).index)
    ## Return
    return vocabularies

def measure_reduced_vocabulary_overlap(reduced_vocabularies):
    """
    
    """
    ## Group Indices
    vocabulary_percentile_groups = {}
    for vocabulary_id, vocabulary_indices in reduced_vocabularies.items():
        ## Skip Baselines
        if vocabulary_id in ["cumulative","intersection"]:
            continue
        ## Parse ID
        vocabulary_selector, vocabulary_selector_percentile = vocabulary_id.split("@")
        vocabulary_selector_percentile = int(vocabulary_selector_percentile)
        ## Initialize Cache (if Necessary)
        if vocabulary_selector_percentile not in vocabulary_percentile_groups:
            vocabulary_percentile_groups[vocabulary_selector_percentile] = {}
        ## Cache
        vocabulary_percentile_groups[vocabulary_selector_percentile][vocabulary_selector] = vocabulary_indices
    ## Measure
    vocabulary_percentile_groups_overlap = {}
    for percentile, group in vocabulary_percentile_groups.items():
        selectors = sorted(group.keys())
        selectors2ind = dict(zip(selectors, range(len(selectors))))
        selectors_overlap = np.zeros((len(selectors), len(selectors)), dtype=int)
        for s1, s1ind in group.items():
            for s2, s2ind in group.items():
                selectors_overlap[selectors2ind[s1], selectors2ind[s2]] = len(set(s1ind) & set(s2ind))
        selectors_overlap = pd.DataFrame(data=selectors_overlap, index=selectors, columns=selectors)
        vocabulary_percentile_groups_overlap[percentile] = selectors_overlap
    ## Format
    vocabulary_percentile_groups_overlap = pd.concat(vocabulary_percentile_groups_overlap).sort_index()
    return vocabulary_percentile_groups_overlap
    
def plot_shift_influence_summary(baseline_classification_weights,
                                 shift_csv,
                                 min_shift_freq_source,
                                 min_shift_freq_target,
                                 enforce_target=False,
                                 dropna=True,
                                 npercentile=25):
    """
    
    """
    ## Aggregate Weights
    baseline_classification_weights_agg = baseline_classification_weights.groupby(["feature"])["weight"].mean().to_frame("weight")
    ## Merge
    merged_df = pd.merge(shift_csv[["overlap"]], baseline_classification_weights_agg, left_index=True, right_index=True)
    ## Filter
    if enforce_target:
        freq_criteria = set(shift_csv.loc[(shift_csv["source_frequency"] >= min_shift_freq_source)&
                                          (shift_csv["target_frequency"] >= min_shift_freq_target)].index)
    else:
        freq_criteria = set(shift_csv.loc[(shift_csv["source_frequency"] >= min_shift_freq_source)].index)
    merged_df = merged_df.loc[merged_df.index.isin(freq_criteria)].copy()
    ## Handle Data without similarity Score
    if dropna:
        merged_df = merged_df.dropna()
    else:
        merged_df = merged_df.fillna(0)
    ## Binning
    pbins = np.linspace(0, 100, npercentile)
    wbins = np.percentile(merged_df["weight"], pbins)
    wbins_lbl = (wbins[:-1] + wbins[1:]) / 2
    merged_df["weight_bin"] = pd.cut(merged_df["weight"], bins=wbins, labels=wbins_lbl, right=True, include_lowest=True)    
    ## Frequency Size
    merged_df["target_frequency"] = shift_csv["target_frequency"].fillna(0)
    merged_df["target_frequency_normed"] = shift_csv["target_frequency"].rank(pct=True)
    ## Standarized Groups
    merged_df["target_frequency_std"] = (merged_df["target_frequency"] - merged_df["target_frequency"].mean()) / merged_df["target_frequency"].std()
    merged_df["weight_std"] = (merged_df["weight"] - merged_df["weight"].mean()) / merged_df["weight"].std()
    merged_df["overlap_std"] = (merged_df["overlap"] - merged_df["overlap"].mean()) / merged_df["overlap"].std()
    ## Quadrant (Based on Group)
    merged_df["quadrant"] = merged_df.apply(lambda row: quadrant(row["weight_std"], row["overlap_std"]), axis=1)
    ## Anomaly Scores
    merged_df["anomaly_score"] = np.sqrt(merged_df["weight_std"] ** 2 + merged_df["overlap_std"] ** 2)
    ## Aggregation
    merged_df_agg = merged_df.groupby(["weight_bin"])["overlap"].apply(bootstrap_ci)
    merged_df_agg = bootstrap_tuple_to_df(merged_df_agg)
    ## Outliers
    qtopall = []
    for quad in range(1,5):
        qtop = merged_df.loc[merged_df["quadrant"]==quad]["anomaly_score"].nlargest(25).index.tolist()
        qtopall.extend(qtop)
    qtopall = merged_df.loc[qtopall].copy()
    qtopall.index = replace_emojis(qtopall.index)
    qtopall["weight_std"] = ((qtopall["weight"] - qtopall["weight"].mean()) / qtopall["weight"].std()).rank(pct=True)
    qtopall["overlap_std"] = ((qtopall["overlap"] - qtopall["overlap"].mean()) / qtopall["overlap"].std()).rank(pct=True)
    ## Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    mm = axes[0].scatter(merged_df["weight"].values,
                    merged_df["overlap"].values,
                    alpha=0.2,
                    c=merged_df["target_frequency_normed"].values,
                    cmap=plt.cm.coolwarm,
                    vmin=0,
                    vmax=1)
    axes[0].scatter(qtopall["weight"].values,
                    qtopall["overlap"].values,
                    alpha=0.4,
                    c=qtopall["target_frequency_normed"].values,
                    cmap=plt.cm.coolwarm,
                    vmin=0,
                    vmax=1,
                    edgecolor="black",
                    linewidth=1)
    axes[0].errorbar(merged_df_agg.index,
                     merged_df_agg["median"],
                     yerr=np.array([(merged_df_agg["median"]-merged_df_agg["lower"]).values,
                                    (merged_df_agg["upper"]-merged_df_agg["median"]).values]),
                     linewidth=1,
                     capsize=2,
                     marker="o",
                     markersize=2.5,
                     color="black",
                     alpha=1,
                     label="Average by Weight Bin")
    axes[0].axvline(merged_df["weight"].mean(), color="black", linestyle="--", alpha=0.5, zorder=-1)
    axes[0].axhline(merged_df["overlap"].mean(), color="black", linestyle="--", alpha=0.5, zorder=-1)
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].scatter(qtopall["weight_std"],
                    qtopall["overlap_std"],
                    c=qtopall["target_frequency_normed"].values,
                    cmap=plt.cm.coolwarm,
                    vmin=0,
                    vmax=1,
                    alpha=0.4)
    txts = []
    for tok in qtopall.index:
        txts.append(axes[1].text(qtopall.loc[tok,"weight_std"],
                                 qtopall.loc[tok,"overlap_std"],
                                 tok,
                                 fontsize=4,
                                 ha="left",
                                 va="center"))
    axes[1].axvline(0.5, color="black", linestyle="--", alpha=0.5, zorder=-1)
    axes[1].axhline(0.5, color="black", linestyle="--", alpha=0.5, zorder=-1)
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(labelsize=14)
    fig.text(0.5, 0.025, "Coefficient Weight", fontweight="bold", fontsize=16, ha="center", va="center")
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[0].set_xlim(wbins_lbl[0]-0.1, wbins_lbl[-1]+0.1)
    axes[0].set_ylabel("Semantic Similarity",
                       fontweight="bold",
                       fontsize=16)
    axes[0].set_title("Semantic Shift vs. Predictive Influence", fontweight="bold", loc="left")
    axes[1].set_title("Outliers by Quadrant", fontweight="bold", loc="left")
    cbar = fig.colorbar(mm, ax=axes[1])
    cbar.set_label("Frequency Percentile", fontweight="bold", fontsize=16, labelpad=15)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.125)
    return fig

def plot_performance_summary(reduced_vocabulary_classification_scores,
                             metric="accuracy"):
    """

    """
    ## Formatting
    test_scores = reduced_vocabulary_classification_scores.loc[reduced_vocabulary_classification_scores["group"]=="test"].copy()
    test_scores["vocabulary_selector"] = test_scores["vocabulary_id"].map(lambda i: i.split("@")[0])
    test_scores["vocabulary_selector_percentile"] = test_scores["vocabulary_id"].map(lambda i: int(i.split("@")[1]) if "@" in i else -1)
    if "sample" not in test_scores.columns:
        test_scores["sample"] = 0
    ## Baseline Duplication
    percentiles = list(filter(lambda i: i != -1, test_scores["vocabulary_selector_percentile"].unique()))
    test_scores_baseline = []
    for p in percentiles:
        pdf = test_scores.loc[test_scores["vocabulary_selector_percentile"]==-1].copy()
        if pdf.shape[0] == 0:
            continue
        pdf["vocabulary_selector_percentile"] = p
        test_scores_baseline.append(pdf)
    if len(test_scores_baseline) > 0:
        test_scores = pd.concat([test_scores.loc[test_scores["vocabulary_selector_percentile"] != -1],
                                pd.concat(test_scores_baseline, sort=False, axis=0)]).reset_index(drop=True)
    ## Metadata
    selector2group_r = {0:["cumulative","intersection"],1:["random","frequency"],2:["chi2","coefficient"]}
    selector2group = {j:x for x, y in selector2group_r.items() for j in y}
    selectors = sorted(test_scores["vocabulary_selector"].unique(), key=lambda s: selector2group.get(s, 3))
    colors = [f"C{i}" for i in range(len(selectors))]
    hatchs = [None for i in range(len(selectors))]
    ## Plot
    fig, axes = plt.subplots(1, 2, figsize=(10,5.8), sharey=False)
    bounds = [[np.inf,-np.inf] for _ in range(2)]
    for s, selector in tqdm(enumerate(selectors), total=len(selectors), file=sys.stdout, desc="[Selector]"):
        ## Get Selector Data
        s_tp_test = test_scores.loc[test_scores["vocabulary_selector"]==selector]
        ## Average Performance
        s_tp_test_mean = bootstrap_tuple_to_df(s_tp_test.groupby(["vocabulary_selector_percentile"]).agg({metric:bootstrap_ci})[metric])
        ## "Optimal" Performance
        s_tp_test = s_tp_test.sort_values([metric,"vocabulary_selector_percentile"],ascending=[False, False]).reset_index(drop=True).copy()
        s_tp_test_opt_p = s_tp_test.groupby(["sample","fold"]).apply(lambda i: i.iloc[0]["vocabulary_selector_percentile"])
        s_tp_test_opt_p = list(map(tuple, s_tp_test_opt_p.map(int).reset_index().values))
        s_tp_test_opt_p = s_tp_test.set_index(["sample","fold","vocabulary_selector_percentile"]).loc[s_tp_test_opt_p,metric]
        s_tp_test_opt_p_mean = bootstrap_ci(s_tp_test_opt_p)
        ## Plot Average Performance
        if selector in ["cumulative","intersection"]:
            axes[0].fill_between([0,100],
                                 s_tp_test_mean["lower"].values[0],
                                 s_tp_test_mean["upper"].values[0],
                                 color=colors[s],
                                 alpha=0.1)
            axes[0].axhline(s_tp_test_mean["median"].values[0],
                            linewidth=2,
                            alpha=0.8,
                            color=colors[s],
                            label=selector)
        else:
            axes[0].errorbar(s_tp_test_mean.index + 0.5 * s,
                             s_tp_test_mean["median"],
                             yerr=np.array([(s_tp_test_mean["median"]-s_tp_test_mean["lower"]).values,
                                             (s_tp_test_mean["upper"]-s_tp_test_mean["median"]).values]), 
                             label=selector, 
                             color=colors[s],
                             linewidth=0,
                             marker="o",
                             capsize=2,
                             elinewidth=2,
                             alpha=0.8)
        ## Plot Optimized Performance
        axes[1].bar(s, 
                    s_tp_test_opt_p_mean[1],
                    yerr=[[s_tp_test_opt_p_mean[1]-s_tp_test_opt_p_mean[0]], [s_tp_test_opt_p_mean[2]-s_tp_test_opt_p_mean[1]]],
                    color=colors[s],
                    label=selector,
                    hatch=hatchs[s],
                    capsize=2,
                    alpha=0.5) 

        for v, val in enumerate([s_tp_test_opt_p_mean]):
            axes[v + 1].text(s, val[2] + 0.002, "{:.3f}\n({:.3f},{:.3f})".format(val[1], val[0], val[2]), fontsize=8, ha="center", va="bottom", rotation=90)
        ## Bounds
        bounds[0][0] = min(bounds[0][0], s_tp_test_mean["lower"].min())
        bounds[0][1] = max(bounds[0][1], s_tp_test_mean["upper"].max())
        bounds[1][0] = min(bounds[1][0], s_tp_test_opt_p_mean[0])
        bounds[1][1] = max(bounds[1][1], s_tp_test_opt_p_mean[1])
    ## Final Formatting
    for b, (bl, bu) in enumerate(bounds):
        mult = [0.025,0.05][b]
        axes[b].set_ylim(bl * (1-mult), bu * (1+mult))
        axes[b].spines["top"].set_visible(False)
        axes[b].spines["right"].set_visible(False)
    axes[0].set_xlim(0, 100)
    axes[0].legend(loc="lower left", ncol=6, bbox_to_anchor=(0.1,-0.2), fontsize=8)
    axes[0].set_title(f"Average {metric.title()} Score", fontweight="bold")
    axes[1].set_title(f"Optimal {metric.title()} Score", fontweight="bold")
    axes[0].set_ylabel(f"{metric.title()} Score", fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_xlabel("Percentile", fontweight="bold")
    for i in [1]:
        axes[i].set_xticks([])
        axes[i].set_xlabel("Selector", fontweight="bold")
        axes[i].set_xlim(-0.5, len(selectors)-0.5)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    return fig

def merge_source_evaluation(X_source,
                            X_evaluation,
                            y_source,
                            y_evaluation,
                            groups_source,
                            groups_evaluation,
                            users_source,
                            users_evaluation):
    """
    
    """
    ## Merge Data
    X_merged = sparse.vstack([X_source,
                              X_evaluation[(groups_evaluation == 1).nonzero()[0]]])
    y_merged = np.hstack([y_source,
                          y_evaluation[groups_evaluation == 1]])
    users_merged = users_source + [users_evaluation[i] for i in (groups_evaluation == 1).nonzero()[0]]
    groups_merged = np.hstack([groups_source, np.ones((groups_evaluation == 1).sum(), dtype=int)])
    merged_id = np.hstack([np.zeros_like(groups_source), np.ones((groups_evaluation==1).sum(), dtype=int)])
    ## Ensure Distinct Users
    train_users = set([users_merged[i] for i in (groups_merged == 0).nonzero()[0]])
    mask = [i for i, (g, user) in enumerate(zip(groups_merged, users_merged)) if g == 0 and user in train_users or g == 1 and user not in train_users]
    ## Apply Distinct Mask
    X_merged = X_merged[mask]
    y_merged = y_merged[mask]
    users_merged = [users_merged[m] for m in mask]
    groups_merged = groups_merged[mask]
    merged_id = merged_id[mask]
    return X_merged, y_merged, groups_merged, users_merged, merged_id

def main():
    """
    
    """
    ## Parse Command Line
    args = parse_command_line()
    ## Load Configuration and Initialize Output Directory
    config, output_dir = load_configuration(args.config,
                                            output_dir=args.output_dir,
                                            allow_overwrite=args.allow_overwrite)
    ## Load Splits, Embeddings, etc.
    source_data = load_resources(config["source"], evaluation=False)
    target_data = load_resources(config["target"], evaluation=False)
    evaluation_data = load_resources(config["evaluation"], evaluation=True)
    ## Align and Compute Similarity
    vocabulary, s_neighbors, t_neighbors, s_frequency, t_frequency, similarity = align_and_compute_similarity(source_data=source_data,
                                                                                                              target_data=target_data,
                                                                                                              config=config)
    ## Generate Summary of Semantic Shift
    shift, shift_csv = describe_shift(global_vocabulary=vocabulary,
                                      source_neighbors=s_neighbors,
                                      target_neighbors=t_neighbors,
                                      source_frequency=s_frequency,
                                      target_frequency=t_frequency,
                                      similarity=similarity,
                                      min_shift_freq_source=config["source"]["min_shift_freq"],
                                      min_shift_freq_target=config["target"]["min_shift_freq"],
                                      top_k=50,
                                      top_k_neighbors=20)
    with open(f"{output_dir}/shift.summary.txt","w") as the_file:
        the_file.write(shift)
    _ = shift_csv.reset_index().to_csv(f"{output_dir}/shift.summary.csv",index=False)
    ## Generate Document Term Matrix
    LOGGER.info("[Generating Document Term Matrices (Source)]")
    X_source, y_source, groups_source, users_source = vectorize_data(splits=source_data["splits"],
                                                                     vocabulary=vocabulary,
                                                                     phrasers=source_data["phrasers"],
                                                                     config=source_data["configuration"],
                                                                     min_n_posts=config["classifier_min_n_posts"],
                                                                     n_posts=config["classifier_n_posts"],
                                                                     groups=["train","test"],
                                                                     njobs=args.jobs,
                                                                     random_state=config["random_state"])
    LOGGER.info("[Generating Document Term Matrices (Evaluation)]")
    X_eval, y_eval, groups_eval, users_eval = vectorize_data(splits=evaluation_data["splits"],
                                                             vocabulary=vocabulary,
                                                             phrasers=source_data["phrasers"],
                                                             config=evaluation_data["configuration"],
                                                             min_n_posts=config["classifier_min_n_posts"],
                                                             n_posts=config["classifier_n_posts"],
                                                             groups=["test"],
                                                             njobs=args.jobs,
                                                             random_state=config["random_state"])
    ## Merge
    LOGGER.info("[Merging Document Term Matrices]")
    X_merged, y_merged, groups_merged, users_merged, ids_merged = merge_source_evaluation(X_source, X_eval,
                                                                                          y_source, y_eval,
                                                                                          groups_source, groups_eval,
                                                                                          users_source, users_eval)
    ## Generate Source Data Splits 
    model_splits = split_data(y=y_merged,
                              groups=groups_merged,
                              ids=ids_merged,
                              n_samples=config["classifier_k_models"],
                              resample_rate=config["classifier_resample_rate"],
                              random_state=config["random_state"])
    ## Fit Baseline Models + Compute Statistics
    LOGGER.info("[Beginning Baseline Model Fitting]")
    baseline_vocabulary_scores = []
    baseline_classification_scores = []
    baseline_classification_weights = []
    for sample, (train_, test_source, test_eval) in tqdm(enumerate(model_splits), total=len(model_splits), desc="[Baseline Model Fitting]", file=sys.stdout):
        ## Get Data
        Xs_train, Xs_test, Xe_test = X_merged[train_], X_merged[test_source], X_merged[test_eval]
        ys_train, ys_test, ye_test = y_merged[train_], y_merged[test_source], y_merged[test_eval]
        ## Vocabulary Filtering
        vmask_sample = get_vocabulary_mask(X=Xs_train,
                                           y=ys_train,
                                           source_frequency=s_frequency,
                                           target_frequency=t_frequency,
                                           config=config,
                                           enforce_frequency_filter=args.classify_enforce_frequency_filter,
                                           enforce_classwise_frequency_filter=args.classify_enforce_classwise_frequency_filter,
                                           enforce_min_shift_frequency_filter=args.classify_enforce_min_shift_frequency_filter,
                                           enforce_target=args.classify_enforce_target)
        ## Apply Frequency Filtering
        Xs_train = Xs_train[:,vmask_sample]
        Xs_test = Xs_test[:,vmask_sample]
        Xe_test = Xe_test[:,vmask_sample]
        vocabulary_sample = [vocabulary[s] for s in vmask_sample]
        ## Statistics
        sample_vocabulary_statistics = compute_vocabulary_statistics(Xs_train, ys_train, vocabulary_sample)
        sample_vocabulary_scores = compute_vocabulary_scores(sample_vocabulary_statistics, alpha=1)
        sample_vocabulary_scores["chi2"], _ = feature_selection.chi2(Xs_train, ys_train)
        sample_vocabulary_scores["fold"] = sample
        ## Fit Classifier
        sample_preprocessor, sample_model, sample_scores_train = fit_classifier(X_train=Xs_train,
                                                                                y_train=ys_train,
                                                                                vocabulary=vocabulary_sample,
                                                                                config=config,
                                                                                njobs=args.jobs,
                                                                                random_state=config["random_state"])
        ## Evaluate Classifier
        sample_scores_test_source = evaluate_classifier(Xs_test, ys_test, sample_preprocessor, sample_model)
        sample_scores_test_eval = evaluate_classifier(Xe_test, ye_test, sample_preprocessor, sample_model)
        ## Format Scores
        sample_scores = [sample_scores_train, sample_scores_test_source, sample_scores_test_eval]
        for cs, group, y_true, data_id in zip(sample_scores, ["train","test","test"], [ys_train, ys_test, ye_test], ["source","source","evaluation"]):
            cs["group"] = group
            cs["support"] = y_true.shape[0]
            cs["class_balance"] = y_true.mean()
            cs["fold"] = sample
            cs["data_id"] = data_id
        ## Weights
        sample_weights = pd.Series(sample_model.coef_[0], [" ".join(f) for f in get_feature_names(sample_preprocessor)]).to_frame("weight")
        sample_weights["fold"] = sample
        ## Cache
        baseline_vocabulary_scores.append(sample_vocabulary_scores)
        baseline_classification_scores.extend(sample_scores)
        baseline_classification_weights.append(sample_weights)
    ## Format Results
    LOGGER.info("[Formatting Baseline Results]")
    baseline_vocabulary_scores = pd.concat(baseline_vocabulary_scores, sort=False).reset_index().rename(columns={"term":"feature"})
    baseline_classification_weights = pd.concat(baseline_classification_weights).reset_index().rename(columns={"index":"feature"})
    baseline_classification_scores = pd.DataFrame(baseline_classification_scores)
    ## Cache Results
    LOGGER.info("[Caching Baseline Results]")
    _ = baseline_vocabulary_scores.to_csv(f"{output_dir}/baseline_vocabulary_scores.csv",index=False)
    _ = baseline_classification_weights.to_csv(f"{output_dir}/baseline_classification_weights.csv",index=False)
    _ = baseline_classification_scores.to_csv(f"{output_dir}/baseline_classification_scores.csv",index=False)
    ## Shift vs. Influence Visualization
    LOGGER.info("[Visualizing Shift vs. Weight Summary]")
    fig = plot_shift_influence_summary(baseline_classification_weights=baseline_classification_weights,
                                       shift_csv=shift_csv,
                                       min_shift_freq_source=config["source"]["min_shift_freq"],
                                       min_shift_freq_target=config["target"]["min_shift_freq"],
                                       enforce_target=args.classify_enforce_target)
    _ = fig.savefig(f"{output_dir}/shift-influence.correlation.png", dpi=150)
    plt.close(fig)
    ## Vocabulary Selection
    LOGGER.info("[Running Vocabulary Selection Procedure]")
    reduced_vocabularies = run_vocabulary_selection(vocabulary=vocabulary,
                                                    source_frequency=s_frequency,
                                                    target_frequency=t_frequency,
                                                    similarity=similarity,
                                                    baseline_classification_weights=baseline_classification_weights,
                                                    baseline_vocabulary_scores=baseline_vocabulary_scores,
                                                    config=config)
    ## Measure Overlap
    reduced_vocabularies_overlap = measure_reduced_vocabulary_overlap(reduced_vocabularies)
    _ = reduced_vocabularies_overlap.to_csv(f"{output_dir}/reduced_vocabularies.overlap.csv")
    ## Initialize Model Output Directory
    LOGGER.info("[Initializing Model Cache Directory]")
    model_output_dir = make_dirs(f"{output_dir}/models/",
                                 allow_overwrite=args.allow_overwrite)
    ## Refit Models with Vocabulary, Evaluate Within-Domain, Cache Models
    LOGGER.info("[Beginning Reduced Vocabulary Model Fitting]")
    reduced_vocabulary_classification_scores = []
    for vocabulary_id, vocabulary_indices in tqdm(reduced_vocabularies.items(),
                                                  total=len(reduced_vocabularies),
                                                  file=sys.stdout,
                                                  position=0,
                                                  leave=True,
                                                  desc="[Vocabulary Set]"):
        ## Translate
        vocabulary_terms = [vocabulary[ind] for ind in vocabulary_indices]
        ## Vocabulary ID Information
        vocabulary_selector = vocabulary_id.split("@")[0]
        vocabulary_selector_percentile = "100" if "@" not in vocabulary_id else vocabulary_id.split("@")[1]
        ## Cache Vocabulary
        vocabulary_output_dir = make_dirs(f"{model_output_dir}/{vocabulary_selector}/{vocabulary_selector_percentile}/",
                                          True)
        with open(f"{vocabulary_output_dir}/vocabulary.txt","w") as the_file:
            for term in vocabulary_terms:
                the_file.write(f"{term}\n")
        ## Iterate Through Terms
        for sample, (train_, test_source, test_eval) in tqdm(enumerate(model_splits),
                                                             total=len(model_splits),
                                                             desc="[Data Sample]",
                                                             file=sys.stdout,
                                                             position=1,
                                                             leave=False):
            ## Establish Directory
            sample_vocabulary_output_dir = make_dirs(f"{vocabulary_output_dir}/{sample}/", True)
            ## Get Data
            Xs_train, Xs_test, Xe_test = X_merged[train_], X_merged[test_source], X_merged[test_eval]
            ys_train, ys_test, ye_test = y_merged[train_], y_merged[test_source], y_merged[test_eval]
            ## Apply Vocabulary Mask
            Xs_train, Xs_test, Xe_test = Xs_train[:,vocabulary_indices], Xs_test[:,vocabulary_indices], Xe_test[:,vocabulary_indices]
            ## Fit Classifier
            sample_preprocessor, sample_model, sample_scores_train = fit_classifier(X_train=Xs_train,
                                                                                    y_train=ys_train,
                                                                                    vocabulary=vocabulary_terms,
                                                                                    config=config,
                                                                                    njobs=args.jobs,
                                                                                    random_state=config["random_state"])
            ## Evaluate Classifier
            sample_scores_test_source = evaluate_classifier(Xs_test, ys_test, sample_preprocessor, sample_model)
            sample_scores_test_evaluation = evaluate_classifier(Xe_test, ye_test, sample_preprocessor, sample_model)
            ## Format Scores
            sample_scores = [sample_scores_train, sample_scores_test_source, sample_scores_test_evaluation]
            for cs, group, y_true, data_id in zip(sample_scores, ["train","test","test"], [ys_train, ys_test, ye_test], ["source","source","evaluation"]):
                cs["group"] = group
                cs["support"] = y_true.shape[0]
                cs["class_balance"] = y_true.mean()
                cs["fold"] = sample
                cs["vocabulary_id"] = vocabulary_id
                cs["data_id"] = data_id
            ## Cache Scores
            reduced_vocabulary_classification_scores.extend(sample_scores)
            ## Cache Model + Preprocessor
            _ = joblib.dump(sample_model, f"{sample_vocabulary_output_dir}/model.joblib")
            _ = joblib.dump(sample_preprocessor, f"{sample_vocabulary_output_dir}/preprocessor.joblib")
    ## Format Results
    LOGGER.info("[Formatting Reduced Vocabulary Classification Results]")
    reduced_vocabulary_classification_scores = pd.DataFrame(reduced_vocabulary_classification_scores)
    ## Cache Results
    LOGGER.info("[Caching Reduced Vocabulary Classification Results]")
    _ = reduced_vocabulary_classification_scores.to_csv(f"{output_dir}/reduced_vocabulary_classification_scores.csv",index=False)
    ## Visualize Classification Performance
    LOGGER.info("[Visualizing Reduced Vocabulary Classification Results]")
    for data_id in ["source","evaluation"]:
        data_id_scores = reduced_vocabulary_classification_scores.loc[reduced_vocabulary_classification_scores["data_id"] == data_id]
        for metric in ["accuracy","f1","auc"]:
            fig = plot_performance_summary(data_id_scores, metric)
            _ = fig.savefig(f"{output_dir}/scores.summary.{data_id}.{metric}.png", dpi=150)
            plt.close(fig)
        ## Done
    LOGGER.info("[Script Complete]")

############################
### Execute
############################

if __name__ == "__main__":
    _ = main()