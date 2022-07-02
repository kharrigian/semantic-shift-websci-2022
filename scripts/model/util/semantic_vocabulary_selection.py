
"""
Use Neighbor Overlap In Word Embedding Space to 
identity semantically stable vocabularies
"""

########################
### Imports
########################

## Standard Library
import os
import sys
import json
import argparse
from glob import glob
from functools import partial
from textwrap import wrap

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

## Project Specific Libraries
from semshift.util.helpers import chunks
from semshift.model.embed import Word2Vec
from semshift.preprocess.tokenizer import STOPWORDS
from semshift.util.logging import initialize_logger

########################
### Globals
########################

## Logger
LOGGER = initialize_logger()

########################
### Functions
########################

def parse_arguments():
    """
    
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Run modeling experiments")
    ## Generic Arguments
    parser.add_argument("model_dirs",
                        type=str,
                        nargs=2,
                        help="Directories where embedding models live")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the vocabularies.")
    parser.add_argument("--overlap_type",
                        type=str,
                        choices=set(["overlap","jaccard"]),
                        default="overlap")
    parser.add_argument("--top_k_neighbors",
                        type=int,
                        default=1000,
                        help="Neighbor window to use for computing overlap.")
    parser.add_argument("--top_k_neighbors_secondary",
                        type=int,
                        default=None,
                        help="If included, can set a second neighbor threshold for second directory.")
    parser.add_argument("--min_vocab_freq",
                        type=int,
                        default=5,
                        help="Minimum frequency in vocabulary to maintain n-gram")
    parser.add_argument("--min_vocab_freq_secondary",
                        type=int,
                        default=None,
                        help="If included, can set separate vocabulary threshold for second directory.")
    parser.add_argument("--min_shift_freq",
                        type=int,
                        default=100,
                        help="Terms with fewer than this frequency aren't candidates for stability.")
    parser.add_argument("--min_shift_freq_secondary",
                        type=int,
                        default=None,
                        help="If included, can set separate vocabulary threshold for second directory.")
    parser.add_argument("--rm_top",
                        type=int,
                        default=200,
                        help="Number of most frequent terms to remove")
    parser.add_argument("--keep_stopwords",
                        action="store_true",
                        default=False,
                        help="If true, do not automatically filter out stopwords.")
    parser.add_argument("--filter_hashtags",
                        action="store_true",
                        default=False,
                        help="If true, excludes hashtags from vocabulary.")
    parser.add_argument("--percentiles",
                        nargs="+",
                        default=[90],
                        type=int,
                        help="Overlap thresholds to use as cutoffs. Input as space separated list.")
    parser.add_argument("--conserve_memory",
                        action="store_true",
                        default=False,
                        help="If included, similarity computation done in chunks.")
    parser.add_argument("--conserve_memory_chunksize",
                        type=int,
                        default=500,
                        help="If --conserve_memory included, this specifies computation chunksize.")
    parser.add_argument("--display_top_k",
                        type=int,
                        default=50,
                        help="Show the top k least stable terms")
    parser.add_argument("--display_top_k_terms",
                        type=int,
                        default=10,
                        help="Show the top k terms closest in cosine distance to a given term.")
    parser.add_argument("--highlight_difference",
                        action="store_true",
                        default=False,
                        help="If included, only highlight top terms that are unique to a domain.")
    parser.add_argument("--keywords",
                        type=str,
                        nargs="+",
                        default=None,
                        help="Individual keywords or keyword files.")
    ## Parse Arguments
    args = parser.parse_args()
    return args

def load_embeddings(directory):
    """

    """
    ## Load Configuration
    if os.path.exists(f"{directory}/config.json"):
        with open(f"{directory}/config.json","r") as the_file:
            config = json.load(the_file)
        ## ID
        exp_id = config.get("experiment_name",None)
        ## Load Embeddings
        embedding_dim = int(config["model_kwargs"]["dim"])
        embedding_file = f"{directory}/embeddings.{embedding_dim}d.txt"
    else:
        config = {}
        exp_id = None
        embedding_file = glob(f"{directory}/embeddings.*txt")[0]
        embedding_dim = int(os.path.basename(embedding_file).split("embeddings.")[1].split("d.txt")[0])
    vocab, embeddings = [], []
    with open(embedding_file,"r") as the_file:
        for line in the_file:
            line = line.strip().split()
            ngram, vector = line[:-embedding_dim], line[-embedding_dim:]
            vocab.append(" ".join(ngram))
            embeddings.append(np.array(vector).astype(float))
    embeddings = np.vstack(embeddings)
    if embeddings.shape[0] != len(vocab):
        raise ValueError("Embedding length does not match vocab length")
    ## Load Model
    model = Word2Vec.load(f"{directory}/")
    vocab_freq = np.array([model.model.wv.get_vecattr(v,"count") for v in vocab])
    return exp_id, vocab, embeddings, vocab_freq, config

def get_keywords(keywords):
    """
    
    """
    ## Initialize Cache
    all_keywords = set()
    ## Check Input
    if keywords is None:
        return all_keywords
    ## Get All Keywords
    for k in keywords:
        if os.path.exists(k):
            with open(k,"r") as the_file:
                for line in the_file:
                    all_keywords.add(line.strip().lower())
        else:
            all_keywords.add(k)
    all_keywords = sorted(all_keywords)
    return all_keywords
                
def filter_embeddings(vocab,
                      embeddings,
                      freq,
                      min_vocab_freq=100,
                      rm_top=200,
                      keep_stopwords=False,
                      filter_hashtags=False):
    """

    """
    ## Decide On Stopset
    stopset = STOPWORDS if not keep_stopwords else set()
    ## Identify Mask
    vocab_mask = set((freq >= min_vocab_freq).nonzero()[0]) & \
                 set([i for i, v in enumerate(vocab) if v not in stopset]) & \
                 set(set(freq.argsort()[:-rm_top]))
    if filter_hashtags:
        vocab_mask = vocab_mask & set(i for i, v in enumerate(vocab) if not v.startswith("<HASHTAG="))
    vocab_mask = sorted(vocab_mask)
    ## Apply Filter
    vocab = [vocab[m] for m in vocab_mask]
    embeddings = embeddings[vocab_mask]
    freq = freq[vocab_mask]
    return vocab, embeddings, freq

def compute_overlap(*neighbors):
    """

    """
    for n in neighbors:
        if n.min() == -1:
            return 0.0, 0.0
    top_k = neighbors[0].shape[0]
    neighbors = [set(n) for n in neighbors]
    intersection = set.intersection(*neighbors)
    union = set.union(*neighbors)
    return len(intersection) / top_k, len(intersection) / len(union)

def get_neighbors(vocab,
                  embeddings,
                  global_ind2vocab,
                  global_vocab2ind,
                  top_k_neighbors=1000,
                  conserve_memory=False,
                  chunksize=500):
    """

    """
    ## Create Dictionary Lookup
    word2ind = dict((word, ind) for ind, word in enumerate(vocab))
    ## Compute Similarity and Get Neighbors
    if conserve_memory:
        sim_neighbors = []
        index_chunks = list(chunks(list(range(embeddings.shape[0])), chunksize))
        for index_chunk in tqdm(index_chunks, position=0, leave=True, file=sys.stdout, desc="Similarity Computation"):
            vector_sim = cosine_similarity(embeddings[index_chunk], embeddings)
            sim_neighbors.append(vector_sim.argsort(axis=1)[:,-top_k_neighbors-1:-1][:,::-1])
        sim_neighbors = np.vstack(sim_neighbors)
    else:
        sim = cosine_similarity(embeddings, embeddings)
        sim_neighbors = sim.argsort(axis=1)[:,-top_k_neighbors-1:-1][:,::-1]
    ## Align Neighboring Indices Globally
    sim_neighbors_global = []
    for neighborhood in tqdm(sim_neighbors, desc="Aligning Neighborhood", position=0, file=sys.stdout, leave=True):
        sim_neighbors_global.append([global_vocab2ind.get(vocab[n]) for n in neighborhood])
    sim_neighbors_global = np.array(sim_neighbors_global, dtype=int)
    sim_neighbors_global = np.vstack([sim_neighbors_global, 
                                      np.ones((1, sim_neighbors_global.shape[1]), dtype=int) * -1])
    ## Sort Vocabulary for Alignment
    sim_neighbors_global = sim_neighbors_global[[word2ind.get(word,-1) for word in global_ind2vocab]]
    return sim_neighbors_global

def compare_terms(term,
                  neighbors_a,
                  neighbors_b,
                  vocab2ind,
                  ind2vocab,
                  overlap_scores_df,
                  top_k=10,
                  highlight_difference=False):
    """

    """
    if term not in vocab2ind or term not in set(overlap_scores_df.index.values):
        a_terms = ["Term does not exist."]
        b_terms = ["Term does not exist."]
        overlap_score = 0
    else:
        a_terms = [ind2vocab[i] for i in neighbors_a[vocab2ind.get(term)]]
        b_terms = [ind2vocab[i] for i in neighbors_b[vocab2ind.get(term)]]
        if not highlight_difference:
            a_terms = a_terms[:top_k]
            b_terms = b_terms[:top_k]
        else:
            a_terms_set = set(a_terms)
            b_terms_set = set(b_terms)
            a_terms = list(filter(lambda t: t not in b_terms_set, a_terms))[:top_k]
            b_terms = list(filter(lambda t: t not in a_terms_set, b_terms))[:top_k]
        overlap_score = overlap_scores_df.loc[term]["overlap"]
    return a_terms, b_terms, overlap_score

def plot_frequency_distribution(freq_a,
                                freq_b):
    """

    """
    ## Transformations
    freq_a_l = np.log10(freq_a)
    freq_b_l = np.log10(freq_b)
    ## Plot
    fig, ax = plt.subplots(1,2,figsize=(10,5.6))
    for d, data in enumerate([freq_a_l,freq_b_l]):
        ax[0].hist(data,
                   color=f"C{d}",
                   alpha=0.3,
                   bins=100,
                   density=True,
                   weights=np.ones_like(data) / data.shape[0])
        ax[1].hist(data,
                   color=f"C{d}",
                   alpha=0.3,
                   bins=100,
                   density=True,
                   cumulative=True,
                   histtype="step",
                   linewidth=3,
                   weights=np.ones_like(data) / data.shape[0])
    for i in range(2):
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        xticks = ax[i].get_xticks()
        xticklabels = [f"10$^{int(x)}$" for x in xticks]
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticklabels)
        ax[i].tick_params(labelsize=12)
        ax[i].set_xlim(min(freq_a_l.min(), freq_b_l.min()),
                       max(freq_a_l.max(), freq_b_l.max()))
        ax[i].set_xlabel("Term Frequency", fontweight="bold", fontsize=14)
    ax[0].set_ylabel("Density (normalized)", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return fig, ax

def display_vocab_sizes(freq_a,
                        freq_b,
                        vocab_a,
                        vocab_b,
                        thresholds=[5,10,25,50,100,150,200]):
    """

    """
    independent_sizes = np.zeros((2, len(thresholds)), dtype=int)
    joint_sizes = np.zeros_like(independent_sizes)
    for t, threshold in enumerate(thresholds):
        av = set(v for f, v in zip(freq_a, vocab_a) if f >= threshold)
        bv = set(v for f, v in zip(freq_b, vocab_b) if f >= threshold)
        independent_sizes[:,t] = [len(av), len(bv)]
        joint_sizes[:,t] = [len(av & bv), len(av | bv)]
    sizes = pd.DataFrame(data=np.vstack([independent_sizes, joint_sizes]),
                         columns=thresholds,
                         index=["[A]","[B]","Intersection","Union"])
    sizes.columns.name = "Threshold"
    sizes.index.name = "Count Type"
    sizes = sizes.applymap(lambda i: "{:,d}".format(i))
    return sizes

def main():
    """

    """
    ## Parse Arguments
    args = parse_arguments()
    ## Establish Output Directory
    if args.output_dir is not None and not os.path.exists(args.output_dir):
        _ = os.makedirs(args.output_dir)
    ## Cache Configuration
    if args.output_dir is not None:
        LOGGER.info("[Caching Selection Arguments]")
        with open(f"{args.output_dir}selection.config.json","w") as the_file:
            _ = json.dump(vars(args), the_file)
    ## Load Embeddings
    LOGGER.info("[Loading Embeddings]")
    exp_a, vocab_a, embeddings_a, freq_a, config_a = load_embeddings(args.model_dirs[0])
    exp_b, vocab_b, embeddings_b, freq_b, config_b = load_embeddings(args.model_dirs[1])
    if exp_a is None:
        exp_a = "A"
    if exp_b is None:
        exp_b = "B"
    ## Extract Vocabulary Distribution
    LOGGER.info("[Extracting Initial Vocabulary Distribution]")
    vocab_sizes = display_vocab_sizes(freq_a, freq_b, vocab_a, vocab_b)
    LOGGER.info(f"\n{':'*75}\n{vocab_sizes.to_string()}\n{':'*75}\n")
    if args.output_dir is not None:
        _ = vocab_sizes.to_csv(f"{args.output_dir}/sizes.csv",index=True)
    ## Plot Distribution
    if args.output_dir is not None:
        LOGGER.info("[Visualizing Vocabulary Frequency Distributions]")
        fig, ax = plot_frequency_distribution(freq_a, freq_b)
        fig.savefig(f"{args.output_dir}/vocabulary_frequency_distribution.png", dpi=150)
        plt.close(fig)
    ## Filter Embeddings By Input Criteria
    LOGGER.info("[Filtering Embeddings]")
    vocab_a, embeddings_a, freq_a = filter_embeddings(vocab_a,
                                                      embeddings_a,
                                                      freq_a,
                                                      min_vocab_freq=args.min_vocab_freq,
                                                      rm_top=args.rm_top,
                                                      keep_stopwords=args.keep_stopwords,
                                                      filter_hashtags=args.filter_hashtags)
    vocab_b, embeddings_b, freq_b = filter_embeddings(vocab_b,
                                                      embeddings_b,
                                                      freq_b,
                                                      min_vocab_freq=args.min_vocab_freq if args.min_vocab_freq_secondary is None else args.min_vocab_freq_secondary,
                                                      rm_top=args.rm_top,
                                                      keep_stopwords=args.keep_stopwords,
                                                      filter_hashtags=args.filter_hashtags)
    ## Align Vocabulary
    LOGGER.info("[Aligning Vocabulary]")
    vocabulary_union = set(vocab_a) | set(vocab_b)
    vocabulary_intersection = set(vocab_a) & set(vocab_b)
    ind2vocab = sorted(vocabulary_union)
    vocab2ind = dict((v,i) for i, v in enumerate(ind2vocab))
    ## Cache Baseline Vocabularies
    exp_a = exp_a.replace("/","-")
    exp_b = exp_b.replace("/","-")
    if args.output_dir is not None:
        LOGGER.info("[Caching Baseline Vocabularies]")
        for v, vname in zip([vocab_a, vocab_b, vocabulary_union, vocabulary_intersection],
                            [f"baseline_{exp_a}.txt",f"baseline_{exp_b}.txt","union.txt","intersection.txt"]):
            with open(f"{args.output_dir}/{vname}","w") as the_file:
                for line in sorted(v):
                    the_file.write(f"{line}\n")
    ## Get Top Neighbors
    LOGGER.info("[Computing Neighborhoods]")
    neighbors_a = get_neighbors(vocab_a,
                                embeddings_a,
                                global_ind2vocab=ind2vocab,
                                global_vocab2ind=vocab2ind,
                                top_k_neighbors=args.top_k_neighbors,
                                conserve_memory=args.conserve_memory,
                                chunksize=args.conserve_memory_chunksize)
    neighbors_b = get_neighbors(vocab_b,
                                embeddings_b,
                                global_ind2vocab=ind2vocab,
                                global_vocab2ind=vocab2ind,
                                top_k_neighbors=args.top_k_neighbors if args.top_k_neighbors_secondary is None else args.top_k_neighbors_secondary,
                                conserve_memory=args.conserve_memory,
                                chunksize=args.conserve_memory_chunksize)
    ## Compute Overlap
    LOGGER.info("[Calculating Overlap Scores]")
    overlap_scores = np.array(list(map(lambda x: compute_overlap(*x), zip(neighbors_a, neighbors_b))))
    ## Format Scores and Cache
    LOGGER.info("[Formatting Scores]")
    overlap_scores_df = pd.DataFrame(overlap_scores,
                                     index=ind2vocab,
                                     columns=["overlap","jaccard"])
    overlap_scores_df["freq_a"] = pd.Series(index=vocab_a, data=freq_a)
    overlap_scores_df["freq_b"] = pd.Series(index=vocab_b, data=freq_b)
    overlap_scores_df = overlap_scores_df.fillna(0)
    for col in ["freq_a","freq_b"]:
        overlap_scores_df[col] = overlap_scores_df[col].astype(int)
    overlap_scores_df = overlap_scores_df.sort_values(["overlap"],ascending=False)
    ## Cache Scores
    if args.output_dir is not None:
        LOGGER.info("[Caching Scores]")
        _ = overlap_scores_df.to_csv(f"{args.output_dir}/scores.csv")
    ## Extract Acceptable Terms
    LOGGER.info("[Isolating Acceptable Terms]")
    secondary_threshold = args.min_shift_freq if args.min_shift_freq_secondary is None else args.min_shift_freq_secondary
    overlap_scores_df_perc = overlap_scores_df.loc[(overlap_scores_df.freq_a > args.min_shift_freq) &
                                                   (overlap_scores_df.freq_b > secondary_threshold)]
    ## Overlap Distribution Statistics
    LOGGER.info("[Extracting Statistics]")
    overlap_stats = overlap_scores_df_perc["overlap"].describe().to_string().replace("\n","\n\t")
    LOGGER.info(f"\tStatistics:\n\t{overlap_stats}")
    ## Extract Vocabularies
    if args.output_dir is not None:
        LOGGER.info("[Extracting Vocabularies]")
        for percentile in args.percentiles:
            p_thresh = np.nanpercentile(overlap_scores_df_perc[args.overlap_type], percentile)
            p_vocab = overlap_scores_df_perc.loc[overlap_scores_df_perc[args.overlap_type] >= p_thresh]
            p_vocab = p_vocab[["freq_a","freq_b"]].sum(axis=1).sort_values(ascending=False).index.tolist()
            with open(f"{args.output_dir}/{args.overlap_type}_{percentile}-threshold.txt","w") as the_file:
                for term in p_vocab:
                    the_file.write(f"{term}\n")
    ## Identify Terms With Most and Least Siginicant Change
    least_stable = overlap_scores_df_perc[args.overlap_type].nsmallest(args.display_top_k).index
    most_stable = overlap_scores_df_perc[args.overlap_type].nlargest(args.display_top_k).index
    ## Display Least Stable Keywords
    display_func = partial(compare_terms,
                           neighbors_a=neighbors_a,
                           neighbors_b=neighbors_b,
                           ind2vocab=ind2vocab,
                           vocab2ind=vocab2ind,
                           overlap_scores_df=overlap_scores_df_perc,
                           top_k=args.display_top_k_terms,
                           highlight_difference=args.highlight_difference)
    stability_str = ""
    for term_set, term_set_type in zip([least_stable,most_stable],["Least","Most"]):
        stability_str += f"\n{term_set_type} Stable Terms:\n"
        for t, term in enumerate(term_set):
            a_terms, b_terms, overlap_score = display_func(term)
            stability_str += f"{t+1}) {term} [{overlap_score}]\n"
            stability_str += "\t{}: {}\n".format(f"[A]", "\n\t     ".join(wrap(", ".join(a_terms), 70)))
            stability_str += "\t{}: {}\n".format(f"[B]", "\n\t     ".join(wrap(", ".join(b_terms), 70)))
    LOGGER.info(stability_str)
    if args.output_dir is not None:
        with open(f"{args.output_dir}/stability.log","w") as the_file:
            the_file.write(stability_str)
    ## Display Specifc Keywords
    all_keywords = get_keywords(args.keywords)
    if all_keywords:
        keyword_str = "\nIndividual Keyword Analysis:\n"
        ## Get Scores and Missing Words
        missing_keywords = []
        keyword_scores = []
        for _, keyword in enumerate(all_keywords):
            a_terms, b_terms, overlap_score = display_func(keyword)
            if a_terms[0] == "Term does not exist.":
                missing_keywords.append(keyword)
                continue
            keyword_scores.append([keyword, a_terms, b_terms, overlap_score])
        ## Sort Scores and Display
        keyword_scores = sorted(keyword_scores, key=lambda x: x[-1])
        for keyword, a_terms, b_terms, overlap_score in keyword_scores:
            keyword_str += f"** {keyword} [{overlap_score}]\n"
            keyword_str += "\t{}: {}\n".format(f"[A]", "\n\t     ".join(wrap(", ".join(a_terms), 70)))
            keyword_str += "\t{}: {}\n".format(f"[B]", "\n\t     ".join(wrap(", ".join(b_terms), 70)))
        ## Show Missing Keywords
        if missing_keywords:
            keyword_str += f"{len(missing_keywords)} keywords were not found in the filtered data."
        LOGGER.info(keyword_str)
        if args.output_dir is not None:
            with open(f"{args.output_dir}/keywords.log","w") as the_file:
                the_file.write(keyword_str)

#######################
### Execution
#######################

if __name__ == "__main__":
    _ = main()