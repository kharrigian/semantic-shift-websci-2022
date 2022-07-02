
####################
### Imports
####################

## Standard Libary
import os
import sys
import math
import json
from collections import Counter
from functools import partial
from uuid import uuid4
from multiprocessing import Pool

## External Libraries
from tqdm import tqdm
import pandas as pd
from nltk.corpus import words, brown

## Local Modules
from .data_loaders import LoadProcessedData
from ..preprocess.tokenizer import PUNCTUATION
from ..util.helpers import flatten, chunks
from .feature_extractors import LIWCTransformer

####################
### Globals
####################

## Data Resources
RESOURCE_DIR = "data/resources/"

####################
### Class Definition
####################

class Vocabulary(object):

    """
    Vocabulary class
    """

    def __init__(self,
                 filter_negate=False,
                 filter_upper=False,
                 filter_punctuation=False,
                 filter_numeric=False,
                 filter_user_mentions=False,
                 filter_url=False,
                 filter_retweet=False,
                 filter_stopwords=False,
                 keep_pronouns=True,
                 preserve_case=True,
                 emoji_handling=None,
                 filter_hashtag=False,
                 strip_hashtag=False,
                 max_vocab_size=100000,
                 min_token_freq=0,
                 max_token_freq=None,
                 ngrams=(1,1),
                 max_tokens_per_document=None,
                 max_documents_per_user=None,
                 binarize_counter=False,
                 filter_mh_subreddits=None,
                 filter_mh_terms=None,
                 keep_retweets=True,
                 external_vocab=[],
                 external_only=False,
                 processor=None,
                 processor_kwargs={},
                 lang=None,
                 random_state=42):
        """
        Vocabulary Learner

        Args:
            filter_negate (bool): Remove <NEGATE_FLAG> tokens
            filter_upper (bool): Remove <UPPER_FLAG> tokens
            filter_punctuation (bool): Remove standalone punctuation
            filter_numeric (bool): Remove <NUMERIC> tokens
            filter_user_mentions (bool): Remove <USER_MENTION> tokens
            filter_url (bool): Remove URL_TOKEN tokens
            filter_retweet (bool): Remove <RETWEET> and proceeding ":" tokens.
            filter_stopwords (bool): Remove stopwords from nltk english stopword set
            keep_pronouns (bool): If removing stopwords, keep pronouns
            filter_empty (bool): Remove empty strings
            preserve_case (bool): Make all tokens lowercase if True.
            emoji_handling (str or None): If None, emojis are kept as they appear in the text. Otherwise,
                                          should be "replace" or "strip". If "replace", they are replaced
                                          with a generic "<EMOJI>" token. If "strip", they are removed completely.
            filter_hashtag (bool):
            strip_hashtag (bool):
            max_vocab_size (int): Maximum number of ngrams to keep in vocabulary.
            min_token_freq (int): Minimum frequency of occurence.
            max_token_freq (int): Maximum frequence of occurence.
            ngrams (tuple: (int, int)): (min_ngram, max_ngram).
            max_tokens_per_document (int or None): Only consider the first N tokens from a single document.
                                                   If None (default), will take all tokens.
            max_documents_per_user (int or None): Only consider the most recent N documents from a user. If
                                                  None (default), will take all documents.
            binarize_counter (bool): If True, multiple usages of an n-gram by a single user only count as 
                                     a single occurrence toward reaching the minimum/maximum usage thresholds.
            filter_mh_subreddits (None or str): If None, no filtering. Otherwise either "depression","rsdd","smhd", or "all"
            filter_mh_terms (None or str): If None, no filtering. Otherwise either "rsdd" or "smhd"
            keep_retweets (bool): If True (default), keeps retweets in processed data. Otherwise,
                                  ignores them in the user's data.
            external_vocab (list): Which vocabularies to include from external sources. 
                                   - LIWC should be input as "liwc"
                                   - GloVe should be input as "glove-DIM-SIZE", where DIM is the embedding
                                     dimension file to load and SIZE is the number of most frequent terms
                                     to load
                                   - Alternatives can be specified as a newline delimited list of n-grams. 
                                     Each token within a single n-gram should be delimited using a space. The
                                     vocabulary parameter ngrams will be overwritten if not in line with
                                     the external vocabulary.
            processor (object or None):
            processor_kwargs (dict): 
            random_state (int): Seed if any post-level sampling is done.
        """
        ## Vocabular Parameters
        self._loading_params = {
                        "filter_negate":filter_negate,
                        "filter_upper":filter_upper,
                        "filter_punctuation":filter_punctuation,
                        "filter_numeric":filter_numeric,
                        "filter_user_mentions":filter_user_mentions,
                        "filter_url":filter_url,
                        "filter_retweet":filter_retweet,
                        "filter_stopwords":filter_stopwords,
                        "emoji_handling":emoji_handling,
                        "filter_hashtag":filter_hashtag,
                        "strip_hashtag":strip_hashtag,
                        "keep_pronouns":keep_pronouns,
                        "preserve_case":preserve_case,
                        "max_tokens_per_document":max_tokens_per_document,
                        "max_documents_per_user":max_documents_per_user,
                        "filter_mh_subreddits":filter_mh_subreddits,
                        "filter_mh_terms":filter_mh_terms,
                        "keep_retweets":keep_retweets,
                        "lang":lang,
                        "random_state":random_state
        }
        self._max_vocab_size = max_vocab_size
        self._min_token_freq = min_token_freq
        self._max_token_freq = max_token_freq
        self._ngrams = ngrams
        self._external_ngrams = ngrams
        self._binarize_counter = binarize_counter
        self._external_only = external_only
        self._processor = processor
        self._processor_kwargs = processor_kwargs
        ## Workspace
        self._loader = LoadProcessedData(**self._loading_params)
        self.ngram_to_idx = dict()
        self.vocab = set()
        self._class_state = "untrained"
        ## Initialize Data Filter
        _ = self._initialize_data_filter()
        ## Initialize External Vocabulary
        _ = self._initialize_external_vocabulary(external_vocab)

    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        l1 = ", ".join("{}={}".format(x,y) for x,y in self._loading_params.items())
        l2 = "max_vocab_size={}, min_token_freq={}, max_token_freq={}, ngrams={}".format(
                    self._max_vocab_size if hasattr(self, "_max_vocab_size") else None,
                    self._min_token_freq if hasattr(self, "_min_token_freq") else 0,
                    self._max_token_freq if hasattr(self, "_max_token_freq") else None,
                    self._ngrams if hasattr(self, "_ngrams") else [1,1]
        )
        desc = "Vocabulary({}, {})".format(l1, l2)
        return desc
    
    def _initialize_external_vocabulary(self,
                                        external_vocab):
        """

        """
        ## Initialize Protected Set
        protected_set = {"liwc":set(),"glove":set()}
        ## Process External Vocabs
        for ev in external_vocab:
            if os.path.exists(ev):
                ev_vocab = []
                min_n, max_n = 1e10, -1
                with open(ev,"r") as the_file:
                    for line in the_file:
                        line_ngram = tuple(line.strip().split())
                        min_n = min(min_n, len(line_ngram))
                        max_n = max(max_n, len(line_ngram))
                        ev_vocab.append(line_ngram)
                protected_set[ev] = set(ev_vocab)
                self._external_ngrams = [min(self._external_ngrams[0], min_n), max(self._external_ngrams[1], max_n)]
        ## Assign Protected Set to Class Attribute
        self._protected_set = protected_set

    def _initialize_data_filter(self):
        """
        Create a loader with specifc params for ignoring certain
        tokens when creating n-grams.

        Args:
            None
        
        Returns:
            None
        """
        df_params = self._loading_params.copy()
        df_params["filter_negate"] = True
        df_params["filter_upper"] = True
        self._data_filter = LoadProcessedData(**df_params)

    def get_ordered_vocabulary(self):
        """
        Get the vocabulary ordered by it's determined index

        Args:
            None
        
        Returns:
            ordered_vocab (list): Vocabular terms ordered by it's learned index mapping
        """
        idx_rev = dict((y, x) for x, y in self.ngram_to_idx.items())
        ordered_vocab = list(map(lambda i: idx_rev[i], range(len(self.vocab))))
        return ordered_vocab
    
    def get_ngrams(self,
                   tokens,
                   min_n=1,
                   max_n=1):
        """
        Get n-gram tuples from a list of tokens

        Args:
            tokens (list): List of strings
            min_n (int): Minimum n-gram
            max_n (int): Maximum ngram
        
        Returns:
            all_ngrams (list of tuples): Ngram lists.
        """
        ## Check Inputs
        if min_n > max_n:
            raise ValueError("min_n must be less than max_n")
        if min_n == 0:
            raise ValueError("min_n should be greater than 0")
        ## Initialize N-Gram List
        all_ngrams = []
        ## Clean Input Tokens
        tt = "text_tokenized"
        cleaned_tokens = self._data_filter.filter_user_data([{tt:tokens}], tokens_only=True)
        if len(cleaned_tokens) > 0:
            cleaned_tokens = cleaned_tokens[0][tt]
        filtered_tokens = [i for i in tokens if i in self._data_filter.filter_set]
        ## Generate N-Gram Tuples
        for n in range(min_n, max_n+1):
            all_ngrams.extend(list(zip(*[cleaned_tokens[i:] for i in range(n)])))
        if len(filtered_tokens) > 0:
            all_ngrams.extend([(ft, ) for ft in filtered_tokens])
        return all_ngrams

    def _count_tokens(self,
                      token_lists,
                      ngrams=None):
        """
        Count n-grams present in a token list

        Args:
            tokens (list of lists): Lists of unigram tokens
        
        Returns:
            fn_counts (Counter): Token counts 
        """
        ## Check Input
        if ngrams is None:
            ngrams = self._ngrams
        ## Filter out empty lists
        fn_tokens = list(filter(lambda i: len(i) > 0, token_lists))
        ## Get N-grams from Tokens
        fn_ngrams = list(map(lambda t: self.get_ngrams(t, ngrams[0], ngrams[1]),
                             fn_tokens))
        ## Count Tokens (binary if class is unfit and binarization turned on)
        if hasattr(self, "_class_state") and \
           self._class_state == "untrained" and \
           hasattr(self, "_binarize_counter") and \
           self._binarize_counter:
            fn_counts = Counter(list(set(flatten(fn_ngrams))))
        else:
            fn_counts = Counter(flatten(fn_ngrams))
        return fn_counts
    
    def _load_and_count(self,
                        filename,
                        min_date=None,
                        max_date=None,
                        n_samples=None,
                        randomized=False,
                        return_post_counts=False,
                        resolution=None,
                        reference_point=None,
                        chunk_n_samples=None,
                        chunk_randomized=False,
                        return_references=False,
                        in_memory=True,
                        temp_dir=None,
                        ngrams=None):
        """
        Load preprocessed text data and count the token usage.

        Args:
            filename (str): Path to a preprocessed text data file for a user.
            min_date (datetime): Lower date bound
            max_date (datetime): Upper date bound
            n_samples (int): Number of Samples to Select (Post-date filtering)
            randomized (bool): If True, sample randomly instead of recent
            return_post_counts (bool): If True, returns number of posts as well
            resolution (None or str)
            reference_point (None or int)
            chunk_n_samples (None, int, or float)
            chunk_randomized (bool)
            return_references (bool)
            in_memory (bool)
            temp_dir (None or str)

        Returns:
            fn_counts (dict): Token counts in the user's data file.
        """
        ## Check Arguments
        if not in_memory and (return_post_counts or return_references):
            raise ValueError("Cannot return post counts and/or references when writing to disk.")
        if not in_memory and temp_dir is None:
            raise FileNotFoundError("Asked to use disk memory but didn't include temp directory.")
        if ngrams is None:
            ngrams = self._ngrams
        ## Load User Data
        fn_data = self._loader.load_user_data(filename,
                                              min_date=min_date,
                                              max_date=max_date,
                                              n_samples=n_samples,
                                              randomized=randomized,
                                              processor=self._processor if hasattr(self,"_processor") else None,
                                              processor_kwargs=self._processor_kwargs if hasattr(self,"_processor_kwargs") else {})
        ## Chunk Data by User
        fn_data = list(filter(lambda f: f.get("user_id_str") is not None, fn_data))
        fn_users = sorted(set([i.get("user_id_str") for i in fn_data]))
        n_posts_by_user = []
        counts_by_user = []
        refs_by_user = []
        for u in fn_users:
            ## Isolate Users Data
            fn_user_data = list(filter(lambda i: i.get("user_id_str")==u, fn_data))
            ## Chunk Data by Time
            fn_user_data = self._loader.chunk_user_data(fn_user_data,
                                                        resolution=resolution,
                                                        reference_point=reference_point,
                                                        n_samples=chunk_n_samples,
                                                        randomized=chunk_randomized)
            ## Analyze Each Chunk Separately
            n_posts = []
            counts = []
            refs = []
            for fn_ref, fn in fn_user_data:
                ## Count Number of Posts
                fn_nposts = len(fn)
                ## Early Return for Zero Data
                if fn_nposts == 0:
                    counts.append(Counter())
                    if return_post_counts:
                        n_posts.append(fn_nposts)
                    if return_references:
                        refs.append(fn_ref)
                    continue
                ## Identify Tokens
                fn_tokens = [i["text_tokenized"] for i in fn]
                ## Get Counts
                fn_counts = self._count_tokens(fn_tokens, ngrams=ngrams)
                ## Caching
                counts.append(fn_counts)
                if return_post_counts:
                    n_posts.append(fn_nposts)
                if return_references:
                    refs.append(fn_ref)
            ## Cache
            counts_by_user.append((u,counts))
            if return_post_counts:
                n_posts_by_user.append((u,n_posts))
            if return_references:
                refs_by_user.append((u,refs))
        ## Write if not using Memory
        if not in_memory:
            ## Flatten
            all_counts_flat = Counter()
            for _, user_counts in counts_by_user:
                user_counts_flat = sum(user_counts, Counter())
                user_counts_flat = Counter(dict(("\t\t".join(x), y) for x, y in user_counts_flat.items()))
                all_counts_flat += user_counts_flat
            ## Write
            temp_file = f"{temp_dir}/{str(uuid4())}.json".replace("//","/")  
            with open(temp_file,"w") as the_file:
                json.dump(all_counts_flat, the_file)
            return temp_file
        else:
            ## Prepare for Return
            if not return_post_counts and not return_references:
                return counts_by_user
            elif return_post_counts and return_references:
                return counts_by_user, n_posts_by_user, refs_by_user
            elif return_post_counts:
                return counts_by_user, n_posts_by_user
            elif return_references:
                return counts_by_user, refs_by_user

    def assign(self,
               ngrams):
        """
        Assign a vocabulary (e.g. n-gram list) to vocabulary class instead of 
        learning from scratch.

        Args:
            ngrams (list): List of strings. If n-grams for n > 1, should already
                           be tuples
        
        Returns:
            self: Sets vocabulary attributes in place, returns class
        """
        ## Update Format of N-Grams
        ngrams = list(map(lambda n: (n, ) if not isinstance(n, tuple) else n, ngrams))
        min_n, max_n = min(map(len, ngrams)), max(map(len, ngrams))
        self._ngrams = [min_n, max_n]
        ## Assign Class Attributes
        self.vocab = set(ngrams)
        self._vocab_count = Counter(ngrams)
        self.ngram_to_idx = dict((ngram, ind) for ind, ngram in enumerate(ngrams))
        self._class_state = "trained"
        return self

    def _keep_vocab_item(self,
                         word,
                         count,
                         min_count,
                         max_count,
                         trim_rule=None):
        """
        Should we keep `word` in the vocab or remove it? Function based largely
        on gensim implemtation with additional max_count filtering
        
        Args:
            word (tuple): N-gram Input
            count (int): Number of times that word appeared in a corpus.
            min_count (int): Discard words with frequency smaller than this.
            max_count (int): Discard words with frequency greater than this 
            trim_rule (function, optional): Custom function to decide whether to keep or discard this word.
                If a custom `trim_rule` is not specified, the default behaviour is simply `count >= min_count`.
        
        Returns:
            keep_vocab (bool): True if `word` should stay, False otherwise.
        """
        ## Check Protected
        for _, terms in self._protected_set.items():
            if word[0].lower() in terms:
                return True
            if word in terms:
                return True
        ## Check Exclusion
        if hasattr(self, "_exclusion_list") and word in self._exclusion_list:
            return False
        ## Check Count (If No Trim Rule Specified, Use Default Check)
        if trim_rule is None:
            above_thresh = count >= min_count if min_count is not None else True
            below_thresh = count <= max_count if max_count is not None else True
            if not below_thresh and hasattr(self, "_exclusion_list"):
                self._exclusion_list.add(word)
            return above_thresh and below_thresh
        else:
            return trim_rule(word, count, min_count, max_count)

    def _prune_vocab(self,
                     counter,
                     n_total_files,
                     n_seen_files,
                     min_count=None,
                     max_count=None,
                     trim_rule=None):
        """
        Filter out vocabulary terms that are below or above class frequency thresholds

        Args:
            counter (Counter): ngram to count mapping
            n_total_files (int): Number of files used for learning the full vocabulary
            n_seen_files (int): Number of files seen for learning the current vocabulary
            min_count (int or None): Specify exact threshold. If None, falls back to class _min_token_freq - margin
            max_count (int or None): Specify exact threshold. If None, falls back to class _max_token_freq + margin
            trim_rule (function or None): Custom rule that returns whether a word is kept (True) or not. Accepts
                                          word, count, min_count, and max_count arguments
        
        Returns:
            counter_pruned (Counter): Prunced vocabulary
        """
        ## Initialize Exclusion List
        if not hasattr(self, "_exclusion_list"):
            self._exclusion_list = set()
        ## Compute Thresholds (With Margin)
        if min_count is None:
            min_count = None if self._min_token_freq == 0 else math.ceil(self._min_token_freq * 0.8 * n_seen_files / n_total_files)
        if max_count is None:
            max_count = None if self._max_token_freq is None else math.ceil(self._max_token_freq * 1.2 * n_seen_files / n_total_files)
        ## Get Acceptable Terms
        accept = set()
        for ngram, count in counter.items():
            keep = self._keep_vocab_item(ngram,
                                         count,
                                         min_count=min_count,
                                         max_count=max_count,
                                         trim_rule=trim_rule)
            if keep:
                accept.add(ngram)
        ## Update and Return Count
        counter_pruned = Counter(dict((n, c) for n, c in counter.items() if n in accept))
        return counter_pruned
        
    def fit(self,
            filenames,
            chunksize=10,
            jobs=8,
            min_date=None,
            max_date=None,
            n_samples=None,
            randomized=False,
            resolution=None,
            reference_point=None,
            chunk_n_samples=None,
            chunk_randomized=False,
            prune=False,
            prune_freq=1,
            prune_min_count=None,
            prune_max_count=None,
            prune_trim_rule=None,
            in_memory=True,
            temp_dir=None):
        """
        Learn a vocabulary from preprocessed data files (or assign
        external vocabulary)

        Args:
            filenames (list): List of preprocessed data files containing
                              "text_tokenized" lists.
            chunksize (int): Number of filenames to process in a multiprocessing
                             chunk before combining into the main counter object.
            jobs (int): Number of processes to use
            min_date (str, datetime, or None): Lower bound in time for learning vocabulary
            max_date (str, datetime, or None): Upper bound in time for learning vocabulary
            n_samples (int or None): Number of Samples to Select (Post-date filtering). All if none.
            randomized (bool): If True, sample randomly instead of recent
            resolution (None or str):
            reference_point (None or str):
            prune (bool): Whether to prune vocabulary during learning procedure
            prune_freq (int): How many chunks to go before pruning again (if desired)
            prune_min_count (int or None): If not specified, will use logic based on class parameters to set
            prune_max_count (int or None): If not specified, will use logic based on class parameters to set
            prune_trim_rule (function or None): Special pruning rule takes word, count, min_count arguments
            in_memory (bool): Learn the vocabulary and cache counts in RAM. Otherwise, write to temp
            files and aggregate after
            temp_dir (str or None): If desired, pre-specify temp directory for out-of-memory caching

        Returns:
            self
        """
        ## Assign Class State
        self._class_state = "untrained"
        ## Initialize Counter
        vocab_counter = Counter()
        ## Filtered Vocab List
        filtered_vocab = []
        ## Reference Point
        if reference_point is not None and isinstance(reference_point, str):
            reference_point = int(pd.to_datetime(reference_point).timestamp())
        ## Learn From Data
        if not self._external_only:
            ## Temporary Directory
            if not in_memory:
                if temp_dir is None:
                    temp_dir = f"temp_vocab_{str(uuid4())}/"
                if not os.path.exists(temp_dir):
                    _ = os.makedirs(temp_dir)
            ## Date Boundaries
            if min_date is not None and isinstance(min_date,str):
                min_date = pd.to_datetime(min_date)
            if max_date is not None and isinstance(max_date,str):
                max_date = pd.to_datetime(max_date)
            ## Chunk Files
            if chunksize is None:
                chunksize = len(filenames)
            file_chunks = list(chunks(filenames, chunksize))
            ## Initialize Multiprocessing
            with Pool(processes=jobs) as mp:
                ## Count Tokens Across Files
                loader = partial(self._load_and_count,
                                min_date=min_date,
                                max_date=max_date,
                                n_samples=n_samples,
                                randomized=randomized,
                                resolution=resolution,
                                reference_point=reference_point,
                                chunk_n_samples=chunk_n_samples,
                                chunk_randomized=chunk_randomized,
                                return_references=False,
                                return_post_counts=False,
                                in_memory=in_memory,
                                temp_dir=temp_dir,
                                ngrams=self._ngrams)
                for f, fn_chunk in tqdm(enumerate(file_chunks),
                                    desc="Counting N-Grams",
                                    total=len(file_chunks),
                                    file=sys.stdout):
                    ## Run Counter
                    mp_counts = list(mp.map(loader, fn_chunk))
                    ## Process Chunk Counts
                    if in_memory:
                        mp_counts = flatten(mp_counts)
                        for fn_user, fn_counts in mp_counts:
                            if isinstance(fn_counts, Counter):
                                vocab_counter += fn_counts
                            elif isinstance(fn_counts, list):
                                for c in fn_counts:
                                    vocab_counter += c
                            else:
                                raise ValueError("Data type not recognized.")
                    else:
                        ## Load Cached Files
                        for file in mp_counts:
                            with open(file, "r") as the_file:
                                file_counts = json.load(the_file)
                            file_counts = Counter(dict((tuple(x.split("\t\t")), y) for x, y in file_counts.items()))
                            vocab_counter += file_counts
                    if prune and f > 0 and (f + 1) % prune_freq == 0:
                        vocab_counter = self._prune_vocab(vocab_counter,
                                                        len(filenames),
                                                        (f + 1) * chunksize,
                                                        min_count=prune_min_count,
                                                        max_count=prune_max_count,
                                                        trim_rule=prune_trim_rule)
            ## Prune By Frequency
            vocab_counter = self._prune_vocab(vocab_counter,
                                              len(filenames),
                                              len(filenames),
                                              self._min_token_freq,
                                              self._max_token_freq)
            ## Vocab Filtering (Max Vocab Size)
            if not hasattr(self, "_max_vocab_size") or self._max_vocab_size is None:
                for i in vocab_counter.most_common():
                    filtered_vocab.append(i)
            else:
                for i in vocab_counter.most_common(self._max_vocab_size):
                    filtered_vocab.append(i)
            ## Remove Temp Directory
            if not in_memory:
                _ = os.system("rm -rf {}".format(temp_dir))
        ## Append External Vocabs
        if self._max_token_freq is None and len(vocab_counter) > 0:
            max_freq = vocab_counter.most_common(1)[0][1]
        else:
            max_freq = 1e10
        for key, terms in self._protected_set.items():
            if not terms:
                continue
            if key in ["glove","liwc"]:
                new_terms = [((w, ), 1) for w in terms if (w, ) not in vocab_counter or \
                                                                vocab_counter[(w, )] < self._min_token_freq or \
                                                                vocab_counter[(w, )] > max_freq]
            elif isinstance(key, str) and os.path.exists(key):
                new_terms = [(w, 1) for w in terms if w not in vocab_counter or \
                                                               vocab_counter[w] < self._min_token_freq or \
                                                               vocab_counter[w] > max_freq]
            else:
                raise ValueError("Protected set key not recognized.")
            filtered_vocab.extend(new_terms)
            for w, c in new_terms:
                vocab_counter[w] += c
        ## Maintain Counter as a Class Attribute
        self._vocab_count = vocab_counter
        ## Construct Vocabulary Objects
        self.vocab = set(list(map(lambda i: i[0], filtered_vocab)))
        self.ngram_to_idx = dict((ngram, ind) for ind, ngram in enumerate(sorted(self.vocab)))
        ## Update Class State
        self._class_state = "trained"
        return self