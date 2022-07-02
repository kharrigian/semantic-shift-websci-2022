
"""
Data loaders for outputs generated from the preprocessing pipeline.

Example Usage of Streams:
```
## Imports
from glob import glob
from semshift.model.data_loaders import PostStream, PostStreamPipeline

## Identify Some Processed Files
filenames = sorted(glob("./data/processed/twitter/qntfy/*.tweets.tar.gz"))
filenames = filenames[:10]

## Initialize Streams
stream_2011 = PostStream(filenames,
                         min_date="2011-01-01",
                         max_date="2011-12-31",
                         mode=0,
                         metadata=["created_utc"],
                         jobs=8)
stream_2012 = PostStream(filenames,
                         min_date="2012-01-01",
                         max_date="2012-12-31",
                         metadata=["created_utc"],
                         mode=0,
                         jobs=8)

## Combine into a Pipeline
stream_pipeline = PostStreamPipeline(stream_2011, stream_2012)
assert len(stream_2011) + len(stream_2012) == len(stream_pipeline)

## Load Data
data_2011 = [i[0] for i in stream_2011]
data_2012 = [i[0] for i in stream_2012]
data_pipeline = [i[0] for i in stream_pipeline]
assert set(data_2011) & set(data_2012) == set()
assert set(data_2011) | set(data_2012) == set(data_pipeline)
```
"""

#################
### Imports
#################

## Standard Libary
import os
import sys
import json
import gzip
from uuid import uuid4
from datetime import datetime
from multiprocessing import Pool

## External Libraries
import emoji
import pandas as pd
import numpy as np
from langid import langid
from tqdm import tqdm

## Local Modules
from ..preprocess.preprocess import tokenizer, sent_tokenize
from ..preprocess.tokenizer import (STOPWORDS,
                                    PRONOUNS,
                                    CONTRACTIONS,
                                    PUNCTUATION)
from ..util.helpers import flatten, chunks
from ..util.logging import initialize_logger

#################
### Globals
#################

## Mental Health Subreddit/Term Filters
RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../data/resources/"
MH_SUBREDDIT_FILE = f"{RESOURCE_DIR}mh_subreddits.json"
MH_TERMS_FILE = f"{RESOURCE_DIR}mh_terms.json"

## Load Mental Health Subreddits
with open(MH_SUBREDDIT_FILE, "r") as the_file:
    MH_SUBREDDITS = json.load(the_file)

## Load Mental Health Terms
with open(MH_TERMS_FILE, "r") as the_file:
    MH_TERMS = json.load(the_file)

## Emojis
if "en" in emoji.UNICODE_EMOJI:
    EMOJI_DICT = emoji.UNICODE_EMOJI["en"]
else:
    EMOJI_DICT = emoji.UNICODE_EMOJI

## Logging
LOGGER = initialize_logger()

## Arguments for Data Loading
DEFAULT_LOADER_KWARGS = {
    "filter_negate":True,
    "filter_upper":True,
    "filter_punctuation":True,
    "filter_numeric":True,
    "filter_user_mentions":True,
    "filter_url":True,
    "filter_retweet":True,
    "filter_stopwords":False,
    "keep_pronouns":True,
    "filter_empty":True,
    "preserve_case":False,
    "emoji_handling":None,
    "filter_hashtag":False,
    "strip_hashtag":False,
    "max_tokens_per_document":None,
    "max_documents_per_user":None,
    "filter_mh_subreddits":None,
    "filter_mh_terms":None,
    "lang":None,
    "stopwords":set(),
    "random_state":42
}

#################
### Helpers
#################

def pattern_match(string,
                  patterns):
    """
    Check to see if any substring in a list
    of patterns matches to a given string

    Args:
        string (str): Input string
        patterns (iterable): Possible patterns
    
    Returns:
        match (bool): Whether or not a match exists
    """
    string_lower = string.lower()
    for p in patterns:
        if p in string_lower:
            return True
    return False

####################
### Classes
####################

class LoadProcessedData(object):

    """
    Generic Data Loading Class, for pre-processed data.
    """

    ## Loading Globals
    _lang_identifier_loaded = False

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
                 filter_empty=True,
                 emoji_handling=None,
                 filter_hashtag=False,
                 strip_hashtag=False,
                 max_tokens_per_document=None,
                 max_documents_per_user=None,
                 filter_mh_subreddits=None,
                 filter_mh_terms=None,
                 keep_retweets=True,
                 lang=None,
                 stopwords=STOPWORDS,
                 random_state=42):
        """
        Generic Data Loading Class

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
            preserve_case (bool): Keep token case as is. Otherwise, make lowercase.
            filter_empty (bool): Remove empty strings
            filter_hashtag (bool): If True and data loader encounters a hashtag, it will remove it
            strip_hashtag (bool): If True (default) and data loader encounters a hashtag with
                                  filter_hashtag set False, it will remove a hashtag prefix.
            emoji_handling (str or None): If None, emojis are kept as they appear in the text. Otherwise,
                                          should be "replace" or "strip". If "replace", they are replaced
                                          with a generic "<EMOJI>" token. If "strip", they are removed completely.
            max_tokens_per_document (int or None): Only consider the first N tokens from a single document.
                                                   If None (default), will take all tokens.
            max_documents_per_user (int or None): Only consider the most recent N documents from a user. If
                                                  None (default), will take all documents.
            filter_mh_subreddits (None or str): If None, no filtering. Otherwise either "depression","rsdd","smhd", or "all"
            filter_mh_terms (None or str): If None, no filtering. Otherwise either "rsdd" or "smhd"
            keep_retweets (bool): If True (default), keeps retweets in processed data. Otherwise,
                                  ignores them in the user's data.
            lang (None or list of str): If not None, languages to accept based on langid
            random_state (int): Seed to use for any random sampling
        """
        ## Class Arguments
        self.filter_negate = filter_negate
        self.filter_upper = filter_upper
        self.filter_punctuation = filter_punctuation
        self.filter_numeric = filter_numeric
        self.filter_user_mentions = filter_user_mentions
        self.filter_url = filter_url
        self.filter_retweet = filter_retweet
        self.filter_stopwords = filter_stopwords
        self.keep_pronouns = keep_pronouns
        self.preserve_case = preserve_case
        self.filter_empty = filter_empty
        self.emoji_handling = emoji_handling
        self.filter_hashtag = filter_hashtag
        self.strip_hashtag = strip_hashtag
        self.filter_mh_subreddits = filter_mh_subreddits
        self.filter_mh_terms = filter_mh_terms
        self.max_tokens_per_document = max_tokens_per_document
        self.max_documents_per_user = max_documents_per_user
        self.keep_retweets = keep_retweets
        self.lang = lang
        self.random_state = random_state
        ## Helpful Variables
        self._punc = set()
        if self.filter_punctuation:
            self._punc = set(PUNCTUATION)
        ## Initialize Language Identifier (if relevant)
        if self.lang is not None:
            ## Set of Languages
            self.lang = set(self.lang)
            ## Load Classification Model from Disk
            if not LoadProcessedData._lang_identifier_loaded:
                ## Update Attribute
                LoadProcessedData._lang_identifier_loaded = True
                ## Load Classification Model
                _ = langid.load_model()
        ## Initialization Processes
        self._initialize_filter_set()
        self._initialize_stopwords(stopwords)
        self._initialize_mh_subreddit_filter()
        self._initialize_mh_terms_filter()
    
    def _initialize_mh_subreddit_filter(self):
        """
        Helper. Initialize the mental-health subreddit filter
        set.

        Args:
            None
        
        Returns:
            None. Sets _ignore_subreddits attribute
        """
        ## Initalize Set of Subreddits to Ignore
        self._ignore_subreddits = set()
        ## Break If No Filtering Specified
        if not hasattr(self, "filter_mh_subreddits") or self.filter_mh_subreddits is None:
            return
        ## Check That Filter Exists
        if self.filter_mh_subreddits not in MH_SUBREDDITS:
            raise KeyError(f"Mental Health Subreddit Filter `{self.filter_mh_subreddits}` not found.")
        ## Update Ignore Set
        self._ignore_subreddits = set([i.lower() for i in MH_SUBREDDITS[self.filter_mh_subreddits]])
    
    def _classify_languages(self,
                            text,
                            max_chunksize=250):
        """
        
        """
        ## Input Check
        if len(text) == 0:
            return []
        ## Format As List
        is_singular = False
        if isinstance(text, str):
            is_singular = True
            text = [text]
        ## Initialize Result Cache
        all_preds = []
        all_conf = []
        for text_chunk in chunks(text, max_chunksize):
            ## Get Feature Space
            features = np.vstack(list(map(langid.identifier.instance2fv, text_chunk)))
            ## Get Probs
            probs = langid.identifier.nb_classprobs(features)
            cl = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            preds = [langid.identifier.nb_classes[c] for c in cl]
            ## Cache
            all_preds.extend(preds)
            all_conf.extend(conf)
        ## Format
        results = list(zip(all_preds, all_conf))
        if is_singular:
            results = results[0]
        return results

    def _initialize_mh_terms_filter(self,
                                    use_mh=True,
                                    use_pos=False,
                                    use_neg=False):
        """
        Helper. Initialize the list of terms to include
        in a set for filtering out posts. Consults
        the class initialization parameter filter_mh_terms
        for choosing the set of terms

        Args:
            use_mh (bool): If True, include the "terms" from
                           the mental-health term dictionary
            use_pos (bool): If True, include positive diagnosis
                           patterns from the dictionary
            use_neg (bool): If True, include negative diagnosis
                           patterns from the dictionary
        
        Returns:
            None, sets _ignore_terms attribute
        """
        ## Initialize Set of Terms to Ignore
        self._ignore_terms = set()
        ## Break if No Filtering Specified
        if not hasattr(self, "filter_mh_terms") or self.filter_mh_terms is None:
            return
        ## Check That Filter Exists
        if self.filter_mh_terms not in MH_TERMS["terms"]:
            raise KeyError(f"Mental Health Term Filter `{self.filter_mh_terms}` not found.")
        ## Construct Patterns
        all_patterns = []
        if use_mh:
            all_patterns.extend(MH_TERMS["terms"][self.filter_mh_terms])
        psets = []
        if use_pos:
            psets.append(MH_TERMS["pos_patterns"][self.filter_mh_terms])
        if use_neg:
            psets.append(MH_TERMS["neg_patterns"][self.filter_mh_terms])
        for pset in psets:
            for p in pset:
                if "_" in p:
                    p_sep = p.split()
                    n = len(p_sep) - 1
                    exp_match = [i for i, j in enumerate(p_sep) if j.startswith("_")][0]
                    exp_match_fillers = MH_TERMS["expansions"][p_sep[exp_match].rstrip(",")]
                    for emf in exp_match_fillers:
                        if p_sep[exp_match].endswith(","):
                            emf += ","
                        if exp_match != n:
                            emf_pat = " ".join(p_sep[:exp_match] + [emf] + p_sep[min(n, exp_match+1):])
                        else:
                            emf_pat = " ".join(p_sep[:exp_match] + [emf])
                        all_patterns.append(emf_pat)
                else:
                    all_patterns.append(p)
        self._ignore_terms = set(all_patterns)

    def _initialize_stopwords(self,
                              stopwords):
        """
        Initialize stopword set and removes pronouns if desired.

        Args:
            None
        
        Returns:
            None
        """
        ## Format Stopwords into set
        if hasattr(self, "filter_stopwords") and self.filter_stopwords:
            self.stopwords = stopwords
            if not isinstance(self.stopwords, set):
                self.stopwords = set(self.stopwords)
        else:
            self.stopwords = set()
            return
        ## Contraction Handling
        self.stopwords = self.stopwords | set(self._expand_contractions(list(self.stopwords)))
        ## Pronoun Handling
        if hasattr(self, "keep_pronouns") and self.keep_pronouns:
            for pro in PRONOUNS:
                if pro in self.stopwords:
                    self.stopwords.remove(pro)

    def _strip_emojis(self,
                      tokens):
        """
        Remove emojis from a list of tokens

        Args:
            tokens (list): Tokenized text
        
        Returns:
            tokens (list): Input list without emojis
        """
        tokens = list(filter(lambda t: t not in EMOJI_DICT and t != "<EMOJI>", tokens))
        return tokens
    
    def _replace_emojis(self,
                        tokens):
        """
        Replace emojis with generic <EMOJI> token

        Args:
            tokens (list): Tokenized text
        
        Returns:
            tokens (list): Tokenized text with emojis replaced with generic token
        """
        tokens = list(map(lambda t: "<EMOJI>" if t in EMOJI_DICT else t, tokens))
        return tokens

    def _strip_hashtags(self,
                        tokens):
        """
        
        """
        tokens = list(map(lambda t: t.replace("<HASHTAG=","")[:-1] if t.startswith("<HASHTAG=") else t, tokens))
        return tokens
    
    def _remove_hashtags(self,
                         tokens):
        """

        """
        tokens = list(filter(lambda t: not t.startswith("<HASHTAG="), tokens))
        return tokens

    def _expand_contractions(self,
                             tokens):
        """
        Expand English contractions.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now with expanded contractions.
        """
        tokens = \
        flatten(list(map(lambda t: CONTRACTIONS[t.lower()].split() if t.lower() in CONTRACTIONS else [t],
                         tokens)))
        return tokens
    
    def _select_n_recent_documents(self,
                                   user_data):
        """
        Select the N most recent documents in user data based on the 
        class initialization parameters.

        Args:
            user_data (list of dict): Processed user data dictionaries
        
        Returns:
            user_data (list of dict): N most recent documents
        """
        ## Downsample Documents
        user_data = sorted(user_data, key = lambda x: x["created_utc"], reverse = True)
        if hasattr(self, "max_documents_per_user") and self.max_documents_per_user is not None:
            user_data = user_data[:min(len(user_data), self.max_documents_per_user)]
        return user_data
    
    def _select_documents_in_date_range(self,
                                        user_data,
                                        min_date=None,
                                        max_date=None):
        """
        Isolate documents in a list of processed user_data
        that fall into a given date range

        Args:
            user_date (list of dict): Processed user data, includes
                                      created_utc key
            min_date (datetime or None): Lower date boundary
            max_date (datetime or None): Upper date boundary
        
        Returns:
            user_data (list of dict): Filtered user data
        """
        ## No additional filtering
        if min_date is None and max_date is None:
            return user_data
        ## Retrive Filtered Data
        filtered_data = []
        for u in user_data:
            tstamp = datetime.fromtimestamp(u["created_utc"])
            if min_date is not None and tstamp < min_date:
                continue
            if max_date is not None and tstamp > max_date:
                continue
            filtered_data.append(u)
        return filtered_data
    
    def _select_documents_randomly(self,
                                   user_data,
                                   n_samples=None):
        """
        Select a random sample of documents from an input
        user_data list

        Args:
            user_data (list of dict): Processed user data
            n_samples (int, float, str, or None): Max number of samples to select. If a float < 1,
                                                  will sample percentage of user data available.
        
        Returns:
            random_sample (list of dict): Sampled user data
        """
        ## Null Samples
        if n_samples is None:
            return user_data
        ## String Processing
        if isinstance(n_samples, str):
            if n_samples.startswith("discrete-"):
                n_samples = n_samples.split("discrete-")[1].split("-")
                if len(n_samples) != 2:
                    raise ValueError("String 'n_samples' with discrete setting is misconfigured. Should be 'discrete-<NSPLITS>-SPLIT1,SPLIT2'")
                try:
                    n_samples = (int(n_samples[0]), set(map(int, n_samples[1].split(","))))
                except Exception as e:
                    raise e
            else:
                raise ValueError("String parameterization of 'n_samples' is misconfigured.")
        ## Improper Spec
        if not isinstance(n_samples, tuple) and n_samples <= 0:
            raise ValueError("Cannot specify n_samples <= 0")
        ## Reset Random State
        if hasattr(self, "random_state"):
            seed = np.random.RandomState(self.random_state)
        else:
            seed = np.random.RandomState()
        ## Sample Documents Without Replacement
        if not isinstance(n_samples, tuple):
            ## Proportion of Data (if desired)
            if n_samples < 1:
                n_samples = int(len(user_data) * n_samples)
            ## Sample Indices
            sample_ind = sorted(list(seed.choice(len(user_data),
                                                 size=min(len(user_data), n_samples),
                                                 replace=False)))
        elif isinstance(n_samples, tuple):
            ## Assign Each Instance to a Bin
            bins = seed.choice(n_samples[0], len(user_data), replace=True)
            ## Isolate Indices in Desired Bin
            sample_ind = [i for i, b in enumerate(bins) if b in n_samples[1]]
        ## Get Subset
        random_sample = [user_data[r] for r in sample_ind]
        return random_sample

    def _parse_date_frequency(self,
                              freq):
        """
        Convert str-formatted frequency into seconds. Base frequencies
        include minutes (m), hours (h), days (d), weeks (w),
        months (mo), and years (y).
        Args:
            freq (str): "{int}{base_frequency}"
        
        Returns:
            period (int): Time in seconds associated with frequency
        """
        ## Frequencies in terms of seconds
        base_freqs = {
            "m":60,
            "h":60 * 60,
            "d":60 * 60 * 24,
            "w":60 * 60 * 24 * 7,
            "mo":60 * 60 * 24 * 31,
            "y":60 * 60 * 24 *  365
        }
        ## Parse String
        freq = freq.lower()
        freq_ind = 0
        while freq_ind < len(freq) - 1 and freq[freq_ind].isdigit():
            freq_ind += 1
        mult = 1
        if freq_ind > 0:
            mult = int(freq[:freq_ind])
        base_freq = freq[freq_ind:]
        if base_freq not in base_freqs:
            raise ValueError("Could not parse frequency.")
        period = mult * base_freqs.get(base_freq)
        return period
        
    def chunk_user_data(self,
                        user_data,
                        resolution=None,
                        reference_point=None,
                        n_samples=None,
                        randomized=False):
        """
        Args:
            user_data (list of dict):
            resolution (None or str):
            reference_point (None or int): Optionally the starting UTC
        """
        ## Option 1: No Chunking
        if not resolution:
            return [((user_data[0].get("created_utc"), user_data[-1].get("created_utc")), user_data)]
        ## Option 2: Post Level
        if resolution == "post":
            user_data = [((u.get("created_utc",i),u.get("created_utc",i)),[u]) for i,u in enumerate(user_data)]
        ## Option 3: Temporal
        else:
            ## Information
            n = len(user_data)
            ## Reverse Sort (So that data is ascending in time)
            user_data = user_data[::-1]
            ## Identify Time in Seconds Between Groups
            time_chunksize = self._parse_date_frequency(resolution)
            ## Get Reference Point
            if reference_point is None and n > 0:
                min_date = datetime.fromtimestamp(user_data[0]["created_utc"])
                reference_point = int(datetime(min_date.year, 1, 1).timestamp())
            ## Apply Chunking Procedure
            new_user_data = []
            current = []
            j = 0
            lower_bound_reached = False
            while j < n:
                ## Get Time Stamp
                jutc = user_data[j]["created_utc"]
                ## Check to See if Reference Point Reached Yet
                if not lower_bound_reached:
                    ## Move On If Still Below Reference Point
                    if jutc < reference_point:
                        j += 1
                        continue
                    else:
                        lower_bound_reached = True
                ## Update Cache
                change = False
                while reference_point + time_chunksize <= jutc:
                    reference_point += time_chunksize
                    change = True
                if change and len(current) > 0:
                    new_user_data.append(((current[0].get("created_utc"), current[-1].get("created_utc")), current))
                    current = []
                current.append(user_data[j])
                j += 1
            if len(current) > 0:
                new_user_data.append(((current[0].get("created_utc"), current[-1].get("created_utc")),current))
            ## Reverse Sort Again
            user_data = new_user_data[::-1]
        ## Sampling
        if n_samples is not None:
            if hasattr(self, "random_state"):
                seed = np.random.RandomState(self.random_state)
            else:
                seed = np.random.RandomState()
            if n_samples <= 0:
                raise ValueError("Chose sample size <= 0")
            if n_samples < 1:
                n_samples = int(len(user_data) * n_samples)
            if randomized:
                sample_ind = sorted(seed.choice(len(user_data),
                                                size=min(len(user_data), n_samples),
                                                replace=False))
                user_data = [user_data[i] for i in sample_ind]
            else:
                user_data = user_data[:min(n_samples, len(user_data))]
        return user_data

    def load_user_data(self,
                       filename,
                       min_date=None,
                       max_date=None,
                       n_samples=None,
                       randomized=False,
                       processor=None,
                       processor_kwargs={},
                       apply_filter=True,
                       ignore_text_filter=False,
                       ignore_sentence_filter=False):
        """
        Load user data from disk. Default works with
        preprocessed, but can load raw data and preprocess

        Args:
            filename (str): Path to .tar.gz file. Pre-processed
                            user data
            min_date (datetime or None): Lower date boundary
            max_date (datetime or None): Upper date boundary
            n_samples (int or None): Number of samples to consider
            randomized (bool): If sampling and set True, use a random
                               sample instead of the most recent posts
            processor (str or None): None, "twitter", or "reddit"
            processor_kwargs (dict): Arguments passed to processor.load_and_process

        Returns:
            user_data (list of dict): Preprocessed, filtered user
                                      data
        """
        ## Load the Preprocessed GZIPed file
        if processor is None:
            opener = gzip.open if filename.endswith(".gz") else open
            with opener(filename,"r") as the_file:
                ## Load
                user_data = []
                for line in the_file:
                    user_data.append(json.loads(line))
                ## Handle Different Types of Storage Format (List vs. Newline Delimited)
                if len(user_data) > 0 and isinstance(user_data[0], list):
                    user_data = user_data[0]
        ## Load Raw Data and Process
        else:
            user_data = processor.load_and_process(filename,
                                                   **processor_kwargs)
        ## Data Amount Filtering
        user_data = self._select_n_recent_documents(user_data)
        ## Date-based Filtering
        user_data = self._select_documents_in_date_range(user_data,
                                                         min_date,
                                                         max_date)
        ## Post-level Sampling
        if n_samples is not None:
            if randomized:
                user_data = self._select_documents_randomly(user_data,
                                                            n_samples)
            else:
                if n_samples < 1:
                    n_samples = int(len(user_data) * n_samples)
                user_data = user_data[:min(len(user_data), n_samples)]
        ## Apply Processing
        if apply_filter:
            user_data = self.filter_user_data(user_data,
                                              tokens_only=False,
                                              ignore_text=ignore_text_filter,
                                              ignore_sentences=ignore_sentence_filter)
        return user_data
    
    def load_user_metadata(self,
                           filename):
        """
        Load user metedata file

        Args:
            filename (str): Path to .tar.gz file
        
        Returns:
            user_data (dict): User metadata dictionary
        """
        ## Load the GZIPed file
        with gzip.open(filename) as the_file:
            user_data = json.load(the_file)
        return user_data
    
    def _filter_in(self,
                   obj,
                   ignore_set):
        """
        Filter a list by excluding matches in a set

        Args:
            obj (list): List to be filtered
            ignore_set (iterable): Set to check items again
        
        Returns:
            filtered_obj (list): Original list excluding objects
                          found in the ignore_set
        """
        return list(filter(lambda l: l not in ignore_set, obj))
    
    def _initialize_filter_set(self):
        """
        Initialize the set of items to filter from
        a tokenized text list based on class initialization
        parameters.

        Args:
            None
        
        Returns:
            None, assigns self.filter_set attribute
        """
        ## Initialize SEt
        self.filter_set = set()
        if hasattr(self,"filter_negate") and self.filter_negate:
            self.filter_set.add("<NEGATE_FLAG>")
        ## Filter Upper
        if hasattr(self,"filter_upper") and self.filter_upper:
             self.filter_set.add("<UPPER_FLAG>")
        ## Filter Numeric
        if hasattr(self,"filter_numeric") and self.filter_numeric:
           self.filter_set.add("<NUMERIC>")
        ## Filter User Mentions
        if hasattr(self,"filter_user_mentions") and self.filter_user_mentions:
            self.filter_set.add("<USER_MENTION>")
        ## Filter URL
        if hasattr(self,"filter_url") and self.filter_url:
            self.filter_set.add("<URL_TOKEN>")
        ## Filter Empty Strings
        if hasattr(self,"filter_empty") and self.filter_empty:
            self.filter_set.add("''")
            self.filter_set.add('""')
    
    def filter_user_data(self,
                         user_data,
                         tokens_only=True,
                         ignore_text=False,
                         ignore_sentences=False):
        """
        Filter loaded user data based on class initialization
        parameters.

        Args:
            user_data (list of dict): Preprocessed user data
            tokens_only (bool)
        
        Returns:
            filtered_data (list of dict): Filtered user data
        """
        ## Tokenized Text Field
        tt = "text_tokenized"
        st = "sentences_tokenized"
        ## Initialize Filtered Data Cache
        filtered_data = []
        for i, d in enumerate(user_data):
            ## Filter Based on Retweets
            if hasattr(self, "keep_retweets") and not self.keep_retweets and "<RETWEET>" in set(d["text_tokenized"]):
                continue
            ## Filter Based on Subreddit
            if hasattr(self, "filter_mh_subreddits") and self.filter_mh_subreddits is not None and "subreddit" in d.keys():
                if d["subreddit"] is not None and d["subreddit"].lower() in self._ignore_subreddits:
                    continue
            ## Filter Based on Terms
            if hasattr(self, "filter_mh_terms") and self.filter_mh_terms is not None:
                if "text" in d.keys():
                    if pattern_match(d["text"], self._ignore_terms):
                        continue
                else:
                    if pattern_match(" ".join(d["text_tokenized"]), self._ignore_terms):
                        continue
            ## Filter Based on Ignore Set
            if not ignore_text:
                d[tt] = self._filter_in(d[tt], self.filter_set)
            if not ignore_sentences:
                for s, _ in enumerate(d.get(st, [])):
                    d[st][s] = self._filter_in(d[st][s], self.filter_set)
            ## Length Check
            if len(d[tt]) == 0:
                filtered_data.append(d)
                continue
            ## Filter Retweet Tokens
            if hasattr(self, "filter_retweet") and self.filter_retweet and d[tt][0] == "<RETWEET>":
                ## Text-level filtering
                if not ignore_text:
                    if len(d[tt]) <= 1:
                        continue
                    d[tt] = d[tt][1:]
                    for _ in range(2):
                        if len(d[tt]) == 0:
                            break
                        if d[tt][0] in ["<USER_MENTION>", ":"]:
                            d[tt] = d[tt][1:]
                ## Sentence Filtering
                if st in d and not ignore_sentences:
                    if len(d[st][0]) <= 1:
                        continue
                    d[st][0] = d[st][0][1:]
                    for _ in range(2):
                        if len(d[st][0]) == 0:
                            break
                        if d[st][0][0] in ["<USER_MENTION>",":"]:
                            d[st][0] = d[st][0][1:]
            if hasattr(self, "filter_retweet") and self.filter_retweet:
                if not ignore_text:
                    d[tt] = list(filter(lambda i: i!="<RETWEET>", d[tt]))
                if not ignore_sentences:
                    for s, _ in enumerate(d.get(st, [])):
                        d[st][s] = list(filter(lambda i: i != "<RETWEET>", d[st][s]))
            ## Filter Hashtags
            if hasattr(self, "filter_hashtag") and self.filter_hashtag:
                if not ignore_text:
                    d[tt] = self._remove_hashtags(d[tt])
                if not ignore_sentences:
                    for s, _ in enumerate(d.get(st, [])):
                        d[st][s] = self._remove_hashtags(d[st][s])
            else:
                if hasattr(self, "strip_hashtag") and self.strip_hashtag:
                    if not ignore_text:
                        d[tt] = self._strip_hashtags(d[tt])
                    if not ignore_sentences:
                        for s, _ in enumerate(d.get(st, [])):
                            d[st][s] = self._strip_hashtags(d[st][s])
            ## Max Tokens
            if hasattr(self, "max_tokens_per_document") and self.max_tokens_per_document is not None:
                if not ignore_text:
                    d[tt] = d[tt][:min(len(d[tt]), self.max_tokens_per_document)]
                if not ignore_sentences:
                    cur_s_n = 0
                    for s, _ in enumerate(d.get(st, [])):
                        stok = []
                        for tok in d[st][s]:
                            if cur_s_n < self.max_tokens_per_document:
                                stok.append(tok)
                                cur_s_n += 1
                            else:
                                break
                        d[st][s] = stok
            ## Filter Stopwords
            if hasattr(self, "filter_stopwords") and self.filter_stopwords:
                if not ignore_text:
                    d[tt] = list(filter(lambda x: x.lower().replace("not_","") not in self.stopwords, d[tt]))
                if not ignore_sentences:
                    for s, _ in enumerate(d.get(st, [])):
                        d[st][s] = list(filter(lambda x: x.lower().replace("not_","") not in self.stopwords, d[st][s]))
            ## Filter Punctuation
            if hasattr(self, "filter_punctuation") and self.filter_punctuation:
                if not ignore_text:
                    d[tt] = list(filter(lambda i: not all(char in self._punc for char in i), d[tt]))
                if not ignore_sentences:
                    for s, _ in enumerate(d.get(st, [])):
                        d[st][s] = list(filter(lambda i: not all(char in self._punc for char in i), d[st][s]))
            ## Case Formatting
            if hasattr(self, "preserve_case") and not self.preserve_case:
                if not ignore_text:
                    d[tt] = list(map(lambda i: "<HASHTAG={}".format(i.replace("<HASHTAG=","").lower()) if i.startswith("<HASHTAG=") else i, d[tt]))
                    d[tt] = list(map(lambda tok: tok.lower() if tok not in self.filter_set and not tok.startswith("<HASHTAG") else tok, d[tt]))
                if not ignore_sentences:
                    for s, _ in enumerate(d.get(st, [])):
                        d[st][s] = list(map(lambda i: "<HASHTAG={}".format(i.replace("<HASHTAG=","").lower()) if i.startswith("<HASHTAG=") else i, d[st][s]))
                        d[st][s] = list(map(lambda tok: tok.lower() if tok not in self.filter_set and not tok.startswith("<HASHTAG") else tok, d[st][s]))
            ## Emoji Handling
            if hasattr(self, "emoji_handling") and self.emoji_handling is not None:
                if self.emoji_handling == "replace":
                    if not ignore_text:
                        d[tt] = self._replace_emojis(d[tt])
                    if not ignore_sentences:
                        for s, _ in enumerate(d.get(st, [])):
                            d[st][s] = self._replace_emojis(d[st][s])
                elif self.emoji_handling == "strip":
                    if not ignore_text:
                        d[tt] = self._strip_emojis(d[tt])
                    if not ignore_sentences:
                        for s, _ in enumerate(d.get(st, [])):
                            d[st][s] = self._strip_emojis(d[st][s])
                else:
                    raise ValueError("emoji_handling should be 'replace', 'strip', or None.")
            filtered_data.append(d)
        ## Language Identification
        if not tokens_only and self.lang is not None:
            languages = self._classify_languages(list(map(lambda d: d.get("text",""), filtered_data)))
            filtered_data = [f for f, l in zip(filtered_data, languages) if l[0] in self.lang]
        return filtered_data


class PostStream(object):

    """

    """

    def __init__(self,
                 filenames,
                 loader_kwargs=DEFAULT_LOADER_KWARGS,
                 processor=None,
                 processor_kwargs={},
                 min_date=None,
                 max_date=None,
                 n_samples=None,
                 randomized=False,
                 tokenizer=None,
                 mode=0,
                 metadata=[],
                 phrasers=None,
                 jobs=1,
                 preserve_order=True,
                 verbose=False,
                 check_filename_size=True,
                 yield_filename=False,
                 cache_data=False,
                 cache_dir="./"):
        """
        Args:
            filenames (list): List of processed data files
            loader_kwargs (dict): Arguments for LoadProcessedData
            min_date (None, str, or int): Minimum date
            max_date (None, str, or int): Maximum date
            n_samples (int or None): Sampling rate and or count
            randomized (bool): How to sample if desired
            tokenizer (Tokenizer): Tokenizer class with .tokenize method if you want to retokenize
            return_sentence (bool): If True, yield sentences. Otherwise, yield an entire post.
            mode (int: {0, 1, 2}): 0 (user - document - sentence), 1 (user - document), 2 (user)
            metadata (list of str): Fields to return with text
            jobs (int): Number of cores available for processing
            verbose (bool): Verbosity of loading/processing
            cache_data (bool): If re-using same stream, can write processed lines to disk
        """
        self.filenames = filenames
        self.tokenizer = tokenizer
        self.loader = LoadProcessedData(**loader_kwargs)
        self.processor = processor
        self.processor_kwargs = processor_kwargs
        self.mode = mode
        self.verbose = verbose
        self.n_samples = n_samples
        self.randomized = randomized
        self.phrasers = phrasers
        self.metadata = metadata
        self.jobs = jobs
        self.preserve_order = preserve_order
        self.yield_filename = yield_filename
        self._base_cache_dir = cache_dir
        self._initialize_cache(cache_data, True)
        self._initialize_dates(min_date, max_date)
        if check_filename_size:
            self._initialize_filenames()
        self._check_init()
    
    def _check_init(self):
        """
        
        """
        if self.mode not in set([0,1,2]):
            raise ValueError("Mode not recognized.")

    def _initialize_dates(self,
                          min_date,
                          max_date):
        """
        
        """
        ## Format Minimum Date
        if min_date is not None and isinstance(min_date, str):
            min_date = pd.to_datetime(min_date)
        elif min_date is not None and isinstance(min_date, int):
            min_date = datetime.fromtimestamp(min_date)
        ## Format Maximum Date
        if max_date is not None and isinstance(max_date, str):
            max_date = pd.to_datetime(max_date)
        elif max_date is not None and isinstance(max_date, int):
            max_date = datetime.fromtimestamp(max_date)
        ## Assign Dates to Class
        self.min_date = min_date
        self.max_date = max_date
    
    def __len__(self):
        """
        
        """
        return len(self.filenames)
    
    def _check_nonzero_file_length(self,
                                   filename):
        """

        """
        lf = self.loader.load_user_data(filename,
                                        min_date=self.min_date,
                                        max_date=self.max_date,
                                        n_samples=self.n_samples,
                                        randomized=self.randomized,
                                        apply_filter=False,
                                        processor=self.processor,
                                        processor_kwargs=self.processor_kwargs,
                                        ignore_text_filter=self.mode==0,
                                        ignore_sentence_filter=self.mode==1)
        if len(lf) > 0:
            return filename
    
    def _initialize_filenames(self):
        """
        
        """
        ## Temporarily Update Mode
        mode = self.mode
        self.mode = 0
        ## Create Pool
        with Pool(self.jobs) as mp:
            act = mp.imap_unordered(self._check_nonzero_file_length, self.filenames)
            ## Run Check
            if self.verbose:
                filenames = list(tqdm(act, total=len(self.filenames), file=sys.stdout, desc="Filesize Filter"))
            else:
                filenames = list(act)
        ## Force Close the Pool
        mp.join()
        mp.close()
        ## Realign Mode
        self.mode = mode
        ## Filter Filenames
        self.filenames = list(filter(lambda i: i is not None, filenames))

    def _rephrase_tokens(self,
                         tokens,
                         phrasers):
        """
        
        """
        ## Split N-Grams
        out = tokens
        for phraser in phrasers:
            out = phraser[out]
        return out
        
    def _initialize_cache(self,
                          cache_data,
                          make_dir=True):
        """

        """
        ## Check for Existing
        if hasattr(self, "cache_dir") and os.path.exists(self.cache_dir):
            raise Exception("Cannot create a new cache directory until removing old one: {}".format(self.cache_dir))
        ## Set Variables
        self.cache_dir = None if not cache_data else "{}/_tmp/{}/".format(self._base_cache_dir, str(uuid4()))
        self.filenames_cache = None if not cache_data else {}
        ## Create Directory
        if cache_data and make_dir:
            _ = os.makedirs(self.cache_dir)
    
    def _remove_cache_dir(self,
                          directory):
        """

        """
        ## Make a Temporary Empty Directory
        temp_empty_dir = "{}/_tmp/{}/".format(self._base_cache_dir, str(uuid4()))
        _ = os.makedirs(temp_empty_dir)
        ## Rsync Remove
        command = "rsync -a --delete {}/ {}/".format(temp_empty_dir, directory).replace("//","/")
        _ = os.system(command)
        ## Remove Empty Directories
        for dd in [temp_empty_dir, directory]:
            _ = os.system("rm -rf {}".format(dd))
        
    def remove_cache(self,
                     make_dir=True):
        """

        """
        ## Not Relevant
        if self.cache_dir is None:
            return None
        ## Remove Old Cache Directory
        if os.path.exists(self.cache_dir):
            if self.verbose:
                LOGGER.info("Removing cache directory: {}".format(self.cache_dir))
            _ = self._remove_cache_dir(directory=self.cache_dir)
        ## Re-initialize Cache Directory (Without Creating Directory itself)
        _ = self._initialize_cache(True, make_dir=make_dir)
    
    def _cache_data(self,
                    file_data_cache,
                    filename_cache):
        """

        """
        file_data_cache = [{"data":d} for d in file_data_cache]
        with open(filename_cache,"w") as the_file:
            for item in file_data_cache:
                the_file.write(f"{json.dumps(item)}\n")
    
    def _load_cached_file(self,
                          filename_cache):
        """

        """
        ## Null File
        if filename_cache is None:
            return []
        ## Load Json
        data = []
        with open(filename_cache,"r") as the_file:
            for line in the_file:
                data.append(json.loads(line)["data"])
        return data
    
    def _iter_inner_loop(self,
                         fid,
                         filename,
                         return_filename=False):
        """
        
        """
        ## Look for Cache
        if self.cache_dir is not None and filename in self.filenames_cache:
            file_data = self._load_cached_file(self.filenames_cache[filename])
            if return_filename:
                return file_data, filename
            else:
                return file_data
        ## Initialize File Cache
        file_data_cache = []
        ## Load Data
        file_data = self.loader.load_user_data(filename,
                                               min_date=self.min_date,
                                               max_date=self.max_date,
                                               n_samples=self.n_samples,
                                               randomized=self.randomized,
                                               apply_filter=self.tokenizer is None,
                                               processor=self.processor,
                                               processor_kwargs=self.processor_kwargs,
                                               ignore_text_filter=self.mode==0,
                                               ignore_sentence_filter=self.mode==1)
        ## Check Length
        if len(file_data) == 0:
            if return_filename:
                return [], filename
            else:
                return []
        ## Retokenize (if Desired)
        if self.tokenizer is not None:
            file_data_retokenized = []
            for post in file_data:
                post_sentences = sent_tokenize(post.get("text"))
                sentences_tokenized = list(map(self.tokenizer.tokenize, post_sentences))
                file_data_retokenized.append(
                    {
                        "text":post.get("text"),
                        "sentences":post_sentences,
                        "text_tokenized":flatten(sentences_tokenized),
                        "sentences_tokenized":sentences_tokenized,
                        "user_id_str":post.get("user_id_str"),
                        "created_utc":post.get("created_utc"),
                        "subreddit":post.get("subreddit",None)
                    }
                )
            ## Filter
            file_data = self.loader.filter_user_data(file_data_retokenized,
                                                     tokens_only=False,
                                                     ignore_text=self.mode==0,
                                                     ignore_sentences=self.mode==1)
        ## Rephrase Data (if Necessary)
        if self.phrasers is not None:
            for p, post in enumerate(file_data):
                ## Isolate Data
                if self.mode != 1:
                    sentences_tokenized = post.get("sentences_tokenized",[])
                if self.mode != 0:
                    text_tokenized = post.get("text_tokenized",[])
                ## Rephrase
                if self.mode != 1:
                    for s, sentence_tokenized in enumerate(sentences_tokenized):
                        sentence_tokenized = self._rephrase_tokens(sentence_tokenized, self.phrasers)
                        sentences_tokenized[s] = sentence_tokenized
                if self.mode != 0:
                    text_tokenized = self._rephrase_tokens(text_tokenized, self.phrasers)
                ## Update Post
                if self.mode != 1:
                    post["sentences_tokenized"] = sentences_tokenized
                if self.mode != 0:
                    post["text_tokenized"] = text_tokenized
                ## Update Data Cache
                file_data[p] = post
        ## Yield Results Based on Mode
        if self.mode == 0 or self.mode == 1:
            ## Cycle Through Posts
            for post in file_data:
                ## Post Metadata
                post_metadata = None
                if self.metadata is not None and isinstance(self.metadata, list) and len(self.metadata) > 0:
                    post_metadata = [post.get(m,None) for m in self.metadata]
                ## Yield Post or Sentence
                if self.mode == 0:
                    yield_tokens_iter = post.get("sentences_tokenized",[])
                elif self.mode == 1:
                    yield_tokens_iter = [post.get("text_tokenized",[])]
                ## Cycle Through Tokens
                for tokens in yield_tokens_iter:
                    ## Check Length
                    if len(tokens) == 0:
                        continue
                    ## Return
                    if post_metadata is not None:
                        if self.cache_dir is not None:
                            file_data_cache.append(post_metadata + [tokens])
                    else:
                        if self.cache_dir is not None:
                            file_data_cache.append(tokens)
        elif self.mode == 2:
            file_data_cache.append(file_data)
        else:
            raise ValueError(f"mode=`{self.mode}` not recognized.")
        ## Caching
        if self.cache_dir is not None:
            _ = self._cache_data(file_data_cache, f"{self.cache_dir}/{fid}.json")
        if return_filename:
            return file_data_cache, filename
        else:
            return file_data_cache
        
    def _mphelper(self,
                  fid_filename):
        """
        
        """
        fid, filename = fid_filename
        return self._iter_inner_loop(fid, filename, return_filename=True)

    def build_cache(self):
        """

        """
        ## Check Attributes
        if self.cache_dir is None and self.filenames_cache is not None:
            raise ValueError("Cannot build a cache without a cache directory initialized.")
        if len(self.filenames_cache) == len(self.filenames):
            raise FileExistsError("Cache has already been created. Must remove before building again.")
        ## Ensure Cache Directory Exists
        if self.cache_dir is not None and not os.path.exists(self.cache_dir):
            _ = self._initialize_cache(True, True)
        ## Alert User
        if self.verbose:
            LOGGER.info("Building Cache")
        ## Build Cache using Multiprocessing
        file2ind = dict(zip(self.filenames, range(len(self.filenames))))
        with Pool(self.jobs) as mp:
            ## Initialize Iterable
            if self.preserve_order:
                iterable = mp.imap(self._mphelper, list(enumerate(self.filenames)))
            else:
                iterable = mp.imap_unordered(self._mphelper, list(enumerate(self.filenames)))
            ## Run Iterable
            for file_index, (items, filename) in enumerate(iterable):
                ## Update User
                if self.verbose and (file_index + 1) % 100 == 0:
                    LOGGER.info("Caching File {}/{}".format(file_index + 1, len(self.filenames)))
                ## Update File Cache (Outside of Inner Loop due to Internal Race Conditions)
                fid = file2ind[filename]
                if filename not in self.filenames_cache:
                    if len(items) > 0:
                        self.filenames_cache[filename] = f"{self.cache_dir}/{fid}.json"
                    else:
                        self.filenames_cache[filename] = None
        ## Close the Pool
        mp.join()
        mp.close()
        ## Done
        if self.verbose:
            LOGGER.info("Caching Complete. Found {}/{} Files With Data".format(
                len([x for x, y in self.filenames_cache.items() if y is not None]),
                len(self.filenames)
            ))

    def __iter__(self):
        """
        
        """
        ## Cache Initialization (if Necessary)
        if self.cache_dir is not None and not os.path.exists(self.cache_dir):
            _ = self._initialize_cache(True, True)
        ## Filename to Index
        file2ind = dict(zip(self.filenames, range(len(self.filenames))))
        ## Cycle Through Filenames
        for iterb in list(enumerate(self.filenames)):
            ## Process File (Or Load Pre-Cached Data from File)
            items, filename = self._mphelper(iterb)
            ## Update File Cache (if cache is not already built)
            fid = file2ind[filename]
            if self.cache_dir is not None and self.filenames_cache is not None and filename not in self.filenames_cache:
                if len(items) > 0:
                    self.filenames_cache[filename] = f"{self.cache_dir}/{fid}.json"
                else:
                    self.filenames_cache[filename] = None
            ## Iterate Through Subitems (e.g. sentences, posts)
            for item in items:
                if self.yield_filename:
                    yield item, filename
                else:
                    yield item

            
class PostStreamPipeline(object):

    """
    
    """

    def __init__(self,
                 *streams):
        """
        
        """
        self.streams = streams
        _ = self._update_cache_dir()

    def __repr__(self):
        """
        
        """
        return "PostStreamPipeline()"

    def __len__(self):
        """
        
        """
        return sum(len(s) for s in self.streams)
    
    def _update_cache_dir(self):
        """

        """
        self.cache_dir = None if all(s.cache_dir is None for s in self.streams) else [s.cache_dir for s in self.streams]

    def remove_cache(self,
                     make_dir=True):
        """

        """
        ## Remove Stream Caches
        for stream in self.streams:
            _ = stream.remove_cache(make_dir=make_dir)
        ## Update Pipeline Cache Dir
        _ = self._update_cache_dir()
    
    def build_cache(self):
        """

        """
        ## Build Caches for Each Stream
        for stream in self.streams:
            _ = stream.build_cache()
        ## Ensure Cache Directory Makes Sense
        _ = self._update_cache_dir()
    
    def __iter__(self):
        """
        
        """
        ## Iterate Over Streams
        for stream in self.streams:
            ## Iterate Over Outputs
            for output in stream:
                yield output