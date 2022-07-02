
"""
Default preprocessing code for all model training.
"""

#############################
### Imports
#############################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from glob import glob
from copy import deepcopy
from functools import partial
from datetime import datetime
from dateutil.parser import parse

## External Library
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize

## Local Modules
from ..util.helpers import flatten
from .tokenizer import Tokenizer

#############################
### Globals
#############################

## Initialize Text Tokenizer
tokenizer = Tokenizer(stopwords=None,
                      keep_case=True,
                      negate_handling=True,
                      negate_token=True,
                      upper_flag=True,
                      keep_punctuation=True,
                      keep_numbers=True,
                      expand_contractions=True,
                      keep_user_mentions=True,
                      keep_pronouns=True,
                      keep_url=True,
                      keep_hashtags=True,
                      keep_retweets=True,
                      emoji_handling=None,
                      strip_hashtag=False)

#############################
### Field Schemas
#############################

## Fields Cached For Each Dataset
db_schema = {
            "qntfy":{
                            "tweet":{   ## Tweet Information
                                        "user_id_str":"user_id_str",
                                        "created_utc":"created_utc",
                                        "text":"text",
                                        "text_tokenized":"text_tokenized",
                                        "sentences":"sentences",
                                        "sentences_tokenized":"sentences_tokenized",
                                        "id_str":"tweet_id",
                                        ## User Metadata
                                        "age":"age",
                                        "gender":"gender",
                                        ## Labels (All Possible)
                                        "depression":"depression",
                                        "anxiety":"anxiety",
                                        "suicide_attempt":"suicide_attempt",
                                        "suicidal_ideation":"suicidal_ideation",
                                        "eating_disorder":"eating_disorder",
                                        "panic":"panic",
                                        "schizophrenia":"schizophrenia",
                                        "bipolar":"bipolar",
                                        "ptsd":"ptsd",
                                        ## Processing Metadata
                                        "entity_type":"entity_type",
                                        "date_processed_utc":"date_processed_utc",
                                        "source":"source",
                                        "dataset":"dataset",
                                    }
            },
            "shen2017":{
                    "tweet":{ ## Tweet Information
                             "user_id_str":"user_id_str",
                             "created_utc":"created_utc",
                             "text":"text",
                             "text_tokenized":"text_tokenized",
                             "sentences":"sentences",
                             "sentences_tokenized":"sentences_tokenized",
                             "id_str":"tweet_id",
                             "retweeted":"retweeted",
                             "lang":"lang",
                             ## User Metadata
                             "user_name":"user_name",
                             "user_screen_name":"user_screen_name",
                             ## Labels
                             "depression":"depression",
                             ## Processing Metadata
                             "entity_type":"entity_type",
                             "date_processed_utc":"date_processed_utc",
                             "source":"source",
                             "dataset":"dataset",
                    }
            },
            "rsdd":{
                    "comment":{ 
                                ## Post Information
                                "author":"user_id_str",
                                "created_utc":"created_utc",
                                "text":"text",
                                "text_tokenized":"text_tokenized",
                                "sentences":"sentences",
                                "sentences_tokenized":"sentences_tokenized",
                                ## Labels
                                "depression":"depression",
                                ## Processing Metadata
                                "entity_type":"entity_type",
                                "date_processed_utc":"date_processed_utc",
                                "source":"source",
                                "dataset":"dataset"
                                }
            },
            "smhd":{
                    "comment":{
                                ## Comment Information
                                "author":"user_id_str",
                                "created_utc":"created_utc",
                                "text":"text",
                                "text_tokenized":"text_tokenized",
                                "sentences":"sentences",
                                "sentences_tokenized":"sentences_tokenized",
                                ## Labels
                                "adhd":"adhd",
                                "anxiety":"anxiety",
                                "autism":"autism",
                                "bipolar":"bipolar",
                                "depression":"depression",
                                "eating":"eating_disorder",
                                "ocd":"ocd",
                                "ptsd":"ptsd",
                                "schizophrenia":"schizophrenia",
                                ## Processing Metadata
                                "entity_type":"entity_type",
                                "date_processed_utc":"date_processed_utc",
                                "source":"source",
                                "dataset":"dataset"
                                }
            },
            "wolohan": {
                        "comment":{
                                    ## Comment Information
                                    "author":"user_id_str",
                                    "subreddit":"subreddit",
                                    "created_utc":"created_utc",
                                    "body":"text",
                                    "id":"comment_id",
                                    "text_tokenized":"text_tokenized",
                                    "sentences":"sentences",
                                    "sentences_tokenized":"sentences_tokenized",
                                    "link_id":"submission_id",
                                    ## Labels
                                    "depression":"depression",
                                    ## Processing Metadata
                                    "entity_type":"entity_type",
                                    "date_processed_utc":"date_processed_utc",
                                    "source":"source",
                                    "dataset":"dataset"
                                    },
                        "submission":{
                                    ## Submission Information
                                    "author":"user_id_str",
                                    "subreddit":"subreddit",
                                    "created_utc":"created_utc",
                                    "selftext":"text",
                                    "title":"title",
                                    "id":"submission_id",
                                    "text_tokenized":"text_tokenized",
                                    "title_tokenized":"title_tokenized",
                                    "sentences":"sentences",
                                    "sentences_tokenized":"sentences_tokenized",
                                    ## Labels
                                    "depression":"depression",
                                    ## Processing Metadata
                                    "entity_type":"entity_type",
                                    "date_processed_utc":"date_processed_utc",
                                    "source":"source",
                                    "dataset":"dataset"
                                    }
                },      
}

#############################
### Functions
#############################

def sent_tokenize(text, **kwargs):
    """

    """
    if text is None:
        return []
    return nltk_sent_tokenize(text, **kwargs)

def _clean_surrogate_unicode(text):
    """
    Clean strings that have non-processable unicode (e.g.
    'RT @lav09rO5KgJS: Tell em J.T. ! 😂😍\ud83dhttp://t.co/Tc_qbFYmFYm')

    Args:
        text (str): Input text
    
    Returns:
        cleaned text if a unicode error otherwise arises
    """
    if text is None:
        return text
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError as e:
        if e.reason == 'surrogates not allowed':
            return text.encode("utf-8", "ignore").decode("utf-8")
        else:
            raise(e)

def format_tweet_data(data):
    """
    Extract tweet and user data from JSON dictionary
    
    Args:
        data (dict): Dictionary of raw tweet data
    
    Returns:
        formatted_data (dict): Data with user-level information extracted
    """
    ## Define Data To Extract (Tweet- and User-level)
    tweet_cols = ["truncated",
                  "text",
                  "in_reply_to_status_id",
                  "id",
                  "favorite_count",
                  "retweeted",
                  "in_reply_to_screen_name",
                  "in_reply_to_user_id",
                  "retweet_count",
                  "id_str",
                  "geo",
                  "in_reply_to_user_id_str",
                  "lang",
                  "created_at",
                  "in_reply_to_status_id_str",
                  "place"]
    user_cols = ["verified",
                 "followers_count",
                 "utc_offset",
                 "statuses_count",
                 "friends_count",
                 "geo_enabled",
                 "screen_name",
                 "lang",
                 "favourites_count",
                 "url",
                 "created_at",
                 "time_zone",
                 "listed_count",
                 "id_str",
                 "name",
                 "location",
                 "description",
                 "time_zone",
                 "utc_offset",
                 "protected",
                 "profile_image_url"]
    ## Extract Data
    formatted_data = {}
    for t in tweet_cols:
        if t in data:
            if t.startswith("in_reply"):
                formatted_data[t] = str(data[t])
            else:
                formatted_data[t] = data[t]
        else:
            formatted_data[t] = None
    if "full_text" in data:
        formatted_data["text"] = data["full_text"]
    if "extended_tweet" in data and "full_text" in data.get("extended_tweet"):
        formatted_data["text"] = data.get("extended_tweet").get("full_text")
    for u in user_cols:
        if "user" not in data:
            formatted_data[f"user_{u}"] = None
        else:
            if u in data["user"]:
                formatted_data[f"user_{u}"] = data["user"][u]
            else:
                formatted_data[f"user_{u}"] = None
    ## Clean Surrogate Unicode if Necessary
    formatted_data["text"] = _clean_surrogate_unicode(formatted_data["text"])
    return formatted_data

#######################
### General Use
#######################

## Cache Schema
GENERAL_SCHEMA = {
    "twitter":{
        "tweet":{
                'user_id_str': 'user_id_str', 
                'created_utc': 'created_utc', 
                'text': 'text', 
                'text_tokenized': 'text_tokenized',
                'sentences':'sentences',
                'sentences_tokenized':'sentences_tokenized',
                'id_str': 'tweet_id', 
                'entity_type': 'entity_type', 
                'date_processed_utc': 'date_processed_utc', 
                'source': 'source'}
    },
    "reddit":{
        "comment":{
                "author_fullname":"user_id_str",
                "created_utc":"created_utc",
                "body":"text",
                "text_tokenized":"text_tokenized",
                'sentences':'sentences',
                'sentences_tokenized':'sentences_tokenized',
                "id":"comment_id",
                "entity_type":"entity_type",
                'date_processed_utc': 'date_processed_utc', 
                'subreddit':'subreddit',
                "link_id":"submission_id",
                'source': 'source'}
    }, 
}

class RawDataLoader(object):
    
    """

    """

    def __init__(self,
                 platform,
                 random_state=None,
                 lang=None,
                 run_pipeline=True,
                 keep_retweets=True,
                 keep_non_english=True):
        """

        """
        self.platform = platform
        self.random_state = random_state
        self.lang = lang if lang is None else set(lang)
        self._run_pipeline = run_pipeline
        self._keep_retweets = keep_retweets
        self._keep_non_english = keep_non_english
        if not self._keep_non_english:
            self.lang = set(["en"])
    
    def _doc_in_range(self,
                      document,
                      min_date=None,
                      max_date=None):
        """
        Isolate documents in a list of processed user_data
        that fall into a given date range

        Args:
            document (dict):
            min_date (datetime or None): Lower date boundary
            max_date (datetime or None): Upper date boundary
        
        Returns:
            boolean (list of dict): Filtered user data
        """
        ## No additional filtering
        if min_date is None and max_date is None:
            return True
        ## Retrive Filtered Data
        tstamp = datetime.fromtimestamp(document["created_utc"])
        if min_date is not None and tstamp < min_date:
            return False
        if max_date is not None and tstamp > max_date:
            return False
        return True
    
    def _filter_post(self,
                     post,
                     filters=None):
        """

        """
        if filters is None:
            return post
        post_filtered = {}
        for f in filters:
            if f not in post:
                continue
            if isinstance(f, str):
                post_filtered[f] = post.get(f)
            elif isinstance(f, list) or isinstance(f, tuple):
                fdata = post
                for _f in f[:-1]:
                    fdata = fdata.get(_f,{})
                post_filtered[tuple(f)] = fdata.get(f[-1])
            else:
                raise ValueError("Filter type not recognized")
        return post_filtered
    
    def _parse_sample_rate(self,
                           sample_rate):
        """

        """
        if sample_rate is not None and isinstance(sample_rate, str):
            if sample_rate.startswith("discrete-"):
                sample_rate = sample_rate.split("discrete-")[1].split("-")
                if len(sample_rate) != 2:
                    raise ValueError("String 'sample_rate' with discrete setting is misconfigured. Should be 'discrete-<NSPLITS>-SPLIT1,SPLIT2'")
                try:
                    sample_rate = (int(sample_rate[0]), set(map(int, sample_rate[1].split(","))))
                except Exception as e:
                    raise e
            else:
                raise ValueError("String parameterization of 'sample_rate' is misconfigured.")
        return sample_rate

    def _load_file_in_full(self,
                           filename,
                           min_date=None,
                           max_date=None,
                           sample_rate=None,
                           sample_size=None,
                           filters=None):
        """

        """
        ## Set Random State
        if self.random_state is not None:
            seed = np.random.RandomState(self.random_state)
        else:
            seed = np.random.RandomState()
        ## Choose Loader
        if filename.endswith(".gz"):
            file_opener = gzip.open
        else:
            file_opener = open
        ## Try to Load Data In Full
        try:
            with file_opener(filename,"r") as the_file:
                data = json.load(the_file)
        except:
            return []
        ## Parse Sample Rate
        sample_rate = self._parse_sample_rate(sample_rate)
        ## Filter Data
        count = 0
        data_mask = []
        for l, line_data in enumerate(data):
            ## Check Sample Size 
            if sample_size and count >= sample_size:
                break
            ## Sampling
            if sample_rate is not None and isinstance(sample_rate, tuple):
                if seed.choice(sample_rate[0]) not in sample_rate[1]:
                    continue
            elif sample_rate is not None and not isinstance(sample_rate, tuple):
                if sample_rate is not None and sample_rate < 1:
                    if seed.uniform(0,1) >= sample_rate:
                        continue
            ## Twitter Format
            if self.platform == "twitter" and isinstance(line_data.get("user"),dict):
                line_data = format_tweet_data(line_data)
            ## Attribute Check
            if self.platform == "twitter" and not self._keep_retweets:
                if line_data["text"] is not None and (line_data["text"].startswith("RT") or " RT " in line_data["text"]):
                    continue
            ## Preliminary Language Check
            if self.lang is not None and self.platform == "twitter":
                line_data_lang = line_data.get("lang")
                if line_data_lang is not None and line_data_lang not in self.lang:
                    continue
            ## Time Check
            if "created_utc" not in line_data:
                if "created_at" in line_data:
                    line_data["created_utc"] = parse(line_data["created_at"])
                else:
                    continue
            if not self._doc_in_range(line_data, min_date, max_date):
                continue
            ## Filter
            data[l] = self._filter_post(line_data, filters)
            ## Cache or Yield
            count += 1
            data_mask.append(l)
        ## Isolate Data Based on Mask
        data = [data[m] for m in data_mask]
        return data

    def _load_file(self,
                   filename,
                   min_date=None,
                   max_date=None,
                   sample_rate=None,
                   sample_size=None,
                   filters=None):
        """

        """
        ## Set Random State
        if self.random_state is not None:
            seed = np.random.RandomState(self.random_state)
        else:
            seed = np.random.RandomState()
        ## Choose Loader
        if filename.endswith(".gz"):
            file_opener = gzip.open
        else:
            file_opener = open
        ## Parse Sample Rate
        sample_rate = self._parse_sample_rate(sample_rate)
        ## Load In Stream
        data = []
        count = 0
        with file_opener(filename,"r") as the_file:
            try:
                for l, line in enumerate(the_file):
                    ## Check Sample Size
                    if sample_size and count >= sample_size:
                        break
                    ## Sampling
                    if sample_rate is not None and isinstance(sample_rate, tuple):
                        if seed.choice(sample_rate[0]) not in sample_rate[1]:
                            continue
                    elif sample_rate is not None and not isinstance(sample_rate, tuple):
                        if sample_rate is not None and sample_rate < 1:
                            if seed.uniform(0,1) >= sample_rate:
                                continue
                    ## Format Data
                    line_data = json.loads(line)
                    ## Check Deletion Status
                    if "delete" in line_data.keys():
                        continue
                    ## Wrong Data Format Expected, Move On to Alternative
                    if isinstance(line_data, list):
                        break
                    ## Twitter Formating
                    if self.platform == "twitter" and isinstance(line_data.get("user"),dict):
                        line_data = format_tweet_data(line_data)
                    ## Attribute Check
                    if self.platform == "twitter" and not self._keep_retweets:
                        if line_data["text"] is not None and (line_data["text"].startswith("RT") or " RT " in line_data["text"]):
                            continue
                    ## Preliminary Language Check
                    if self.lang is not None and self.platform == "twitter":
                        line_data_lang = line_data.get("lang")
                        if line_data_lang is not None and line_data_lang not in self.lang:
                            continue
                    ## Time Check
                    if "created_utc" not in line_data:
                        if "created_at" in line_data:
                            line_data["created_utc"] = parse(line_data["created_at"])
                        else:
                            continue
                    if not self._doc_in_range(line_data, min_date, max_date):
                        continue
                    ## Filtering
                    line_data = self._filter_post(line_data, filters)
                    ## Cache or Stream
                    count += 1
                    data.append(line_data)
            except Exception as e:
                print("Error occurred", e)
                pass
        ## Check For Alternative Data Options if No Data Found
        if len(data) == 0 and count == 0:
            data = self._load_file_in_full(filename,
                                           min_date=min_date,
                                           max_date=max_date,
                                           sample_rate=sample_rate,
                                           sample_size=sample_size,
                                           filters=filters)
        return data
    
    def _rename_keys(self,
                     post):
        """

        """
        ## Get Schema
        if self.platform == "twitter":
            schema = GENERAL_SCHEMA["twitter"]["tweet"] 
        elif self.platform == "reddit":
            schema = GENERAL_SCHEMA["reddit"]["comment"] 
        ## Rename
        for source, target in schema.items():
            if source not in post:
                continue
            post[target] = post.pop(source,None)
        return post
    
    def _process_tweet(self,
                       tweet):
        """

        """
        ## Check None
        if tweet is None:
            return None
        ## Format Raw Tweet Object (Flatten)
        if isinstance(tweet.get("user"), dict):
            tweet = format_tweet_data(tweet)
        ## Early Return
        if not self._run_pipeline:
            tweet = self._rename_keys(tweet)
            return tweet
        ## Check for Text
        if "text" not in tweet or tweet.get("text") is None:
            return None
        ## Tokenization
        tweet["sentences"] = sent_tokenize(tweet["text"])
        tweet["sentences_tokenized"] = list(map(tokenizer.tokenize, tweet["sentences"]))
        tweet["text_tokenized"] = flatten(tweet["sentences_tokenized"])
        ## Datetime Conversion
        if "created_utc" not in tweet:
            if "created_at" in tweet:
                tweet["created_utc"] = int(parse(tweet["created_at"]).timestamp())
            else:
                tweet["created_utc"] = 0
        else:
            if isinstance(tweet.get("created_utc"), datetime):
                tweet["created_utc"] = int(tweet["created_utc"].timestamp())
        ## Add Meta
        if "source" not in tweet:
            if "id_str" in tweet:
                tweet["source"] = tweet["id_str"]
            else:
                tweet["source"] = None
        tweet["entity_type"] = "tweet"
        tweet["date_processed_utc"] = int(datetime.utcnow().timestamp())
        ## Rename Keys
        tweet = self._rename_keys(tweet)
        ## Subset
        col_subset = list(filter(lambda i: i in tweet, GENERAL_SCHEMA["twitter"]["tweet"].values()))
        tweet = {c:tweet.get(c) for c in col_subset}
        return tweet

    def _process_tweets(self,
                        tweet_data):
        """
        Process raw tweet data and cache in a processed form

        Args:
            tweet_data (list):  list of raw tweets. Expected to contain all tweets
                    desired for a single individual

        Returns:
            formatted_data (list): List of processed dictionaries

        """
        ## Null Data
        if len(tweet_data) == 0:
            return []
        ## Process Tweets
        tweet_data = list(map(self._process_tweet, tweet_data))
        ## Filter Null
        tweet_data = list(filter(lambda i: i is not None, tweet_data))
        return tweet_data

    def _process_reddit_comment(self,
                                comment):
        """

        """
        ## Check Null
        if comment is None:
            return None
        ## Early Return
        if not self._run_pipeline:
            comment = self._rename_keys(comment)
            return comment
        ## Identify Text Column
        text_col = None
        for text_col_opt in ["body","text"]:
            if text_col_opt in comment:
                text_col = text_col_opt
        if text_col is None or comment.get(text_col) is None:
            return None
        ## Tokenization
        comment["sentences"] = sent_tokenize(comment[text_col])
        comment["sentences_tokenized"] = list(map(tokenizer.tokenize, comment["sentences"]))
        comment["text_tokenized"] = flatten(comment["sentences_tokenized"])
        ## Add Meta
        if "source" not in comment:
            if "id" in comment:
                comment["source"] = comment["id"]
            else:
                comment["source"] = None
        comment["entity_type"] = "comment"
        comment["date_processed_utc"] = int(datetime.utcnow().timestamp())
        ## Rename Keys
        comment = self._rename_keys(comment)
        ## Subset
        col_subset = list(filter(lambda i: i in comment, GENERAL_SCHEMA["reddit"]["comment"].values()))
        comment = {c:comment.get(c) for c in col_subset}
        return comment

    def _process_reddit_comments(self,
                                 comment_data):
        """
        Process raw reddit data and cache in a processed form

        Args:
            comment_data (list) list of raw dictionaries. Expected to contain all comments
                    desired for a single individual

        Returns:
            formatted_data (list): List of processed dictionaries

        """
        ## Data Size Check
        if len(comment_data) == 0:
            return []
        ## Process
        comment_data = list(map(self._process_reddit_comment, comment_data))
        ## Filter Nulls
        comment_data = list(filter(lambda i: i is not None, comment_data))
        return comment_data
    
    def load(self,
             filename,
             min_date=None,
             max_date=None,
             sample_rate=None,
             sample_size=None,
             filters=None):
        """

        """
        ## Load Data
        data = self._load_file(filename,
                               min_date=min_date,
                               max_date=max_date,
                               sample_rate=sample_rate,
                               sample_size=sample_size,
                               filters=filters)
        return data
    
    def load_and_process(self,
                         filename,
                         min_date=None,
                         max_date=None,
                         sample_rate=None,
                         sample_size=None,
                         load_filters=None,
                         process_filters=None):
        """

        """
        ## Load Data
        data = self._load_file(filename,
                               min_date=min_date,
                               max_date=max_date,
                               sample_rate=sample_rate,
                               sample_size=sample_size,
                               filters=load_filters)
        ## Processing
        if self.platform == "twitter":
            data = self._process_tweets(data)
        elif self.platform == "reddit":
            data = self._process_reddit_comments(data)
        else:
            raise ValueError("Did not recognize platform: `{}`".format(self.platform))
        ## Procssed Data Filtering
        if process_filters is not None:
            data = list(map(lambda d: self._filter_post(d, process_filters), data))
        return data
