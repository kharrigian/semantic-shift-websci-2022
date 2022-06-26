

##################
### Imports
##################

## Standard Library
import os
import sys
from functools import partial

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import (csr_matrix, vstack)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

## Local Modules
from .vocab import Vocabulary
from ..util.multiprocessing import MyPool as Pool
from ..util.helpers import flatten

##################
### Vectorization
##################

class File2Vec(object):
    
    """
    Preprocessed Data to Vectorization
    """

    def __init__(self,
                 vocab_kwargs={},
                 favor_dense=False,
                 processor=None,
                 processor_kwargs={}):
        """
        File2Vec transforms preprocessed (e.g. tokenized) data files into a clean
        vector format that can be used for learning.

        Args:
            vocab_kwargs (dict): Arguments to pass to vocabulary class
            favor_dense (bool): If True, vectorization results in dense array instead 
                                of csr sparse matrix. Useful because certain preprocessing
                                steps (e.g. Standardization) and classifiers require 
                                dense data representations for learning.
        """
        ## Cache kwargs
        self._vocab_kwargs = vocab_kwargs
        self._processor = processor
        self._processor_kwargs = processor_kwargs
        ## Other Class Attributes
        self._favor_dense = favor_dense
        ## Initialize Vocabulary Class
        for param in ["_processor","_processor_kwargs"]:
            if hasattr(self, param) and getattr(self, param) and param not in vocab_kwargs:
                vocab_kwargs[param[1:]] = getattr(self, param)
        self.vocab = Vocabulary(**vocab_kwargs)
        ## Workspace Variables
        self._count2vec = None
    
    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        desc = f"File2Vec(vocab_kwargs={self._vocab_kwargs}, favor_dense={self._favor_dense})"
        return desc
    
    def _initialize_dict_vectorizer(self):
        """
        Initialize a vectorizer that transforms a counter dictionary
        into a sparse vector of counts (with a uniform feature index)

        Args:
            None
        
        Returns:
            None, initializes self._count2vec inplace.
        """
        self._count2vec = DictVectorizer(separator=":", dtype=int)
        self._count2vec.vocabulary_ = self.vocab.ngram_to_idx.copy()
        rev_dict = dict((y, x) for x, y in self.vocab.ngram_to_idx.items())
        self._count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
        return

    def _vectorize_labels(self,
                          keys,
                          label_dict,
                          pos_class="depression"):
        """
        Create a 1d-array with class labels for the model
        based on filenames and a label dictionary.

        Args:
            keys (list): Ordered list of keys for label_dict
            label_dict (dict): Map between keys and label
            pos_class (str): Positive Class Name. Default is "depression". Users with "control" will 
                             be labeled as 0. Users without either will be labeled as -1
        
        Returns:
            labels (1d-array): Vector of class targets.
        """
        labels = np.array(list(map(lambda f: 1 if label_dict[f]==pos_class else 0 if label_dict[f]=="control" else -1, keys)))
        labels = labels.astype(int)
        return labels

    def _vectorize_single_file(self,
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
                               return_users=False,
                               ngrams=None):
        """
        Vectorize the tokens in a preprocessed data file.

        Args:
            filename (str): Path to preprocessed text data
            min_date (str): Lower data boundary. ISO-format.
            max_date (str): Upper date boundary. ISO-format
            n_samples (int or None): Number of post samples
            randomized (bool): If True and n_samples is not None,
                               sample randomly. Otherwise, samples
                               most recent posts
            return_post_counts (bool): If True, return the number of posts used
                                       to generate the vector as well
            resolution (None or str)
            reference_point (None or int)
            chunk_n_samples (None, int, or float)
            chunk_randomized (bool)
            return_references (bool)

        Returns:
            filename (str): Input filename
            vec (sparse vector): Vector of feature counts
        """
        ## Get N-Gram Range
        if ngrams is None:
            ngrams = [min(self.vocab._ngrams[0], self.vocab._external_ngrams[0]),
                      max(self.vocab._ngrams[1], self.vocab._external_ngrams[1])]
        ## Load File Data, Count Tokens
        token_counts = self.vocab._load_and_count(filename,
                                                  min_date=min_date,
                                                  max_date=max_date,
                                                  n_samples=n_samples,
                                                  randomized=randomized,
                                                  return_post_counts=return_post_counts,
                                                  resolution=resolution,
                                                  reference_point=reference_point,
                                                  chunk_n_samples=chunk_n_samples,
                                                  chunk_randomized=chunk_randomized,
                                                  return_references=return_references,
                                                  ngrams=ngrams)
        if return_post_counts and return_references:
            token_counts, n_posts, refs = token_counts
            users = [[t[0]]*len(t[1]) for t in token_counts]
            token_counts = [t[1] for t in token_counts]
            n_posts = [n[1] for n in n_posts]
            refs = [r[1] for r in refs]
        elif return_post_counts:
            token_counts, n_posts = token_counts
            users = [[t[0]]*len(t[1]) for t in token_counts]
            token_counts = [t[1] for t in token_counts]
            n_posts = [n[1] for n in n_posts]
        elif return_references:
            token_counts, refs = token_counts
            users = [[t[0]]*len(t[1]) for t in token_counts]
            token_counts = [t[1] for t in token_counts]
            refs = [r[1] for r in refs]
        else:
            users = [[t[0]]*len(t[1]) for t in token_counts]
            token_counts = [t[1] for t in token_counts]
        ## Vectorize
        if len(token_counts) == 0:
            vec = [self._count2vec.transform([{}]) for u in users]
        else:
            vec = [self._count2vec.transform(tc) for tc in token_counts]
        ## Prepare Return
        return_vals = [filename, vec]
        if return_post_counts:
            return_vals.append(n_posts)
        if return_references:
            return_vals.append(refs)
        if return_users:
            return_vals.append(users)
        return return_vals

    def _vectorize_files(self,
                         filenames,
                         jobs=4,
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
                         return_users=False,
                         ngrams=None):
        """
        Vectorize several files tokens using multiprocessing.

        Args:
            filenames (list of str): Preprocessed text files
            jobs (int): Number of processes to use for vectorization
            min_date (str, datetime, or None): Date Lower Bound
            max_date (str, datetime, or None): Date Upper Bound
            n_samples (int or None): Number of post samples
            randomized (bool): If True and n_samples is not None,
                               sample randomly. Otherwise, samples
                               most recent posts
            return_post_counts (bool): If True, returns the number of individual
                                posts used to generate each feature vector.
            resolution (None or str):
            reference_point (None or int):
            chunk_n_samples (None, int, or float):
            chunk_randomized (bool):
            return_references (bool):

        Returns:
            filenames (list): List of filenames (in case order changed during
                              multiprocessing)
            vectors (array): Sparse or dense document-term matrix (based on
                             class intiialization parameters)
            n_posts (array, optional): Array of post counts for each filename
        """
        ## Date Boundaries
        if min_date is not None and isinstance(min_date,str):
            min_date = pd.to_datetime(min_date)
        if max_date is not None and isinstance(max_date,str):
            max_date = pd.to_datetime(max_date)
        ## Reference Point
        if reference_point is not None and isinstance(reference_point, str):
            reference_point = int(pd.to_datetime(reference_point).timestamp())
        ## Take Note of Initial Sorting
        file2ind = dict(zip(filenames, range(len(filenames))))
        ## Get Vectors
        vectorizer = partial(self._vectorize_single_file,
                             min_date=min_date,
                             max_date=max_date,
                             n_samples=n_samples,
                             randomized=randomized,
                             return_post_counts=return_post_counts,
                             resolution=resolution,
                             reference_point=reference_point,
                             chunk_n_samples=chunk_n_samples,
                             chunk_randomized=chunk_randomized,
                             return_references=return_references,
                             return_users=return_users,
                             ngrams=ngrams)
        with Pool(processes=jobs) as mp_pool:
            vectors = list(tqdm(mp_pool.imap_unordered(vectorizer,
                                                    filenames,
                                                    chunksize=min(100, max(len(filenames) // 10,1))),
                                total=len(filenames),
                                desc="Filecount",
                                file=sys.stdout))
        ## Ignore Missing Data
        vectors = [v for v in vectors if len(v[1]) > 0]
        if len(vectors) == 0:
            ## Assign Nulls
            X = None
            filenames = []
            users = []
            n_posts = []
            refs = []
        else:
            ## Clean Pool Result
            X = [vstack(v[1]) for v in vectors]
            filenames = flatten([[v[0]]*x.shape[0] for v, x in zip(vectors,X)])
            if return_users:
                users = [flatten(v[-1]) for v in vectors]
                if len(users) > 0 and isinstance(users[0], list):
                    users = flatten(users)
            if return_post_counts and return_references:
                n_posts = [flatten(v[2]) for v in vectors]
                refs = [flatten(v[3]) for v in vectors]
                if len(n_posts) > 0 and isinstance(n_posts[0], list):
                    n_posts = flatten(n_posts)
                if len(refs) > 0 and isinstance(refs[0], list):
                    refs = flatten(refs)
                n_posts = np.array(n_posts)
                refs = np.array(refs)
            elif not return_post_counts and not return_references:
                pass
            elif return_post_counts:
                n_posts = [flatten(v[2]) for v in vectors]
                if len(n_posts) > 0 and isinstance(n_posts[0], list):
                    n_posts = flatten(n_posts)
                n_posts = np.array(n_posts)
            elif return_references:
                refs = [flatten(v[2]) for v in vectors]
                if len(refs) > 0 and isinstance(refs[0], list):
                    refs = flatten(refs)
                refs = np.array(refs)
            X = vstack(X)
            ## Transform Into Dense
            if hasattr(self, "_favor_dense") and self._favor_dense and isinstance(X, csr_matrix):
                X = X.toarray()
        ## Get Sorting
        sorting = [i[0] for i in sorted(enumerate(filenames), key=lambda j: file2ind.get(j[1]))]
        ## Apply Sorting
        filenames = [filenames[s] for s in sorting]
        X = X[sorting]
        if return_post_counts:
            n_posts = n_posts[sorting]
        if return_references:
            refs = refs[sorting]
        if return_users:
            users = [users[s] for s in sorting]
        ## Logical Return
        return_vals = [filenames, X]
        if return_post_counts:
            return_vals.append(n_posts)
        if return_references:
            return_vals.append(refs)
        if return_users:
            return_vals.append(users)
        return return_vals