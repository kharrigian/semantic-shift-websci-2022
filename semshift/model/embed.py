
"""
Methods for learning word embeddings

```
## Imports
from glob import glob
from semshift.model.embed import Word2Vec
from semshift.model.vocab import Vocabulary
from semshift.model.feature_extractors import EmbeddingTransformer

## Identify Subset of Processed Filenames
filenames = sorted(glob("./data/processed/twitter/qntfy/*.tweets.tar.gz"))
filenames = filenames[:500]

## Parameterize Model and Call Fit Procedure
model = Word2Vec(dim=100,
                 window=5,
                 sample=0.001,
                 jobs=8,
                 sg=0,
                 hs=0,
                 negative=5,
                 max_iter=10,
                 compute_loss=True,
                 max_n_gram=1,
                 phrase_threshold=10,
                 min_vocab_count=5,
                 max_vocab_size=500000,
                 stream_kwargs={
                            "mode":0,
                            "jobs":8,
                            },
                 random_seed=42,
                 verbose=True)
model = model.fit(filenames)

## Get Similar Terms Based on Cosine Distance
terms = model.get_similar("depression",25)

## Save Model and Write Embeddings
_ = model.save("./data/resources/word2vec/qntfy/")
_ = model.write("./data/resources/word2vec/qntfy/embeddings")

## Load Existing Model
model = Word2Vec.load("./data/resources/word2vec/qntfy/")

## Use Custom Embeddings in a Feature Transformer
vocabulary = Vocabulary()
vocabulary = vocabulary.assign(model.model.wv.index_to_key)
transformer = EmbeddingTransformer(vocab=vocabulary,
                                   dim="./data/resources/word2vec/qntfy/embeddings.100d.txt")
```
"""

####################
### Imports
####################

## Standard Library
import os
import json
from glob import glob
from datetime import datetime
from collections import Counter

## External Libraries
import joblib
import numpy as np
from gensim.models import Phrases
from gensim.models.word2vec import Word2Vec as W2V

## Local Imports
from .data_loaders import PostStream, PostStreamPipeline
from ..util.logging import initialize_logger

#######################
### Globals
#######################

## Initialize Logger
LOGGER = initialize_logger()

#######################
### Class
#######################

class Word2Vec(object):

    """
    
    """

    def __init__(self,
                 dim=100,
                 window=5,
                 sample=0.001,
                 jobs=1,
                 sg=0,
                 hs=0,
                 negative=5,
                 max_iter=10,
                 compute_loss=True,
                 max_n_gram=1,
                 phrase_threshold=10,
                 min_vocab_count=5,
                 max_vocab_size=500000,
                 stream_kwargs={},
                 random_seed=42,
                 verbose=False,
                 **w2v_kwargs):
        """
        
        """
        ## Word2Vec Arguments
        self._dim = dim
        self._window = window
        self._sample = sample
        self._jobs = jobs
        self._sg = sg
        self._hs = hs
        self._negative = negative
        self._max_iter = max_iter
        self._compute_loss = compute_loss
        self._random_seed = random_seed
        self._w2v_kwargs = w2v_kwargs
        ## Vocabulary/Phrasing Arguments
        self._max_n_gram = max_n_gram
        self._phrase_threshold = phrase_threshold
        self._min_vocab_count = min_vocab_count
        self._max_vocab_size = max_vocab_size
        self._stream_kwargs = stream_kwargs
        ## Working Variables
        self.model = None
        self.phrasers = None
        self._vocabulary = None
        ## Class Arguments
        self.verbose = verbose
        ## Check Initialization
        self._check_init()
        ## Assign Initialization Time
        self._init_time = datetime.now()
    
    def __repr__(self):
        """
        
        """
        return f"Word2Vec(dim={self._dim}, sg={self._sg})"
    
    def _check_init(self):
        """
        
        """
        if isinstance(self._stream_kwargs, dict) and self._stream_kwargs.get("mode") is not None:
            assert self._stream_kwargs.get("mode") in set([0, 1])
        elif isinstance(self._stream_kwargs, dict) and self._stream_kwargs.get("metadata") is not None:
            _ = self._stream_kwargs.pop("metadata",None)
            LOGGER.warning("Removing metadata field from stream kwargs.")
        elif isinstance(self._stream_kwargs, dict):
            self._stream_kwargs["jobs"] = self._jobs
    
    def assign_phrasers(self,
                        phrasers):
        """
        
        """
        self.phrasers = phrasers
        return self
        
    def _learn_phrasers(self,
                        filenames):
        """
        
        """
        ## Check for Existing Phrasers
        if self.phrasers is not None:
            LOGGER.info("Phrasers already present. Skipping phrase learning procedure.")
            return self.phrasers
        ## Learn Vocabulary
        if not isinstance(filenames, PostStream) and not isinstance(filenames, PostStreamPipeline):
            vocab_stream = PostStream(filenames,
                                      verbose=self.verbose,
                                      **self._stream_kwargs)
        else:
            vocab_stream = filenames
        ## Cache Initialization
        if vocab_stream.cache_dir is not None:
            if self.verbose:
                LOGGER.info("Initializing Cache of Unphrased Data")
            _ = vocab_stream.build_cache()
        ## User Updating
        if self.verbose:
            if self._max_n_gram > 1:
                LOGGER.info("Learning Initial Vocabulary (1-2 Grams)")
            else:
                LOGGER.info("Learning Initial Vocabulary")
        ## Format Stream
        verbosity = vocab_stream.verbose if hasattr(vocab_stream, "verbose") else False
        if hasattr(vocab_stream, "verbose"):
            vocab_stream.verbose = False
        if isinstance(vocab_stream, PostStreamPipeline):
            for stream in vocab_stream.streams:
                stream.verbose = False
        ## Initial Phrase Detection
        phrasers =  [Phrases(sentences=vocab_stream,
                             min_count=self._min_vocab_count,
                             max_vocab_size=self._max_vocab_size,
                             threshold=self._phrase_threshold if self._max_n_gram > 1 else np.inf,
                             delimiter=" ")]
        ## Freeze
        phrasers[-1] = phrasers[-1].freeze()
        ## Initialize Phrased Stream
        phrased_vocab_stream = phrasers[-1][vocab_stream]
        ## Get Additional N-Grams
        current_n = 2
        while current_n < self._max_n_gram:
            ## Update User
            if self.verbose:
                LOGGER.info(f"Learning {current_n+1}-grams")
            ## Learn Phrasers
            phrasers.append(Phrases(phrased_vocab_stream,
                                    min_count=self._min_vocab_count,
                                    max_vocab_size=self._max_vocab_size,
                                    threshold=self._phrase_threshold,
                                    delimiter=" "))
            current_n += 1
            ## Freeze
            phrasers[-1] = phrasers[-1].freeze()
            ## Update Phrased Stream
            phrased_vocab_stream = phrasers[-1][phrased_vocab_stream]
        ## Alert User
        if self.verbose:
            LOGGER.info("Phrase Learning Complete")
        ## Remove Cache (if It Exists)
        if isinstance(vocab_stream, PostStream) or isinstance(vocab_stream, PostStreamPipeline):
            if self.verbose and vocab_stream.cache_dir is not None:
                LOGGER.info("Erasing Cache of Unphrased Data")
            _ = vocab_stream.remove_cache(make_dir=False)
        ## Reformat Stream
        if hasattr(vocab_stream, "verbose"):
            vocab_stream.verbose = verbosity
        if isinstance(vocab_stream, PostStreamPipeline):
            for stream in vocab_stream.streams:
                stream.verbose = verbosity
        return phrasers
    
    def _extract_vocabulary(self,
                            w2v_model):
        """
        
        """
        ## Parameters
        min_count = self._min_vocab_count if self._min_vocab_count is not None else 0
        ## Extract Vocabulary Meeting Class Parameters (Either Independent or From Trained Model)
        if len(w2v_model.wv.vectors) == 0:
            vocabulary = Counter({x:y for x, y in w2v_model.raw_vocab.items() if y >= min_count})
            if self._max_vocab_size is not None:
                vocabulary = Counter({x:y for x, y in vocabulary.most_common(self._max_vocab_size)})
            ## Format
            vocabulary = dict(vocabulary)
        else:
            ## Get Terms and Frequency
            terms = w2v_model.wv.index_to_key
            frequency = [w2v_model.wv.get_vecattr(v,"count") for v in terms]
            ## Format
            vocabulary = dict(zip(terms, frequency))
        ## Return
        return vocabulary
    
    def _prepare_and_extract_vocabulary(self,
                                        text_stream):
        """
        
        """
        ## Initialize Word2Vec Class without Training Parameters
        vlearner = W2V(min_count=self._min_vocab_count,
                       max_vocab_size=self._max_vocab_size,
                       workers=self._jobs)
        ## Scan the Stream
        _, _ = vlearner.scan_vocab(text_stream)
        ## Extract Vocabulary and Cache
        self._vocabulary = self._extract_vocabulary(vlearner)

    def fit(self,
            filenames,
            phrase_learning_only=False,
            extract_vocabulary=False):
        """
        
        """
        ## Learn Vocabulary + Phrases
        if self.verbose:
            LOGGER.info("Beginning Phrase Learning")
        self.phrasers = self._learn_phrasers(filenames)
        ## Initialize Iterable for Embedding Learning
        if isinstance(filenames, PostStream):
            text_stream = filenames
            text_stream.phrasers = self.phrasers
        elif isinstance(filenames, PostStreamPipeline):
            for stream in filenames.streams:
                stream.phrasers = self.phrasers
            text_stream = filenames
        else:
            text_stream = PostStream(filenames,
                                     verbose=self.verbose,
                                     phrasers=self.phrasers,
                                     **self._stream_kwargs)
        ## Initialize Phrased Cache
        if (phrase_learning_only and extract_vocabulary) or not phrase_learning_only:
            if text_stream.cache_dir is not None:
                if self.verbose:
                    LOGGER.info("Initializing Phrased Cache")
                _ = text_stream.build_cache()
        ## Extract Vocabulary Before Saving (if Not Running Training Procedure)
        if extract_vocabulary and phrase_learning_only:
            LOGGER.info("Extracting Vocabulary")
            _ = self._prepare_and_extract_vocabulary(text_stream=text_stream)
        ## Early Exit (No Embedding Learning)
        if phrase_learning_only:
            ## Alert User
            if self.verbose:
                LOGGER.info("Phrase Learning Complete. Embedding learning procedured turned off.")
            ## Cache Removal
            if text_stream.cache_dir is not None:
                if self.verbose:
                    LOGGER.info("Erasing Cache of Phrased Data")
                _ = text_stream.remove_cache(make_dir=False)
            ## Exit
            return self
        ## Turn Of Verbosity for Word2Vec Training
        if isinstance(text_stream, PostStream):
            text_stream.verbose = False
        elif isinstance(text_stream, PostStreamPipeline):
            for stream in text_stream.streams:
                stream.verbose = False
        else:
            raise TypeError("Text Stream is expected to be either a PostStream or PostStream Pipeline.")
        ## Fit Model
        if self.verbose:
            LOGGER.info("Beginning Model Fit Procedure")
        self.model = W2V(sentences=text_stream,
                         min_count=self._min_vocab_count,
                         max_vocab_size=self._max_vocab_size,
                         vector_size=self._dim,
                         window=self._window,
                         sample=self._sample,
                         workers=self._jobs,
                         sg=self._sg,
                         hs=self._hs,
                         epochs=self._max_iter,
                         compute_loss=self._compute_loss,
                         negative=self._negative,
                         seed=self._random_seed,
                         **self._w2v_kwargs)
        ## Extract Vocabulary
        LOGGER.info("Fit Procedure Complete. Extracting Vocabulary.")
        self._vocabulary = self._extract_vocabulary(self.model)
        ## Cache Removal
        if text_stream.cache_dir is not None:
            if self.verbose:
                LOGGER.info("Erasing Cache of Phrased Data")
            _ = text_stream.remove_cache(make_dir=False)
        ## Done
        return self
    
    def get_ordered_vocabulary(self):
        """
        
        """
        return self.model.wv.index_to_key
    
    def get_similar(self,
                    word,
                    top_k=10):
        """
        
        """
        return self.model.wv.similar_by_word(word, top_k)
    
    def save(self,
             directory):
        """
        
        """
        if not os.path.exists(directory):
            _ = os.makedirs(directory)
        ## Save Phrasers and Model
        phraser_files = []
        for p, phraser in enumerate(self.phrasers):
            phraser.save(f"{directory}/phraser.{p}")
            phraser_files.append(f"{directory}/phraser.{p}")
        ## Save Vocabulary
        if hasattr(self, "_vocabulary") and self._vocabulary is not None:
            terms = self._vocabulary.keys()
            with open(f"{directory}/vocabulary.txt","w") as the_file:
                for term in terms:
                    the_file.write(f"{term}\n")
        ## Check for Model
        model_exists = False
        if self.model is not None:
            model_exists = True
            self.model.save(f"{directory}/word2vec.model")
        ## Set None and Dump
        self.phrasers = None
        self.model = None
        _ = joblib.dump(self, f"{directory}/word2vec.class")
        ## Reload Attributes
        if model_exists:
            self.model = W2V.load(f"{directory}/word2vec.model")
        self.phrasers = [Phrases.load(pf) for pf in phraser_files]
    
    @staticmethod
    def load_phrasers(directory):
        """
        
        """
        phraser_files = sorted(glob(f"{directory}/phraser.*"), key=lambda x: int(x.split("/")[-1].split(".")[-1]))
        phrasers = [Phrases.load(pf) for pf in phraser_files]
        return phrasers

    @staticmethod
    def load(directory):
        """
        
        """
        ## Check Existence
        if not os.path.exists(directory):
            raise ValueError(f"Model not found at path: {directory}")
        ## Load Class
        w2v = joblib.load(f"{directory}word2vec.class")
        ## Load Model
        phraser_files = sorted(glob(f"{directory}/phraser.*"), key=lambda x: int(x.split("/")[-1].split(".")[-1]))
        w2v.phrasers = [Phrases.load(pf) for pf in phraser_files]
        if os.path.exists(f"{directory}word2vec.model"):
            w2v.model = W2V.load(f"{directory}word2vec.model")
        else:
            LOGGER.warning("Warning: Model file does not exist.")
            w2v.model = None
        return w2v

    def write(self,
              filename):
        """
        Write embeddings to a file using the same format found in the
        pretrained gloVe embeddings
        """
        if self.model is None:
            if self.verbose:
                LOGGER.info("Model has not been fit. No embeddings will be saved.")
            return None
        if not filename.endswith(".txt"):
            filename = f"{filename}.txt"
        filename = filename.replace(".txt",f".{self._dim}d.txt")
        _ = self.model.wv.save_word2vec_format(filename, write_header=False)

