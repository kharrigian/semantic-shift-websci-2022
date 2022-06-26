
##################
### Imports
##################

## Standard Library
import sys
from copy import deepcopy
from collections import Counter

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid

## Local Modules
from .classifiers import *
from .file_vectorizer import File2Vec
from .feature_extractors import FeaturePreprocessor
from ..util.logging import initialize_logger

##################
### Globals
##################

## Create Logger
LOGGER = initialize_logger()

## Default Preprocessing Params (e.g. Feature Set)
DEFAULT_PREPROCESSING_KWARGS = {"feature_flags": {
                                                "bag_of_words":True
                                                 },
                                "feature_kwargs":{},
                                "standardize":True,
}

## Update Model Dict
CLASSIFIERS = {**MODEL_DICT}
CLASSIFIERS_PARAMS = {**DEFAULT_PARAMETERS}
CLASSIFIERS_PARAMETER_GRID = {**PARAMETER_GRID}

##################
### Modeling Class
##################

class MentalHealthClassifier(File2Vec):

    """
    Mental Health Status Classification
    """

    def __init__(self,
                 target_disorder,
                 model="logistic",
                 model_kwargs={},
                 vocab_kwargs={},
                 preprocessing_kwargs=DEFAULT_PREPROCESSING_KWARGS,
                 min_date=None,
                 max_date=None,
                 n_samples=None,
                 randomized=False,
                 resolution=None,
                 reference_point=None,
                 chunk_n_samples=None,
                 chunk_randomized=False,
                 drop_null=False,
                 vocab_fit_args={"chunksize":10,"in_memory":True},
                 jobs=4,
                 random_state=42):
        """
        Mental Health Status Classifier wrapper.

        Args:
            target_disorder (str): Target mental health disorder (e.g. depression, ptsd)
            model (str): Classifier to use for modeling. If using adaption, will expect appropriate fit method
            model_kwargs (dict): Arguments to pass to initialization of the classifier.
            vocab_kwargs (dict): Arguments to pass to Vocabulary class
            preprocessing_kwargs (dict): Arguments to pass to preprocessor of document-
                                         term matrix.
            min_date (str or None): ISO-format string representing lower bound in date range for
                                    training
            max_date (str or None): ISO-format string representing upper bound in date range for
                                    training 
            n_samples (int or None): Number of post samples to user for training. All if None.
            randomized (bool): If sampling posts, turning on this flag will sample randomly instead
                               of selecting the k most recent posts
            vocab_fit_args (dict): {"chunksize":int, "in_memory":bool}
            jobs (int): Number of processes to use during the model learning procedure.
            random_state (int): Random state. Default is 42
        """
        ## Class Attributes
        self._target_disorder = target_disorder
        self._min_date = min_date
        self._max_date = max_date
        self._n_samples = n_samples
        self._randomized = randomized
        self._resolution = resolution
        self._reference_point = reference_point
        self._chunk_n_samples = chunk_n_samples
        self._chunk_randomized = chunk_randomized
        self._drop_null = drop_null
        self._jobs = jobs
        self._vocab_fit_args = vocab_fit_args
        self._random_state = random_state
        self._model_type = model
        ## Initialize Class kwargs
        self._initialize_class_kwargs(vocab_kwargs,
                                      model,
                                      model_kwargs,
                                      preprocessing_kwargs)
        ## Initialize Date Boundaries
        self._initialize_date_bounds()

    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        if not hasattr(self, "_vocab_kwargs") or len(self._vocab_kwargs) == 0:
            vs = ""
        else:
            vs = ", ".join("{}={}".format(x,y) for x,y in self._vocab_kwargs.items())
        if not hasattr(self, "_preprocessing_kwargs") or len(self._preprocessing_kwargs) == 0:
            ps = ""
        else:
            ps = ", ".join("{}={}".format(x,y) for x,y in self._preprocessing_kwargs.items())
        if not hasattr(self, "_target_disorder"):
            self._target_disorder = None
        desc = "MentalHealthClassifier(target_disorder={}, model={}, {}, {})".format(
            self._target_disorder if hasattr(self, "_target_disorder") else None,
            self.model,
            vs,
            ps)
        desc = desc.replace(", ,","").replace(" )",")")
        return desc
    
    def _initialize_class_kwargs(self,
                                 vocab_kwargs,
                                 model,
                                 model_kwargs,
                                 preprocessing_kwargs):
        """
        Initialize class attributes, checking that random states
        are set uniformly across sub-classes.

        Args:
            vocab_kwargs (dict): Vocabulary parameters
            model (str): Name of the estimator to use
            model_kwargs (dict): Arguments to pass to estimator
            preprocessing_kwargs (dict): Feature set arguments.
        
        Returns:
            None, sets class attributes in class
        """
        ## Cache kwargs
        self._model = model
        self._vocab_kwargs = vocab_kwargs
        self._model_kwargs = model_kwargs
        self._preprocessing_kwargs = preprocessing_kwargs
        ## Initialize Classification Model
        if len(self._model_kwargs) == 0:
            self._model_kwargs = CLASSIFIERS_PARAMS.get(model)
        if "random_state" not in model_kwargs:
            self._model_kwargs["random_state"] = self._random_state
        else:
            self._model_kwargs["random_state"] = self._random_state
        if model == "mlp" and "hidden_layers" in model_kwargs:
            if isinstance(model_kwargs.get("hidden_layers"), list):
                self._model_kwargs["hidden_layers"] = tuple(self._model_kwargs["hidden_layers"])
        if "class_weight" in model_kwargs and isinstance(self._model_kwargs.get("class_weight"),dict):
            cw = {0:self._model_kwargs["class_weight"]["0"],
                  1:self._model_kwargs["class_weight"]["1"]}
            self._model_kwargs["class_weight"] = cw
        if model == "naive_bayes" and "random_state" in self._model_kwargs:
            _ = self._model_kwargs.pop("random_state", None)
        self.model = CLASSIFIERS.get(model)(**self._model_kwargs)
        ## Randomization in Vocabulary
        if "random_state" not in self._vocab_kwargs:
            self._vocab_kwargs["random_state"] = self._random_state
        ## File2Vec Inheritence Initialization
        super(MentalHealthClassifier, self).__init__(vocab_kwargs)

    def _initialize_date_bounds(self):
        """
        Initialize data boundaries as datetime objects

        Args:
            None
        
        Returns:
            None, updates _min_date and _max_date parameters
        """
        if not hasattr(self, "_min_date"):
            self._min_date = None
        if not hasattr(self, "_max_date"):
            self._max_date = None
        if self._min_date is not None:
            self._min_date = pd.to_datetime(self._min_date)
        if self._max_date is not None:
            self._max_date = pd.to_datetime(self._max_date)

    def _learn_vocabulary(self,
                          filenames):
        """
        Fit a Vocabulary class based on preprocessed user data
        files.

        Args:
            filenames (list of str): Path to files to use for constructing
                                     the vocabulary.
        
        Returns:
            None, sets self.vocab in place.
        """
        ## Learn Vocabulary
        LOGGER.info("Learning Vocabulary")
        self.vocab = self.vocab.fit(filenames,
                                    chunksize=self._vocab_fit_args.get("chunksize",10) if hasattr(self, "_vocab_fit_args") else 10,
                                    jobs=self._jobs if hasattr(self, "_jobs") else 1,
                                    min_date=self._min_date if hasattr(self, "_min_date") else None,
                                    max_date=self._max_date if hasattr(self, "_max_date") else None,
                                    n_samples=self._n_samples if hasattr(self, "_n_samples") else None,
                                    randomized=self._randomized if hasattr(self, "_randomized") else False,
                                    resolution=self._resolution if hasattr(self, "_resolution") else None,
                                    reference_point=self._reference_point if hasattr(self,"_reference_point") else None,
                                    chunk_n_samples=self._chunk_n_samples if hasattr(self,"_chunk_n_samples") else None,
                                    chunk_randomized=self._chunk_randomized if hasattr(self,"_chunk_randomized") else False,
                                    in_memory=self._vocab_fit_args.get("in_memory",True) if hasattr(self, "_vocab_fit_args") else True
                                    )
        ## Initialize Dict Vectorizer
        _ = self._initialize_dict_vectorizer()
    
    def get_feature_names(self,
                          preprocessor=None):
        """
        Extract model feature names

        Args:
            None
        
        Returns:
            features (list): Named list of features in the processed feature matrix.
        """
        ## Isolate Appropriate Preprocessor
        if preprocessor is None:
            preprocessor = self.preprocessor
        ## Get Features
        features = []
        if not hasattr(preprocessor, "_transformer_names"):
            preprocessor._transformer_names = []
        for t in preprocessor._transformer_names:
            transformer = preprocessor._transformers[t]
            if t in ["bag_of_words","tfidf"]:
                tf = self.vocab.get_ordered_vocabulary()
            elif t == "glove":
                tf =  list(map(lambda i: f"GloVe_Dim_{i+1}", range(transformer.dim)))
            elif t == "liwc":
                tf = [f"LIWC={n}" for n in transformer.names]
            elif t == "lda":
                tf = [f"LDA_TOPIC_{i+1}" for i in range(transformer.n_components)] 
            features.extend(tf)
        if hasattr(preprocessor, "_min_variance") and preprocessor._min_variance is not None:
            feature_mask = preprocessor._transformers["variance_threshold"].get_support()
            features = [f for f, m in zip(features, feature_mask) if m]
        return features

    def _format_filenames(self,
                          filenames):
        """

        """
        if len(filenames) == 0:
            return filenames
        file_count = Counter()
        new_filenames = []
        for f in filenames:
            file_count[f] += 1
            new_filenames.append((f, file_count[f]))
        if file_count.most_common(1)[0][1] == 1:
            new_filenames = filenames
        return new_filenames

    def _load_vectors(self,
                      filenames,
                      label_dict=None,
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
        Load the raw document-term matrix and (optionally)
        associated label vector for a list of user data files.

        Args:
            filenames (list of str): Preprocessed user data files
            label_dict (None or dict): If desired, load user labels
                                       as a vector.
            min_date (str, datetime, or None): Lower date bound
            max_date (str, datetime, or None): Upper date bound
            n_samples (int or None): Possible number of samples
            randomized (bool): Whether to use random sample selection instead
                               of most recent
            return_post_counts (bool): If True, return the number of posts used
                                    to generate each feature vector
            resolution ():
            reference_point ():
            chunk_n_samples ():
            chunk_randomized ():
            return_references ():
            return_users ():

        Returns:
            filenames (list of str): List of filenames associated with 
                                     rows in the feature matrix.
            X (2d-array): Raw document-term matrix.
            y (1d-array or None): Target classes if label_dict passed.
        """
        ## Check Attributes
        if not hasattr(self, "_target_disorder"):
            raise AttributeError("Did not find _target_disorder attribute in model class.")
        ## Vectorize the data (count-based)
        result = self._vectorize_files(filenames,
                                       self._jobs,
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
        ## Format Labels
        if label_dict is not None:
            LOGGER.info("Formatting Labels")
            y = self._vectorize_labels(result[0],
                                       label_dict,
                                       pos_class=self._target_disorder)
            if y.shape[0] != result[1].shape[0]:
                raise Exception("Learned label length not equal to feature matrix shape")
        else:
            y = None
        ## Prepare Return
        return_vals = result[:2] + [y] + result[2:]
        return return_vals

    def _fit(self,
             X_train,
             y_train,
             preprocessor=None,
             primary=True,
             verbose=False):
        """
        
        """
        ## Preprocessing
        if verbose:
            LOGGER.info("Generating Feature Set")
        if preprocessor is None:
            preprocessor = FeaturePreprocessor(vocab=self.vocab,
                                               feature_flags=self._preprocessing_kwargs.get("feature_flags") if hasattr(self, "_preprocessing_kwargs") else {},
                                               feature_kwargs=self._preprocessing_kwargs.get("feature_kwargs") if hasattr(self, "_preprocessing_kwargs") else {},
                                               standardize=self._preprocessing_kwargs.get("standardize") if hasattr(self, "_preprocessing_kwargs") else False,
                                               min_variance=self._preprocessing_kwargs.get("min_variance") if hasattr(self, "_preprocessing_kwargs") else None,
                                               verbose=primary)
            X_train = preprocessor.fit_transform(X_train)
        else:
            X_train = preprocessor.transform(X_train)
        ## Fit Model
        if verbose:
            LOGGER.info("Fitting Classifier")
        if primary:
            model = self.model
        else:
            model = CLASSIFIERS.get(self._model_type)(**self._model_kwargs)
        model = model.fit(X_train, y_train)
        return preprocessor, model
    
    def _fit_multisample(self,
                         X_train,
                         y_train,
                         sample_inds,
                         preprocessor=None,
                         verbose=True):
        """
        
        """
        preprocessors = []
        models = []
        if verbose:
            wrapper = lambda x: tqdm(x, total=len(x), desc="Fitting Resampled Models", file=sys.stdout)
        else:
            wrapper = lambda x: x
        for ind in wrapper(sample_inds):
            _p, _m = self._fit(X_train[ind], y_train[ind], preprocessor=preprocessor, primary=False, verbose=False)
            preprocessors.append(_p)
            models.append(_m)
        return preprocessors, models

    def fit(self,
            train_files,
            label_dict,
            return_training_preds=True,
            resampling_iter=None,
            resample_rate=1):
        """
        Args:
            train_files (list of str): Paths to the training files
            label_dict (dict): Map between filename and training label
            return_training_preds (bool): If True, return training predictions in addition
                    to the model class itself. Useful for saving vectorization time.
            resampling_iter (int or None): If desired, resample document term matrix
                                           and train this many models with independent 
                                           feature preprocessing for each iteration.
            resample_rate (float [0,1]): If 1 (default), resample with replacement. Otherwise, use this
                                         proportion of data during each resampling iteration (without replacement)
        
        Returns:
            self: Base class with trained classifier.
            y_pred (dict): Training predictions, mapping between filename and model
                           prediction. Prediction is either a class (1 = target mental health status)
                           or positive mental health status probability based on availability of probs
                           in the base classifier. If resampling turned on, will also return a secondary
                           dictionary with resampled predictions
        """
        ## Learn Vocabulary in Training Files
        _ = self._learn_vocabulary(train_files)
        ## N Gram Range
        ngrams = [min(self.vocab._ngrams[0], self.vocab._external_ngrams[0]),
                  max(self.vocab._ngrams[1], self.vocab._external_ngrams[1])]
        ## Load Vectors From Disk
        LOGGER.info("Vectorizing Training Data")
        train_files, X_train, y_train = self._load_vectors(train_files,
                                                           label_dict,
                                                           min_date=self._min_date if hasattr(self, "_min_date") else None,
                                                           max_date=self._max_date if hasattr(self, "_max_date") else None,
                                                           n_samples=self._n_samples if hasattr(self, "_n_samples") else None,
                                                           randomized=self._randomized if hasattr(self, "_randomized") else False,
                                                           resolution=self._resolution if hasattr(self,"_resolution") else None,
                                                           reference_point=self._reference_point if hasattr(self,"_reference_point") else None,
                                                           chunk_n_samples=self._chunk_n_samples if hasattr(self,"_chunk_n_samples") else None,
                                                           chunk_randomized=self._chunk_randomized if hasattr(self,"_chunk_randomized") else False,
                                                           ngrams=ngrams
                                                           )
        ## Ensure Vocabulary Exists
        if not hasattr(self, "vocab") or self.vocab is None:
            raise AttributeError("vocab attribute is missing from model class")
        ## Alert User to Null Feature Sets
        if isinstance(X_train, csr_matrix):
            null_rows = X_train.getnnz(axis=1) == 0
        else:
            null_rows = (X_train==0).all(axis=1)
        LOGGER.info("Found {}/{} Null Training Rows ({} Control, {} Target)".format(
                    null_rows.sum(),
                    len(null_rows),
                    null_rows.sum()-y_train[null_rows].sum(),
                    y_train[null_rows].sum()
        ))
        ## Drop Null Rows During Training if Desired
        if hasattr(self, "_drop_null") and self._drop_null:
            LOGGER.info("Dropping null data points")
            mask = np.nonzero(null_rows==0)[0]
            X_train = X_train[mask]
            y_train = y_train[mask]
            train_files = [train_files[i] for i in mask]
        ## Format Files (Tuples When There is Aggregation)
        train_files = self._format_filenames(train_files)
        ## Preprocessing and Fitting (Primary Classifier)
        self.preprocessor, self.model = self._fit(X_train, y_train, primary=True, verbose=True)
        ## Preprocessing and Fitting (Resampling)
        if resampling_iter is not None:
            rseed = np.random.RandomState(self._random_state)
            if resample_rate == 1:
                rinds = rseed.choice(X_train.shape[0], size=(resampling_iter, X_train.shape[0]), replace=True)
            else:
                nr = int(X_train.shape[0] * resample_rate)
                rinds = rseed.choice(X_train.shape[0], size=(resampling_iter, nr), replace=False)
            self.preprocessor._resampled, self.model._resampled = self._fit_multisample(X_train,
                                                                                        y_train,
                                                                                        preprocessor=None,
                                                                                        sample_inds=rinds,
                                                                                        verbose=True)
        ## Return Training Predictions if Desired
        if return_training_preds:
            LOGGER.info("Making predictions on training data")
            ## Primary Predictions
            y_pred = self._predict(X_train, self.model, preprocessor=self.preprocessor)
            y_pred = dict((x, y) for x,y in zip(train_files, y_pred))
            ## Exit or Make Predictions Using Resampled Models
            if resampling_iter is None:
                return self, y_pred
            LOGGER.info("Making predictions on training data using resampled models")
            y_pred_resample = self._predict_multisample(preprocessors=self.preprocessor._resampled,
                                                        models=self.model._resampled,
                                                        X=X_train,
                                                        verbose=True)
            y_pred_resample = dict((x, list(y)) for x, y in zip(train_files, y_pred_resample))
            return self, y_pred, y_pred_resample
        return self
    
    def _predict(self,
                 X,
                 model,
                 preprocessor=None):
        """
        
        """
        if preprocessor is not None:
            X = preprocessor.transform(X)
        try:
            y_pred = model.predict_proba(X)[:, 1]
        except:
            y_pred = list(map(int, model.predict(X)))
        return y_pred
    
    def _predict_multisample(self,
                             preprocessors,
                             models,
                             X,
                             verbose=False):
        """
        
        """
        ## Check Input
        assert len(preprocessors) == len(models)
        ## Wrapper
        if verbose:
            wrapper = lambda x, l: tqdm(x, desc="Making Predictions", file=sys.stdout, total=l)
        else:
            wrapper = lambda x, l: x
        ## Cache of Predictions
        y_pred = np.zeros((X.shape[0], len(preprocessors)))
        ## Make Predictions
        for i, (_p, _m) in wrapper(enumerate(zip(preprocessors, models)), len(preprocessors)):
            y_pred[:,i] = self._predict(X, model=_m, preprocessor=_p)
        return y_pred

    def predict(self,
                test_files,
                min_date=None,
                max_date=None,
                n_samples=None,
                randomized=False,
                drop_null=False,
                resolution=None,
                reference_point=None,
                chunk_n_samples=None,
                chunk_randomized=False):
        """
        Make mental health status predictions on a list of test user files.

        Args:
            test_files (list of str): Preprocessed user data files.
            min_date (str, datetime, or None): Lower date bound
            max_date (str, datetime, or None): Upper date bound
            n_samples (int or None): Number of post-level samples to consider
            randomized (bool): If True, sample randomly instead of most recent
            drop_null (bool): If True, do not make predictions for rows without any matched vocab
            resolution (None or str):
            reference_point (None or int):
            chunk_n_samples ():
            chunk_randomized ()
        
        Returns:
            y_pred (dict): Mental health status probability or class prediction
                           depending on base classifier availability. Maps
                           between filename and prediction.
        """
        ## Date Boundaries
        if min_date is not None and isinstance(min_date,str):
            min_date=pd.to_datetime(min_date)
        if max_date is not None and isinstance(max_date,str):
            max_date=pd.to_datetime(max_date)
        ## N Gram Range based on Vocabulary
        ngrams = [min(self.vocab._ngrams[0], self.vocab._external_ngrams[0]),
                  max(self.vocab._ngrams[1], self.vocab._external_ngrams[1])]
        ## Temporarily Extract Any Resampling Models
        resample_models = None
        resample_preprocessors = None
        if hasattr(self.model, "_resampled"):
            resample_models = deepcopy(self.model._resampled)
            self.model._resampled = None
        if hasattr(self.preprocessor, "_resampled"):
            resample_preprocessors = deepcopy(self.preprocessor._resampled)
            self.preprocessor._resampled = None
        ## Vectorize the data
        LOGGER.info("Vectorizing Files")
        test_files, X_test, _ = self._load_vectors(test_files,
                                                   None,
                                                   min_date=min_date,
                                                   max_date=max_date,
                                                   n_samples=n_samples,
                                                   randomized=randomized,
                                                   resolution=resolution,
                                                   reference_point=reference_point,
                                                   chunk_n_samples=chunk_n_samples,
                                                   chunk_randomized=chunk_randomized,
                                                   ngrams=ngrams)
        ## Assign Resampling Instances Back (if Necessary)
        if resample_models is not None:
            self.model._resampled = resample_models
        if resample_preprocessors is not None:
            self.preprocessor._resampled = resample_preprocessors
        ## Alert User To Null Rows
        if isinstance(X_test, csr_matrix):
            null_rows = X_test.getnnz(axis=1) == 0
        else:
            null_rows = (X_test==0).all(axis=1)
        LOGGER.info("Found {}/{} Null Rows".format(
                    null_rows.sum(),
                    len(null_rows),
        ))
        ## Drop Null Rows if Desired
        if drop_null:
            LOGGER.info("Dropping null data points")
            mask = np.nonzero(null_rows==0)[0]
            X_test = X_test[mask]
            test_files = [test_files[i] for i in mask]
        ## Format Files (in case of aggregation)
        test_files = self._format_filenames(test_files)
        ## Primary Predictions
        LOGGER.info("Generating Feature Set and Making Predictions")
        y_pred = self._predict(X_test, model=self.model, preprocessor=self.preprocessor)
        y_pred = dict((x, y) for x,y in zip(test_files, y_pred))
        ## Resampled Predictions
        if hasattr(self.model, "_resampled") and \
           hasattr(self.preprocessor, "_resampled") and \
           self.model._resampled is not None and \
           self.preprocessor._resampled is not None:
            LOGGER.info("Generating Feature Set and Making Predictions using Resampled Models")
            y_pred_resample = self._predict_multisample(preprocessors=self.preprocessor._resampled,
                                                        models=self.model._resampled,
                                                        X=X_test,
                                                        verbose=True)
            y_pred_resample = dict((x, list(y)) for x, y in zip(test_files, y_pred_resample))
            return y_pred, y_pred_resample
        else:
            return y_pred

    def copy(self):
        """
        Make a copy of the MentalHealthClassifier class.

        Args:
            None
        
        Returns:
            deepcopy of the MentalHealthClassifier
        """
        return deepcopy(self)
    
    def dump(self,
             filename,
             compress=5):
        """
        Save the model instance to disk using joblib.

        Args:
            filename (str): Name of the model for saving.
            compress (int): Level of compression to pass to
                            joblib dump function.
        
        Returns:
            None, saves model to disk.
        """
        if not filename.endswith(".joblib"):
            filename = filename + ".joblib"
        _ = joblib.dump(self,
                        filename,
                        compress=compress)
        return