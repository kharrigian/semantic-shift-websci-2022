
"""
Utility Functions for Longitudinal Analysis
"""

####################
### Imports
####################

## External Libraries
import demoji
import numpy as np
from scipy import sparse
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer

## Local
from semshift.util.logging import initialize_logger

####################
### Globals
####################

## Initialize Logger
LOGGER = initialize_logger()

####################
### Classes
####################

## Dict Vectorizer
class DVec(object):

    def __init__(self,
                 items=None):
        """

        """
        if items is not None:
            _ = self._initialize_dict_vectorizer(items)

    def _initialize_dict_vectorizer(self, items):
        """
        Initialize a vectorizer that transforms a counter dictionary
        into a sparse vector of counts (with a uniform feature index)

        Args:
            None
        
        Returns:
            None, initializes self._count2vec inplace.
        """
        self._count2vec = DictVectorizer(separator=":", dtype=int)
        self._count2vec.vocabulary_ = dict(zip(items, range(len(items))))
        rev_dict = dict((y, x) for x, y in self._count2vec.vocabulary_.items())
        self._count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
    
    def transform(self,
                  counts):
        """

        """
        return self._count2vec.transform(counts)

####################
### Functions
####################

def replace_emojis(features):
    """
    
    """
    features_clean = []
    for f in features:
        f_res = demoji.findall(f)
        if len(f_res) > 0:
            for x,y in f_res.items():
                f = f.replace(x,f"<{y}>")
            features_clean.append(f)
        else:
            features_clean.append(f)
    return features_clean

def align_vocab(X,
                xs_vocab,
                xt_vocab):
    """
    Transform matrix from one vocabulary to another

    Args:
        X (csr matrix): Matrix to Transform
        xs_vocab (iterable): Current Vocabulary (column-wise)
        xt_vocab (iterable): Target Vocabulary (column-wise)
    
    Returns:
        X_T (csr matrix): Original Data with New Column-wise Alignment
    """
    ## Existing Vocabulary Mapping
    vocab2ind = dict(zip(xs_vocab, range(len(xs_vocab))))
    ## Append Empty Column
    X_T = sparse.hstack([X, sparse.csr_matrix((X.shape[0],1))]).tocsr()
    ## New Alignment Sorting According to Target Vocabulary Sorting
    i_null = X.shape[1]
    v_new = list(map(lambda v: vocab2ind.get(v, i_null), xt_vocab))
    ## Apply Alignment
    X_T = X_T[:,v_new]
    return X_T

def score_predictions(y_true,
                      y_score,
                      threshold=0.5):
    """
    Compute standard classification performance metrics
    for a set of predictions and ground truth

    Args:
        y_true (array): Ground-truth labels (binary)
        y_score (array): Predicted positive-class probability
        threshold (float): Binarization threshold for probabilities
    
    Returns:
        scores (dict): Precision, Recall, F1, Accuracy, AUC, Average Precision
    """
    ## Binarize
    y_pred = (y_score >= threshold).astype(int)
    ## Check for Variation
    varies = True
    gt_varies = True
    if set(y_true) == set([0,1]) and set(y_pred) == set([0]):
        varies = False
    if len(set(y_true)) == 1:
        gt_varies = False
    ## Scores
    scores = {
        "precision":metrics.precision_score(y_true, y_pred) if varies else 0,
        "recall":metrics.recall_score(y_true, y_pred) if varies else 0,
        "f1":metrics.f1_score(y_true, y_pred) if varies else 0,
        "accuracy":metrics.accuracy_score(y_true, y_pred) if varies else 0,
        "average_precision":metrics.average_precision_score(y_true, y_score) if varies else 0,
        "auc":metrics.roc_auc_score(y_true, y_score) if gt_varies else np.nan,
    }
    return scores
    
def get_feature_names(feature_transformer):
    """
    Extract feature names, aware of variance filtering

    Args:
        feature_transformer (FeaturePreprocessor): Fit processor
    
    Returns:
        features (list): Named list of features in the processed feature matrix.
    """
    features = []
    if not hasattr(feature_transformer, "_transformer_names"):
        feature_transformer._transformer_names = []
    for t in feature_transformer._transformer_names:
        transformer = feature_transformer._transformers[t]
        if t in ["bag_of_words","tfidf"]:
            tf = feature_transformer.vocab.get_ordered_vocabulary()
        elif t == "glove":
            tf =  list(map(lambda i: f"GloVe_Dim_{i+1}", range(transformer.dim)))
        elif t == "liwc":
            tf = [f"LIWC={n}" for n in transformer.names]
        elif t == "lda":
            tf = [f"LDA_TOPIC_{i+1}" for i in range(transformer.n_components)] 
        features.extend(tf)
    if hasattr(feature_transformer, "_min_variance") and feature_transformer._min_variance is not None:
        feature_mask = feature_transformer._transformers["variance_threshold"].get_support()
        features = [f for f, m in zip(features, feature_mask) if m]
    return features

## Time Bin Assigner
def batch_time_bin_assign(time_bounds,
                          time_bins):
    """
    Args:
        time_bounds (list of tuple): Lower, Upper Epoch Times
        time_bins (list of tuple): Lower, Upper Time Bin Boundaries
    """
    ## Assign Original Indice
    time_bounds_indexed = [(i, x, y) for i, (x, y) in enumerate(time_bounds)]
    ## Sort Indexed Time Bins By Lower Bound
    time_bounds_indexed = sorted(time_bounds_indexed, key=lambda x: x[1])
    ## Initialize Counters and Cache
    m = 0
    n = 0
    M = len(time_bins)
    N = len(time_bounds)
    assignments = []
    ## First Step: Assign Nulls to Bounds Before Time Bin Range
    while n < N:
        if time_bounds_indexed[n][2] < time_bins[m][0]:
            assignments.append(None)
            n += 1
        else:
            break
    ## Second Step: Assign Bins in Batches
    while n < N:
        ## Get Time Range for Data Point
        lower, upper = time_bounds_indexed[n][1], time_bounds_indexed[n][2]
        ## Check to See If Data Point Falls Outside Max Range
        if len(time_bins) == 1 and lower > time_bins[0][1]:
            assignments.append(None)
            n += 1
            continue
        elif len(time_bins) > 1 and lower > time_bins[-1][1]:
            assignments.append(None)
            n += 1
            continue
        ## Increment Time Bins Until Reaching Lower Bound
        while m < M and time_bins[m][1] <= lower:
            m += 1
        ## Cache Assignment
        assignments.append(m)
        n += 1
    ## Add Assignments With Index
    assignments_indexed = [(x[0], y) for x, y in zip(time_bounds_indexed, assignments)]
    ## Sort Assignments by Original Index
    assignments_indexed = sorted(assignments_indexed, key=lambda x: x[0])
    ## Isolate Assignments
    assignments_indexed = [i[1] for i in assignments_indexed]
    return assignments_indexed