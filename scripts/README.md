# Scripts

## Phase 1: Data Acquisition

The majority of data in our study comes from external sources (e.g., institutional cache of the Twitter 1% stream, SMHD dataset). We assume these are stored in the project repository at `data/raw/<platform>/`, where `<platform>` indicates whether the data comes from `twitter` or `reddit`.

Two samples of data are queried from scratch -- the "Topic-restricted Text" Reddit dataset and the "Active" sample of Reddit users (for downstream model evaluation). Code for acquiring these datasets can be found in `scripts/acquire/`.

If you would like to apply our analysis to your own data, please move on to the next phase while taking note of where your data is stored. For simplicity, we recommend symlinking the data to this project repository to make it easier to understand relative paths.

## Phase 2: Data Preprocessing

Assuming you have acquired all relevant raw data, we will now move on to preprocessing. The goal of this phase is to standardize the formatting of all datasets to make modeling easier. There are two categories of preprocessing.

#### Annotated Datasets

Each of the annotated datasets should be processed using their respective scripts in either `scripts/preprocess/twitter/` or `scripts/preprocess/reddit/`. These not only format the raw text data, but also keep track of the annotated labels (i.e. depression status).

Once these datasets are processed, please run the `scripts/preprocess/compile_metadata_lookup.py` script to generate CSV files that contain relevant metadata and label mapping for each of the annotated datasets.

#### Unlabeled Datasets

To preprocess the unlabeled datasets, please use `scripts/preprocess/preprocess.py` with the appropriate command-line arguments as provided by the `--help` command. The unlabeled Twitter dataset in particular is quite large; for this reason, we recommend processing the data in parallel over many compute jobs. That said, we have provided examples of processing both the Twiter Gardenhose and Reddit Active datasets (see `.sh` files).

Once you have the unlabeled datasets processed, you can apply additional filtering if necessary to clean things up (e.g., isolate a subset of users meeting a certain criteria, initialize date-based boundaries). In our study, we used the `scripts/preprocess/preprocess_filter.py` to isolate Twitter users with at least 50 posts in the preprocessed dataset.

## Phase 3: Modeling and Experimentation

Once data has been curated, we can move on to the fun stuff - modeling and data analysis. Everything relevant can be found in `scripts/model/`. There are 4 main sections of analysis, each of which is described in detail within the README within the `scripts/model/` directory.

* `track`: Measure prevalence of certain keywords/phrases over the course of time.
* `word2vec`: Train word2vec embedding models and evaluate use of semantic stability as a feature selection method.
* `estimator`: Train depression classifiers using a variety of vocabularies (e.g., via semantically stable feature selection) and evaluate their estimate of change in depression prevalence.
* `visualizations`: Supplementary visualizations that merge data across multiple experimental runs.