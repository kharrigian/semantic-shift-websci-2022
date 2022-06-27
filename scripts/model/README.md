# Modeling and Data Analysis

This section of the repository contains code that is used for the primary meat of our paper (e.g., modeling and data analysis). In addition to visualization code that summarizes results across analyses, we have three main types of analysis denoted with the following prefixes:

* `track`: Measure prevalence of certain keywords/phrases over the course of time.
* `word2vec`: Train word2vec embedding models and evaluate use of semantic stability as a feature selection method.
* `estimator`: Train depression classifiers using a variety of vocabularies (e.g., via semantically stable feature selection) and evaluate their estimate of change in depression prevalence.

## Track

Measure usage of keywords/keyphrases over time within a dataset. There are two options of running this analysis. By default, results are stored in `data/results/track/`.

#### Option 1: Map/Reduce

If you have a smaller number of large files, this approach is recommended. For example, we have on the order of 2000 data files for the "gardenhose" Twitter dataset, each of which is quite large. For this reason, we map the counting procedure across the files during the first stage of analysis and then aggregate the results during the second stage.

```
## Parallelize Across Compute Cluster
python scripts/model/track_schedule.py

## Aggregate Results
python scripts/model/track_postprocess.py
```

#### Option 2: Multiprocessing

If you have a large number of small files, this approach is recommended. For example, we have 50,000 data files in the "active" Reddit dataset, each of which is small. Here, it doesn't make sense to schedule a single process for each file. Instead, we use multiprocessing within a single job to process all the files.

```
## Apply Tracking
python scripts/model/track_apply.py configs/track/active.json --make_plots
```

## Word2Vec

#### Procedure

## Estimator

#### Procedure

1. Filter the unlabeled data samples (filtering)

```
## Count User Activity
python scripts/experiment/longitudinal/explore/estimator_preprocess.py \
    ./configurations/experiments/longitudinal/explore/estimator/apply_template.json \
    --counter parallel \
    --jobs 1 \
    --grid_max_array_size 1000 \
    --grid_memory_per_job 4

## Analyze Counts
python scripts/experiment/longitudinal/explore/estimator_preprocess.py \
    ./configurations/experiments/longitudinal/explore/estimator/apply_template.json \
    --analyze_counts

## Chunk Users (Using a Desired threshold)
python scripts/experiment/longitudinal/explore/estimator_preprocess.py \
    ./configurations/experiments/longitudinal/explore/estimator/apply_template.json \
    --chunk_users

## Filter Data, Grouping by Chunk
python scripts/experiment/longitudinal/explore/estimator_preprocess.py \
    ./configurations/experiments/longitudinal/explore/estimator/apply_template.json \
    --filter_users parallel \
    --jobs 8 \
    --grid_max_array_size 100 \
    --grid_memory_per_job 8
```

2. Fit word2vec models on all desired datasets (labeled + unlabeled).

*Note*: Should turn on language filter for labeled, but not unlabeled (pre-filtered) datasets.

```
## Fit Primary (Using All Available Data)
python scripts/model/embed.py \
    configurations/experiments/longitudinal/explore/word2vec/train_labeled.json \
    --output_dir ./data/results/longitudinal/results/explore/estimator/word2vec/ \
    --stream_cache_dir ./data/results/longitudinal/results/explore/estimator/word2vec/

## Fit Subsets (Resampled Subsets)
python scripts/model/embed.py \
    configurations/experiments/longitudinal/explore/word2vec/train_labeled.json \
    --output_dir ./data/results/longitudinal/results/explore/estimator/word2vec/ \
    --stream_cache_dir ./data/results/longitudinal/results/explore/estimator/word2vec/ \
    --phraser_dir <PATH_TO_PRIMARY_MODEL_DIR> \
    --resample_parallel \
    --grid_memory_request_size <SIZE> \
    --grid_log_dir <LOG_DIR>
```

3. Extract vocabulary from learned embeddings (primary version)

```
python scripts/experiments/longitudinal/explore/util/word2vec_extract_vocabulary.py <PATH_TO_EMBEDDINGS_FILE>
```

4. Run training procedure with semantic feature selection to generate models. Run for each desired subset learned in Step 1.

```
python scripts/experiment/longitudinal/explore/estimator_train.py \
    ./configurations/experiments/longitudinal/explore/estimator/train_template.json \
    --output_dir ./data/results/longitudinal/results/explore/estimator/models/ \
    --classify_enforce_min_shift_frequency_filter \
    --jobs 8
```

5. Vectorize the Unlabeled Data Samples

```
## Vectorize the Data (Using Phrasers Learned From Primary Embeddings)
python scripts/experiment/longitudinal/explore/estimator_preprocess.py \
    ./configurations/experiments/longitudinal/explore/estimator/preprocess_template.json \
    --vectorize_users parallel \
    --jobs 1 \
    --grid_max_array_size 500 \
    --grid_memory_per_job 8
```    

5. Apply trained models (each using different vocabularies) to vectorized, unlabeled data.

```
python scripts/experiment/longitudinal/explore/estimator_apply.py \
    ./configurations/experiments/longitudinal/explore/estimator/apply_template.json \
    --output_dir ./data/results/longitudinal/results/explore/estimator/predictions/ \
    --jobs 8
```

## Preprocessing Parameters

Both datasets have the following preprocessing parameters:

* Isolate English (using `langid`)
* Date Boundaries: [2019-03-01 to 2019-07-01, 2020-03-01 to 2020-07-01]
* Posts with mental health keywords or from mental health subreddits are *included*.
* For filtering, we maintain all data from 2019-01-01 to 2020-07-01
* For vectorization, we concatenate all available posts amongst users who meet the specified minimums (see below)

The Gardenhose (Twitter) dataset has the following preprocessing parameters:

* Retweets are *included*
* Minimum Number of Posts Per Boundary (filtering criteria): 50. Drops from 40M unique users to 486k.
* Each user chunk has 10,000 users. Within each user chunk, we generate files containing roughly 10,000 posts each.
* Minimum Number of Posts Per Boundary (vectorization criteria): 200. 25,379 unique users (5,024 in both)

The Active (Reddit) dataset has the following preprocessing parameters:

* Minimum Number of Posts Per Boundary (filtering criteria): 50. Drops from 51k unique users to 47k users.
* Each user chunk has 5,000 users. Within each chunk, we generate files containing roughly 5,000 posts each.
* Minimum Number of Posts Per Boundary (vectorization criteria): 100. 40,671 unique users (28,835 in both).

## Visualization