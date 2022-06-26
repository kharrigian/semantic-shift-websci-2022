

## Track

#### Procedure

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