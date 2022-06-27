# Modeling and Data Analysis

This section of the repository contains code that is used for the primary meat of our paper (e.g., modeling and data analysis). In addition to visualization code that summarizes results across analyses, we have three main types of analysis denoted with the following prefixes:

* `track`: Measure prevalence of certain keywords/phrases over the course of time.
* `word2vec`: Train word2vec embedding models and evaluate use of semantic stability as a feature selection method.
* `estimator`: Train depression classifiers using a variety of vocabularies (e.g., via semantically stable feature selection) and evaluate their estimate of change in depression prevalence.

## Keyword Tracking

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

## Semantic Stability for Feature Selection

This series of experiments is designed to understand whether use of semantic stability scores can improve predictive generalization over time. Our implementation makes heavy use of parallelization over our institution's grid, though it is not explicitly necessary. One should be able to run the commands manually in place of our scheduler.

### Procedure

1. **Train word2vec embedding models on multiple subsets of an annotated dataset.** Use `scripts/model/word2vec_schedule.py` to create configuration files (and optionally schedule training) for the word2vec models. For those not using an SGE schedule, you can set `USE_SCHEDULER=False` in the header of the script. This will generate necessary runtime files which can then be executed independently. See the header of the schedule file to see what parameters can be changed. Outside of dataset-specific parameters (e.g., name, time periods), all other parameters are those which were used in the final paper.

For those who do not have access to an SGE scheduler, look for the .sh files generated in the output directory of the scheduler (e.g., `data/results/word2vec-context/clpsych/`). The python command at the end of the script (after environment initialization) can be used to train the models in serial. For example:

```
python scripts/model/word2vec_train.py \
    ./data/results/word2vec-context/clpsych/2012-01-01_2013-01-01.json \
    --output_dir ./data/results/word2vec-context/clpsych/ \
    --stream_cache_dir ./data/results/word2vec-context/clpsych/cache/
```

2. **Vectorize each of the data subsets.** Each of the word2vec models trained in the step above is associated with a "Phraser" vocabulary model. We will use this model to tokenize and re-phrase the training/evaluation subsets, ultimately transforming everything into a bag-of-words representation.

Begin by updating the file `configs/word2vec/postprocess_template.json` to represent your modeling interests (e.g., specify path to the models trained). Our goal is vectorization. This can be done in parallel, serial, or by making use of previously vectorized data for the given dataset.

#### Option 1: Serial (No SGE Access)

```
python scripts/model/word2vec_postprocess.py \
    configs/word2vec/postprocess_template.json \
    --vectorize serial
    --jobs 2
```

or you can specify individual split IDs (replacing `$ID`)

```
python scripts/model/word2vec_postprocess.py \
    configs/word2vec/postprocess_template.json \
    --vectorize parallel \
    --vectorize_id $ID
    --jobs 2
```

#### Option 2: Parallel (SGE Access)

```
python scripts/model/word2vec_postprocess.py \
    configs/word2vec/postprocess_template.json \
    --vectorize parallel \
    --grid_memory_per_job 16 \
    --grid_max_array_size 100 \
    --jobs 2
```

#### Option 3: Symbolic Vectorization

If you have already vectorized one of these datasets and want to try a new set of modeling parameters (i.e. one of the later stages), there's no need to revectorize the dataset. You can instead tweak the previous configuration file and then specify you'd like to generate a symbolic vector set based on an existing cache.

```
## Replace <analysis_dir_name> with the existing vector path
python scripts/model/word2vec_postprocess.py \
    configs/word2vec/postprocess_template_new.json \
    --vectorize symbolic \
    --vectorize_symbolic_path "./data/results/word2vec/clpsych/analysis/<analysis_dir_name>/data-cache/"
```

3. **Train Classification Models.** With the data vectorized, you can now train classification models using a variety of feature selection strategies. All modeling parameters can be specified in the configuration JSON file. Note that "classifier_k_models" indicates how many models will be trained for each of the prior data splits (i.e., by performing additional data splitting that respects train/test splits but uses a single word2vec embedding based on the training data).

#### Option 1: Serial (No SGE)

```
python scripts/model/word2vec_postprocess.py \
    configs/word2vec/postprocess_template.json \
    --classify serial \
    --classify_enforce_frequency_filter \
    --jobs 8
```

or by directly specifying the ID.

```
python scripts/model/word2vec_postprocess.py \
    configs/word2vec/postprocess_template.json \
    --classify parallel \
    --classify_id $ID \
    --classify_enforce_frequency_filter \
    --jobs 8
```

### Option 2: Parallel (SGE)

If you have already tested the code by running the serial implementation (or started a series of jobs and experienced some unexpected failure), you can include the `--classify_skip_existing` flag below. 

```
python scripts/model/word2vec_postprocess.py \
    configs/word2vec/postprocess_template.json \
    --classify parallel \
    --classify_enforce_frequency_filter \
    --classify_skip_existing \
    --grid_memory_per_job 16 \
    --grid_max_array_size 100 \
    --jobs 8
```

4. **Merge and Analyze the Results.** Once all samples have completed, we can merge the results together to generate a single analysis of the effectiveness of semantic stability as a feature selection criteria. This part does not use any SGE scheduling.

```
python scripts/model/word2vec_postprocess.py \
    configs/word2vec/postprocess_template.json \
    --analyze \
    --jobs 8
```

5. **Interpret the Results.** Dive into the output directory and examine the difference in classification performance as a function of feature selection methodology. Note that fine-grained results for each split can also be found within the `result-cache/*` subdirectories.

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