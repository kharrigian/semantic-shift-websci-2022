# Modeling and Data Analysis

This section of the repository contains code that is used for the primary meat of our paper (e.g., modeling and data analysis). In addition to visualization code that summarizes results across analyses, we have three main types of analysis denoted with the following prefixes:

* `track`: Measure prevalence of certain keywords/phrases over the course of time.
* `word2vec`: Train word2vec embedding models and evaluate use of semantic stability as a feature selection method.
* `estimator`: Train depression classifiers using a variety of vocabularies (e.g., via semantically stable feature selection) and evaluate their estimate of change in depression prevalence.

## Keyword Tracking

Measure usage of keywords/keyphrases over time within a dataset. There are two options of running this analysis. By default, results are stored in `data/results/track/`. An example of the appropriate configuration file structure is `configs/track/gardenhose.json` and is explained immediately below.

### Configuration

```
{
    ## Name of the output directory created
    "experiment_name":"gardenhose",

    ## List of datasets to include in counting procedure
    "datasets":[
        {
            ## Path to preprocessed files
            "dataset":"/export/c01/kharrigian/semantic-shift-websci-2022/data/processed/twitter/gardenhose/*.gz",
            ## If unlabeled, "custom-<platform>"
            "target_disorder":"custom-twitter",
            ## Time period to constrain processing
            "date_boundaries":{
                "min_date":null,
                "max_date":null
            },
            ## Downsample on a file-level
            "downsample":false,
            "downsample_size":null,
            ## Downsample on a post-level
            "post_sampling":{
                "n_samples":null
            },
            ## Apply language filtering
            "lang":["en"]
        }
    ],
    ## Whether to count username mentions
    "model_kwargs":{
        "include_mentions":false
    },
    ## Agregation frequency (hour, day, or month)
    "agg_kwargs":{
        "frequency":"day"
    },
    ## Proportion of matched posts to save to disk
    "cache_rate":null,
    ## List of keywords/phrases.
    "terms":[
        {
            "include_hashtags":true,
            "terms":"./data/resources/keywords/pmi.keywords"
        },
        {
            "include_hashtags":true,
            "terms":"./data/resources/keywords/crisis_level1.keywords"
        },
        {
            "include_hashtags":true,
            "terms":"./data/resources/keywords/crisis_level2.keywords"
        },
        {
            "include_hashtags":true,
            "terms":"./data/resources/keywords/crisis_level3.keywords"
        }
    ],  
    ## Processing Parameters
    "jobs": 1,
    "chunksize":1,
    "random_seed":42
}
```

### Procedure

*Parallel (SGE Scheduler)*

If you have a smaller number of large files, this approach is recommended. For example, we have on the order of 2000 data files for the "gardenhose" Twitter dataset, each of which is quite large. For this reason, we map the counting procedure across the files during the first stage of analysis and then aggregate the results during the second stage. Note that this option assumes access to an SGE scheduler.

```
## Parallelize Across Compute Cluster
python scripts/model/track_schedule.py

## Aggregate Results
python scripts/model/track_postprocess.py
```

*Parallel (No SGE Scheduler)*

If you have a large number of small files, this approach is recommended. For example, we have 50,000 data files in the "active" Reddit dataset, each of which is small. Here, it doesn't make sense to schedule a single process for each file. Instead, we use multiprocessing within a single job to process all the files.

```
## Apply Tracking
python scripts/model/track_apply.py configs/track/active.json --make_plots
```

## Semantic Stability for Feature Selection

This series of experiments is designed to understand whether use of semantic stability scores can improve predictive generalization over time. Our implementation makes heavy use of parallelization over our institution's grid, though it is not explicitly necessary. One should be able to run the commands manually in place of our scheduler.

The first phase focuses on training the embedding models, with the relevant configuration template being `configs/word2vec/embed_template.json`. The second phase uses the learned embeddings in conjunction with inference experiments, with the relevant configuration template being `configs/word2vec/postprocess_template.json`.

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

Although it is important we show semantic stability has the ability to improve predictive performance over time, perhaps more interesting is the effect semantic stability has on classifiers applied to measure downstream outcomes (e.g., change in prevalence of depression). In this section of the repository, we provide a framework for training semantically stable models and evaluating downstream outcomes.

### Procedure

This is probably the most convoluted process (involving the most steps). Where possible, we we delineate between parallel (SGE scheduler) and serial (no SGE) implementations. It is broken down into phases and steps.

#### Phase 1: Filter Down Unlabeled Data

Configurations with parameterizations used in our publication can be found in `configs/estimator/parameterized/preprocess/`, while the template can be found at `configs/estimator/preprocess_template.json`.

1. **Count number of posts made by each user in an unlabeled dataset.** The goal will be to remove users who don't have enough post volume to do useful modeling. The first part of this is counting how many posts each user made during different time periods in the unlabeled dataset.

Fields to update in the configuration file at this stage:
* `"output"`: Where you would like the results to be stored.
* `"inputs"`: Glob-supported data paths that indicate where the preprocessed data files live.
* `"inputs_metadata"/"platform"`: "twitter" or "reddit"
* `"inputs_metadata"/"user_per_file"`: whether the preprocessed data files already have a single user's data per file.
* `"cache_user_critiera"/"date_boundaries"`: include the two time-periods you plan on using to measure year-over-year change.
* `"cache_user_critiera"/"lang"`: include any languages you wish to isolate (e.g., `["en"]` if the datasets are not already filtered.)
* `"cache_user_critiera"/"include_retweets"`: whether to include retweets in a Twitter dataset. You can always remove these later if you keep them now.

*Serial Implementation*

```
## Count User Activity
python scripts/model/estimator_preprocess.py \
    ./configs/estimator/preprocess_template.json \
    --counter serial \
    --jobs 8
```

*Parallel Implementation*

```
## Count User Activity
python scripts/model/estimator_preprocess.py \
    ./configs/estimator/preprocess_template.json \
    --counter parallel \
    --jobs 1 \
    --grid_max_array_size 1000 \
    --grid_memory_per_job 4 \
    --grid_max_tasks_concurrent 10
```

2. **Analyze/aggregate the counts.** Use the outputs from this process to set a minimum post threshold for maintaining users in the dataset. It's better to have a lower threshold at this point, since we can always filter results post-hoc based on post volume. However, you should also remove users who do not have enough data to do any kind of modeling. There is *no* need to update the configuration at this stage.

```
## Analyze Counts
python scripts/model/estimator_preprocess.py \
    ./configs/estimator/preprocess_template.json \
    --analyze_counts \
    --jobs 8
```

3. **Group users into chunks.** This step and the next step are useful for modeling since it puts all posts from a single user into the same file (reducing I/O). This piece simply applies the threshold filtering critiera and assigns users into multiple chunks (without changing any of the data).

Fields to update include:

* `"cache_user_criteria"/"min_posts_per_date_boundary"`: Post threshold you choose to set based on the analysis outputs.
* `"cache_user_criteria"/"intersection"`: Set `true` if you would only like to keep users who meet the threshold in both time periods.
* `"cache_kwargs"/"chunksize"`: How many authors should be grouped together.
* `"cache_kwargs"/"within_chunk_chunksize"`: Maximum number of posts to include within each file.
* `"cache_kwargs"/"date_boundaries"`: Range of data to keep in the chunks.
* `"cache_kwargs"/"sample_rate"`: Option to downsample on a post-level.

```
## Chunk Users (Using a Desired threshold included in the configuration file)
python scripts/model/estimator_preprocess.py \
    ./configs/estimator/preprocess_template.json \
    --chunk_users \
    --jobs 8
```

4. **Re-aggregate the data based on the chunks.** With the users assigned into chunks, we can now re-aggregate the data so that each file contains all data for a given user. There is *no* need to change any parameters in the configuration for this step (assuming you updated the parameters in step 3).

*Serial Implementation*

```
python scripts/model/estimator_preprocess.py \
    ./configs/estimator/preprocess_template.json \
    --filter_users serial \
    --jobs 8
```

*Parallel Implementation*

```
## Filter Data, Grouping by Chunk
python scripts/model/estimator_preprocess.py \
    ./configs/estimator/preprocess_template.json \
    --filter_users parallel \
    --jobs 8 \
    --grid_max_array_size 100 \
    --grid_memory_per_job 8 \
    --grid_max_tasks_concurrent 10
```

#### Phase 2: Fit Word2Vec Models (Labeled + Unlabeled Datasets)

*Note*:  We turn on language filter for labeled datasets, but not unlabeled datasets. This is because (if all steps above completed correctly), the unlabeled datasets have already been filtered to only include English data. The labeled datasets, on the other hand, have had the filtering done on the fly prior to modeling in the previous experiments.

Parameterized versions of configurations used in the paper are included in `configs/estimator/parameterized/embed/`. General configuration templates are found at `configs/estimator/embed_labeled_template.json` and `configs/estimator/embed_unlabeled_template.json` for labeled and unlabeled datasets, respectively.

1. **Fit a "primary" word2vec embedding model using all available data for a given dataset.** This will be used primarily for learning a standardized vocabulary and doing post-hoc analysis.

```
## Fit Primary Model (Using All Available Data - Labeled Dataset Example)
python scripts/model/word2vec_train.py \
    configs/estimator/parameterized/embed/primary/clpsych.json \
    --output_dir ./data/results/estimator/embeddings/ \
    --stream_cache_dir ./data/results/estimator/embeddings/

## Fit Primary Model (Using All Available Data - Unlabeled Dataset Example)
python scripts/model/word2vec_train.py \
    configs/estimator/parameterized/embed/primary/gardenhose.json  \
    --output_dir ./data/results/estimator/embeddings/ \
    --stream_cache_dir ./data/results/estimator/embeddings/
```

2. **Fit "secondary" models using subsets of the full dataset.** These are the models which will actually be used for selecting semantically stable features and training downstream classifiers. Note that in these configuration files, resampling is turned on.

*Serial Implementation*

```
## Fit Subsets (Resampled Subsets)
python scripts/model/word2vec_train.py \
    configs/estimator/parameterized/embed/subsets/clpsych.json \
    --output_dir ./data/results/estimator/embeddings/ \
    --stream_cache_dir ./data/results/estimator/embeddings/ \
    --phraser_dir ./data/results/estimator/embeddings/clpsych/primary/
```

*Parallel Implementation*

```
## Fit Subsets (Resampled Subsets)
python scripts/model/word2vec_train.py \
    configs/estimator/parameterized/embed/subsets/clpsych.json \
    --output_dir ./data/results/estimator/embeddings/ \
    --stream_cache_dir ./data/results/estimator/embeddings/ \
    --phraser_dir ./data/results/estimator/embeddings/clpsych/primary/
    --resample_parallel \
    --grid_memory_request_size 16 \
    --grid_log_dir ./data/results/estimator/word2vec/clpsych/logs/
```

3. **Extract vocabulary from learned embeddings.** Recall that the same phraser model (vocabulary) is used for all subsets and comes from the primary model.

```
python scripts/model/util/word2vec_extract_vocabulary.py \
    ./data/results/estimator/embeddings/clpsych/primary/embeddings.100d.txt
```

#### Phase 3: Train Machine Learning Classifiers

We will now train the classifiers using knowledge of semantic shift between embeddings. Accordingly, it is required to have access to at least two datasets worth of embeddings to move forward. A template of the primary configuration template for this phase is `configs/estimator/train_template.json`. Parameterized versions from the paper are found in `configs/estimator/parameterized/train/`. 

You will note that we only include general files for Reddit and Twitter, since the parameterizations are essentially the same (absent directories) for CLPsych and Multi-Task, and SMHD and Topic-Restricted Text, respectively. The "parallelized" version of this phase will make use of these general files to schedule multiple jobs for all the datasets (see `scripts/model/estimator_train_schedule.py`). As always, we will detail both procedures below.

*Config File*
```
{
    "experiment_id":"clpsych-gardenhose", ## Identifier for the training procedure
    
    "source":{
        ## Configuration file used for learning the word embeddings
        "config":"./data/results/estimator/embeddings/clpsych/subsets/config.json",
        ## Path to the split for the subset of data to use for training the classifier
        "splits":"./data/results/estimator/embeddings/clpsych/subsets/splits/clpsych/0.json",
        ## Path to the directory containing the embeddings trained on the training subset of the split
        "embeddings":"./data/results/estimator/embeddings/clpsych/subsets/models/sample-0/",
        ## Minimum frequency of vocabulary to maintain in vocabulary
        "min_vocab_freq":50,
        ## Minimum frequency of vocabulary that is a candidate for being a feature in the classifier
        "min_shift_freq":50,
        ## Neighborhood size to use for semantic stability computation
        "top_k_neighbors":500,
        ## Number of most frequent terms to remove
        "rm_top":250,
        ## Whether to remove stopwords
        "rm_stopwords":true
    },
    "target":{
        ## Same as above, but using the target dataset (e.g. unlabeled Twitter Gardenhose)
    },
    "evaluation":{
        ## Configuration of embedding split to use for evaluating performance (should be a labeled dataset)
        "config":"./data/results/estimator/embeddings/clpsych/subsets/config.json",
        ## JSON file containing the splits for the evaluation dataset
        "splits":"./test-embeddings/clpsych/subsets/splits/clpsych/0.json"
    },

    ## Number of models to train using resampled versions of the training data
    "classifier_k_models":10, 
    ## Percentage of training data to sample for each resampling iteration
    "classifier_resample_rate":0.8,
    ## Minimum vocabulary frequency of features passed to classifier (can set to 1 to rely only on above)
    "classifier_min_vocab_freq":1,
    ## Minimum user frequency of features passed to classifier (can set to 1 to rely only on above)
    "classifier_min_vocab_sample_freq":1,
    ## Number of posts required from a user to consider them part of the training sample
    "classifier_min_n_posts":200,
    ## Number of posts to use from each user for training. Set null to use all available posts.
    "classifier_n_posts":null,

    ## Feature size percentiles
    "selectors_percentiles":[90, 80, 70, 60, 50, 40, 30, 20, 10],
    ## Proportion of times a feature was included in baseline resampling to consider for final model
    "selectors_sample_support":0.9,
    ## Which feature selectors to train models with
    "selectors":[
        "cumulative",
        "intersection",
        "random",
        "frequency",
        "chi2",
        "coefficient",
        "overlap",
        "weighted_overlap-coefficient_0.5"
    ],

    ## For logistic-regression-cv estimator, optimization metric
    "cv_metric_opt":"f1",
    ## Whether to standardize feature representation
    "feature_standardize":false,
    ## Only "tfidf" is supported currently
    "feature_flags":{
        "tfidf":true
    },
    ## Parameters passed to the TfIDF transformer
    "feature_kwargs":{
        "tfidf":{"norm":"l2"}
    },

    ## Random Sample State
    "random_state":42
}
```

*Procedure*

1. **Run training procedure with semantic feature selection to generate models.** Run for each desired subset learned in Step 1. We provide the core serial implementation call below. Note however that you can also use `scripts/model/estimator_train_scheduler.py` with `USE_SCHEDULER=False` to generate the basic command structures necessary for all relevant dataset combinations.

*Serial Implementation (No SGE)*

```
python scripts/model/estimator_train.py \
    ./configs/estimator/train_template.json \
    --output_dir ./data/results/estimator/train-adaptation/ \
    --classify_enforce_min_shift_frequency_filter \
    --jobs 8
```

*Parallel Implementation (SGE)*

```
python scripts/model/estimator_train_schedule.py
```

#### Phase 4: Apply Trained Models and Compute Prevalence

1. *Vectorize the Unlabeled Data Samples*. This transforms everything into a document-term representation.


*Parallel (SGE Available)*

```
## Vectorize the Data (Using Phrasers Learned From Primary Embeddings)
python scripts/model/estimator_preprocess.py \
    ./configs/estimator/preprocess_template.json \
    --vectorize_users parallel \
    --jobs 1 \
    --grid_max_array_size 500 \
    --grid_memory_per_job 8
```    

*Serial (No SGE)*

```
python scripts/model/estimator_preprocess.py \
    ./configs/estimator/preprocess_template.json \
    --vectorize_users serial \
    --jobs 8
```

2. **Apply trained models** (each using different vocabularies) to vectorized, unlabeled data. As in the previous steps, you can run this step in parallel using SGE scheduler. Alternatively, you can use the scheduler to generate template config/execution files by setting `USE_SCHEDULER=False`.

*Serial (SGE)*
```
python scripts/model/estimator_apply.py \
    ./configs/estimator/apply_template.json \
    --run_predict \
    --run_analyze \
    --jobs 8
```

*Parallel (SGE)*
```
## Run Scheduler
python scripts/model/estimator_apply_schedule.py
```

*Preprocessing Parameters*

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

## Supplemental Analysis

We can generate some additional summary visualizations that look at relationships across the previous experiments.

### Between-Period Change

We can look at year-over-year change in semantics for a dataset using the utility script `scripts/model/util/semantic_vocabulary_selection.py`. In the paper, we use this script in conjunction with embeddings learned using data independently from 2019 and 2020 to look at year-over-year change in language correlated with the pandemic.

### Visualizations

We use code in `scripts/model/visualizations_v1.py` and `scripts/model/visualizations_v2.py` to generate visualizations for the camera-ready paper.

