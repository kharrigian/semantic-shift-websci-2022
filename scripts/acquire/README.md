# Data Acquisition

Scripts for the acquisition of data on which our models are trained and evaluated. Annotated datasets are available from their respective authors under appropriate data usage agreements. We provide sources where available below.

## Labeled Datasets

#### Twitter

* `clpsych`: "CLPsych 2015 Shared Task: Depression and PTSD on Twitter". For more information, contact Mark Dredze (mdredze@cs.jhu.edu).
* `multitask`: "Multi-Task Learning for Mental Health using Social Media Text". For more information, contact Glen Coppersmith (glen@qntfy.com).

#### Reddit

* `smhd`: "SMHD: A Large-Scale Resource for Exploring Online Language Usage for Multiple Mental Health Conditions". For more infroatmion, see http://ir.cs.georgetown.edu/resources/smhd.html.
* `wolohan`: "Detecting Linguistic Traces of Depression in Topic-Restricted Text: Attenting to Self-Stigmatized Depression in NLP". For more information, see `scripts/acquire/0_get_wolohan.py`

## Unlabeled Datasets

#### Twitter

* `gardenhose`: 1% Twitter data sample available on the JHU CLSP grid. File location is `/export/c12/mdredze/twitter/public/`. Raw data files are in JSON format. They can be preprocessed into the standardized modeling framework using `scripts/preprocess/preprocess.py`. The processed format should be stored in `data/processed/twitter/gardenhose/`

#### Reddit

* `active`: Sample of approximately 50k users from Reddit who were active at the beginning of 2019 and at the end of May 2020. Scripts are provided in order within `scripts/acquire/reddit/active/` subdirectory. To preprocess the data, use `scripts/preprocess/preprocess.py`, providing the appropriate filepath and choosing the "reddit" option.