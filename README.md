# The Problem of Semantic Shift in Longitudinal Monitoring of Social Media

Experimental code in support of "The Problem of Semantic Shift in Longitudinal Monitoring of Social Media." To be presented In Proceedings of the 14th ACM Web Science Conference. If you make use of this code for your own analysis, please consider using the following reference citation:

```
@inproceedings{harrigian2022problem,
  title={The Problem of Semantic Shift in Longitudinal Monitoring of Social Media: A Case Study on Mental Health During the COVID-19 Pandemic},
  author={Harrigian, Keith and Dredze, Mark},
  booktitle={14th ACM Web Science Conference 2022},
  pages={208--218},
  year={2022}
}
```

## Contribution

The paper provides two primary contributions. 

1. We introduce modifications to an existing semantic-shift detection algorithm so that it may be used to generate a semantically-stable vocabulary.
2. We demonstrate that longitudinal measurements of population-level depression are highly sensitive to the underlying vocabulary (moreso when the vocabulary is selected using semantic awareness).

## Semantically-Aware Vocabulary Selection

The general algorithm for semantically-aware feature/vocabulary selection is as follows.

1. Fit word embeddings using source domain data (e.g. historical data).
2. Fit word embeddings using target domain data (e.g. present data). 
3. For each word in the shared vocabulary of the domains, identify the neighborhood of most similar terms (size *K*) in each embedding space (i.e. using vectory similarity)
4. For each word in the shared vocabulary, compute semantic stability as the number of terms that exist in the intersection of the neighborhoods from both embedding spaces, divided by the neighborhood size K.
5. Select the top-M features with the highest semantic stability scores.

*Implementation Note:* In theory, feel free to choose any methodology for learning word embeddings. In this work, we use Word2Vec as implemented in the `gensim` python library.

## Usage

For a comprehensive overview of the code in the repository, please read the README files located in `scripts/`, `scripts/acquire/`, `scripts/preprocess/` and `scripts/model/`.

Much of the codebase relies on access to the appropriate datasets. The README files describe where this data can be accessed. If you have any issues acquiring the data or formatting it for use with this code base, please feel free to reach out to Keith Harrigian at kharrigian@jhu.edu.

Once data has been acquired, you should install the library of functions (`semshift`) that serves as the foundation for the experiments presented in the paper. This can be done simply using `pip install -e .` from the root of the repository.

## Questions/Comments

For any additional questions, please feel free to reach out. This repository has been translated from an internal research repository and as such may be subject to small bugs. If you find anything out of the ordinary, please consider submitting a pull-request.

From a personal perspective, this codebase is overly-heavy due to the experimental procedures. If you are simply interested in applying semantically-aware feature selection, it is probably more efficient to implement it from scratch (e.g., using the general algorithm above).
