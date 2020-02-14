# Code



This folder is organized as follows:



- collection/ -- Setup for data collection.
- classification/ -- Classifier based on n-grams.
  - classify_pipeline.py -- classification pipeline used in all experiments (except the open world).
  - classify_ow.py -- classification pipeline used in the open world experiment.
  - process_results.py -- script to process results.
  - pipeline
    - ngrams_classif.py -- ngrams feature extraction code.
    - tsfresh_basic.py -- tsfresh feature extraction code.
  - utils
    - utils.py -- useful functions used in the classification pipeline.
    - entropy.py -- functions used in the entropy calculations.