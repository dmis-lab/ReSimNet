#!/bin/bash
# predict pair scores when given with two input drud_ids.
# calculate prediction scores based on averged scores of all 10 models.
# if you do not want this, set --save-pair-score-ensemble to false
CUDA_VISIBLE_DEVICES=1 python main.py --save-pair-score true --save-pair-score-zinc true --pair-dir './tasks/data/pairs_zinc/zinc-test/' --example-dir './tasks/data/pairs_zinc/example_drugs.csv' --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'ReSimNet7.mdl' --rep-idx 2
