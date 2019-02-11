#!/bin/bash
# predict pair scores when given with two input drud_ids.
# calculate prediction scores based on averged scores of all 10 models.
# if you do not want this, set --save-pair-score-ensemble to false
CUDA_VISIBLE_DEVICES=1 python main.py --save-pair-score true --save-pair-score-ensemble true --pair-dir './tasks/data/post-analysis/zinc/' --fp-dir './tasks/data/pertid2fingerprint.pkl' --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'ReSimNet.mdl' --rep-idx 2
