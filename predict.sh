#!/bin/bash
# predict pair scores when given with two input drud_ids.
# calculate prediction scores based on averged scores of all 10 models.
# if you do not want this, set --save-pair-score-ensemble to false
python main.py --save-pair-score true --pair-dir './tasks/data/pairs/' --fp-dir './tasks/data/pertid2fingerprint.pkl' --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'ReSimNet7.mdl' --rep-idx 2
