#!/bin/bash
# Train with Best Model
# 0 : smiles, 1: inchikey, 2: ecfp, 3: mol2vec
python main.py --data-path './tasks/data/ReSimNet-Dataset.pkl' --model-name 'trained_model.mdl' --rep-idx 2
