# ReSimNet
A Pytorch Implementation of paper
> ReSimNet: Drug Response Similarity Prediction using Siamese Neural Networks <br>
> Jeon and Park et al., 2018

## Abstract
Traditional drug discovery approaches identify a target for a disease and find a compound that binds to the target. In this approach, structures of compounds are considered as the most important features because it is assumed that similar structures will bind to the same target. Therefore, structural analogs of the drugs that bind to the target are selected as drug candidates. However, even though compounds are not structural analogs, they may achieve the desired response. A new drug discovery method based on drug response, which can complement the structure-based methods, is needed.

We implemented Siamese neural networks called ReSimNet that take as input two chemical compounds and predicts the CMap score of the two compounds, which we use to measure the transcriptional response similarity of the two counpounds. ReSimNet learns the embedding vector of a chemical compound in a transcriptional response space. ReSimNet is trained to minimize the difference between the cosine similarity of the embedding vectors of the two compounds and the CMap score of the two compounds. ReSimNet can find pairs of compounds that are similar in response even though they may have dissimilar structures. In our quantitative evaluation, ReSimNet outperformed the baseline machine learning models. The ReSimNet ensemble model achieves a Pearson correlation of 0.518 (p value of <10^-6) and a precision@1% of 0.989. In addition, in the qualitative analysis, we tested ReSimNet on the ZINC15 database and showed that ReSimNet successfully identifies chemical compounds that are relevant to a prototype drug whose mechanism of action is known.

## Pipeline
![Full Pipeline](/images/pipeline_updated_kang2.png)

## Requirements
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downlaods)
- Install [cudnn-v5.1](https://developer.nvidia.com/cudnn)
- Install [Pytorch 0.3.0](https://pytorch.org/)
- Install [Numpy 1.61.1](https://pypi.org/project/numpy/)
- Python version >= 3.4.3 is required

## Git Clone & Initial Setting
Clone our source codes and make folders to save data you need.

```bash
# clone the source code on your directory
$ git clone https://github.com/jhyuklee/ReSimNet.git
$ cd ReSimNet

# make folder to save and load your data
$ cd tasks
$ mkdir -p data

# make folder to save and load your model
cd ../../..
$ mkdir -p results
```

## Download Files You Need to Run ReSimNet

### Dataset for Training
- **[ReSimNet-Dataset.pkl](https://drive.google.com/file/d/1Q-vsozn1mL2b4QnhzC5izxuLoidkxAZ6/view?usp=sharing) (43MB)** <br>
*Save this file to ./ReSimNet/tasks/data/**ReSimNet-Dataset.pkl***

### Pre-Trained Models
- **[ReSimNet-model-best.zip](https://drive.google.com/file/d/1QAD_ftYu7eu-2ZeSGiVuu0P6tQ8sE8Vb/view?usp=sharing) (12MB)** <br>
*Save this file to ./ReSimNet/results/**ReSimNet-models-best.zip** and Unzip.*

### All 10 Models for Ensemble
- **[ReSimNet-models-ensenble.zip](https://drive.google.com/file/d/1SDgSaCiVOEXHHm-8ulJB18Ru6ETj8upf/view?usp=sharing) (117MB)** <br>
*Save this file to ./ReSimNet/results/**ReSimNet-model-ensemble.zip** and Unzip.*

### Example Input Pairs
- **[examples.csv](https://drive.google.com/file/d/1Vd7tikk8cZ5B_cDFqWX5Ou5yyYZ2r_CN/view?usp=sharing) (244byte)** <br>
*Save this file to ./ReSimNet/tasks/data/pairs/**examples.csv***

### Click the link ""Download the FingerPrint Respresentation"".
- **[pertid2fingerprint.pkl](https://drive.google.com/file/d/1r7kwmnRluaUDws1mOvvITn3EFfpyjnDX/view?usp=sharing) (10MB)** <br>
*Save this file to ./ReSimNet/tasks/data/**pertid2fingerprint.pkl***


## Training the ReSimNet

```bash
# Train for new model.
$ bash train.sh

# Train for the new ensemble models.
$ bast train_ensemble.sh
```

## CMap Score Prediction using ReSimNet
For your own fingerprint pairs, ReSimNet provides a predicted CMap score for each pair. Running download.sh and predict.sh will first download pretrained ReSimNet with sample datasets, and save a result file for predicted CMap scores.
```bash
# Save scores of sample pair data
$ bash predict_example.sh
```
Input Fingerprint pair file must be a .csv file in which every row consists of two columns denoting two Fingerprints of each pair. Please, place files under './tasks/data/pairs/'. 
```bash
# Sample Fingerprints (./tasks/data/pairs/examples.csv)
id1,id2
BRD-K43164539,BRD-A45333398
BRD-K83289131,BRD-K82484965
BRD-K06817181,BRD-A41112154
BRD-K06817181,BRD-K67977190
BRD-K06817181,BRD-A87125127
BRD-K68095457,BRD-K38903228
BRD-K68095457,BRD-K01902415
BRD-K68095457,BRD-K06817181
```
Predicted CMap scores will be saved at each row of a file './results/input-pair-file.model-name.csv'.
```bash
# Sample results (./results/examples.csv.ReSimNet7.csv')
prediction
0.9146181344985962
0.9301251173019409
0.8519644737243652
0.9631381034851074
0.7272981405258179
```
## CMap Score Prediction of ZINC using ReSimNet
```bash
# Save scores of sample pair data
$ bash predict_zinc.sh
```
### Click the link ""Download the ZINC files"".
- **[zinc-test.zip](https://drive.google.com/file/d/1YI2J81rahVWst3yU7741CVr8L-0Jc_NK/view?usp=sharing) (8KB)** <br>
*Save this file to ./ReSimNet/tasks/data/pairs_zinc/**zinc-test.zip** and unzip.*

```bash
# Sample Zinc files (./tasks/data/pairs_zinc/zinc-test/AABA.csv)
,smiles,zinc_id,inchikey,mwt,logp,reactive,purchasable,tranche_name,features,fingerprint
0,O=C1O[C@@H]([C@@H](O)CO)C(=O)C1=O,ZINC000001532646,SBJKKFFYIZUCET-VHUNDSFISA-N,174.108,-2.597,10,50,AACA,in-vitro,01000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000100000000000000000000000000000000000000010000000010000000000000000000000000000000000000100000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000010000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000001000000000000000000000000000000000
```

### Click the link ""Download the example pairings"".
- **[example_drugs.csv](https://drive.google.com/file/d/1Rok_oU8mwZbFMgYFJTx4i4NRAdz8oIg7/view?usp=sharing) (7KB)** <br>
*Save this file to ./ReSimNet/tasks/data/pairs_zinc/**example_drugs.csv***

```bash
# Sample example files (./tasks/data/pairs_zinc/example_drugs.csv)
pair,fp
ZINC18279871,00000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000100000000000000000000000000000000000000000000000000100000000000010010000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000010000000000000000000000000000000000000000000000000000000000000000000000100000000000100000000000000000000000000000000000010000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000001000000000000000000000000000000000000010000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000010000000001000000000000000000000000000000000000000000000001000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000001000000000000000001000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000100000000000001000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100001000000000000000000000000000001001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

Predicted CMap scores will be saved at each row of a file './results/input-pair-file.model-name.csv'.
```bash
# Sample results (./results/AACA.csv.ReSimNet7.csv')
pair1,pair2,prediction
ZINC000018204142,ZINC18279871,0.90729403
ZINC000018204142,ZINC3938668,0.91043824
```

## Liscense
Apache License 2.0
