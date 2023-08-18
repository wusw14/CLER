# Blocker and Matcher Can Mutually Benefit: A Co-Learning Framework for Low-Resource Entity Resolution

CLER is an end-to-end iterative Co-learning framework for ER,
aimed at jointly training the blocker and the matcher by leveraging
their cooperative relationship. In particular, we let the blocker and
the matcher share their learned knowledge with each other via
iteratively updated pseudo labels, which broaden the supervision
signals. To mitigate the impact of noise in pseudo labels, we develop
optimization techniques from three aspects: label generation, label
selection and model training.

## Requirements
python==3.7   
torch   
pandas   
scikit-learn   
packaging   
urllib3==1.26.6   
importlib-metadata   
sentence_transformers   
apex (git clone https://github.com/NVIDIA/apex.git; python setup.py install)  

## Datasets
The datasets used in this work can be downloaded from this link.  
https://drive.google.com/drive/folders/1ZnGLUpYFZSC9Ru8HKFCrTthM--1aBqD-?usp=drive_link   
```
dataset_dict = {
    'AG': 'Amazon-Google',\
    'DA': 'DBLP-ACM',\
    'DS': 'DBLP-Scholar',\
    'FZ': 'Fodors-Zagats',\
    'WA': 'Walmart-Amazon',\
    'AB': 'Abt-Buy',
    'M': 'monitor',
}
```


## Training and Evaluation
```
# run the train.py and test.py for training the model and evaluating on the test set with different random seeds.  
python run.py <dataset> <gpu id> <annotation budget>  
e.g., python run.py FZ 0 500

# for training
python -u train.py --fp16 --lr 1e-5 --total_budget 500 --gpu 0 --dataset FZ --run_id 0 --batch_size 64 --save_model --topK 10  
# for dynamic inference evaluation
python -u test_dynamic.py --fp16 --total_budget 500 --gpu 0 --dataset FZ --topK 10 --run_id 0
# for evaluation on the processed megallan dataset
python -u test_magellan.py --fp16 --total_budget 500 --gpu 0 --dataset FZ --topK 10 --run_id 0
```


## Acknowledgements
This repository is developed based on https://github.com/megagonlabs/ditto