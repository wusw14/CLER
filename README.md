# Blocker and Matcher Can Mutually Benefit: A Co-Learning Framework for Low-Resource Entity Resolution

CLER is an end-to-end iterative Co-learning framework for ER,
aimed at jointly training the blocker and the matcher by leveraging
their cooperative relationship. In particular, we let the blocker and
the matcher share their learned knowledge with each other via
iteratively updated pseudo labels, which broaden the supervision
signals. To mitigate the impact of noise in pseudo labels, we develop
optimization techniques from three aspects: label generation, label
selection and model training.

![Image text](https://github.com/wusw14/CLER/blob/master/figs/CLER.png)
Figure (a) Illustration of the Co-learning between the blocker and the matcher in terms of information breadth and prediction accuracy. The blocker learns from the matcher's precise classification ability while the matcher learns from the blocker's global view of the similarity ranking. The gray arrows represent the data flow.
(b) The overview of our CLER framework in one training iteration, containing three steps (1) Data Annotation: The blocker (BK) first produces a candidate set $C$ from all pairs of entities $(e, e')$ where $e\in D$ and $e' \in D'$. The matcher (MC) then generates scores for each candidate, which are used to select informative examples to be annotated.
(2) Pseudo-labeling: The blocker and the matcher generate pseudo labels for $C$ separately. The generated ones are further processed into two sets feeding the blocker and the matcher, respectively.
(3) Training: Both the annotated data $S_{annot}$ and the pseudo-labeled data are utilized for training the blocker and the matcher.

## Requirements
The implementation requires python 3.7. All the packages except apex could be installed via "pip intall <package_name>".  
For apex, we recommend to install via this command "git clone https://github.com/NVIDIA/apex.git; python setup.py install".
```
python==3.7   
torch   
pandas   
scikit-learn   
packaging   
urllib3==1.26.6   
importlib-metadata   
sentence_transformers 
transformers  
apex
```


## Datasets
we conduct experiments on seven widely adopted public datasets from various domains for ER tasks. 
These datasets are obtained from the Magellan data repository and the Alaska benchmark. 
A summary of the dataset statistics can be found in Table 1.
    
| Dataset  | \# entries $D,D'$ | \# matches | (%) matches 
| :----: | :----: | :----: | :----: |
| Amazon-Google (AG) | 1363, 3226 | 1300 | 0.0296 
| DBLP-ACM (DA) | 2616, 2294 | 2224 | 0.0371 
| DBLP-Scholar (DS) | 2616, 64263 | 5347 | 0.0032 
| Fodors-Zagats (FZ) | 533, 331 | 112 | 0.0635 
| Walmart-Amazon (WA) | 2554, 22074 | 1154 | 0.0020
| Abt-Buy (AB) | 1081, 1092 | 1098 | 0.0930 
| Monitor (M) | 603, 4323 | 343 | 0.0132 

The datasets used in this work can be downloaded from this [link](https://drive.google.com/drive/folders/1ZnGLUpYFZSC9Ru8HKFCrTthM--1aBqD-?usp=sharing).  


## Training and Evaluation
Run the run.py (including train.py and test_dynamic.py) for training the model and evaluating the overall performance on the test set with five different random seeds.  
run.py only requires the dataset name, gpu id and annotation budget; all other hyperparamters are set to the values we used in our experiments.
```
python run.py <dataset> <gpu id> <annotation budget>  

e.g., python run.py FZ 0 500
```

### Training
If you only want to conduct the training process, you can conduct the following command. In this way, the trained blocker and matcher would be saved and could be used for further evaluation.
```
python -u train.py --fp16 --lr <learning_rate> --total_budget <annotation budget> --gpu <gpu id> --dataset <dataset> --run_id <random_seed> --batch_size <batch_size> --save_model --topK <K, the number of most similar entries retrieved for each left entry during the training>  

e.g., python -u train.py --fp16 --lr 1e-5 --total_budget 500 --gpu 0 --dataset FZ --run_id 0 --batch_size 64 --save_model --topK 10  
```

### Evaluation
To evaluate the overall performance of the ER model, including the blocking and the matching steps
``` 
python -u test_dynamic.py --fp16 --total_budget <annotation_budget> --gpu <gpu id> --dataset <dataset> --topK <K> --run_id <random_seed>

e.g., python -u test_dynamic.py --fp16 --total_budget 500 --gpu 0 --dataset FZ --topK 10 --run_id 0
```

To evaluate the blocker
```
# the output would be the recall values corresponding to retrieve [1,2,5,10,20,50] entries for each entry in the test set

python -u eval_blocker.py --dataset <dataset> --topK <K, the same hyperparamter as used in the training> --total_budget <annotation budget> --run_id <random_seed>

e.g., python -u eval_blocker.py --dataset FZ --topK 10 --total_budget 500 --run_id 0
```

To evaluate the matcher on the processed megallan dataset
```
python -u test_magellan.py --fp16 --total_budget <annotation budget> --gpu <gpu id> --dataset <dataset> --topK <K, the same hyperparamter as used in the training> --run_id <random_seed>

e.g., python -u test_magellan.py --fp16 --total_budget 500 --gpu 0 --dataset FZ --topK 10 --run_id 0
```

## Contact Information
If you have any questions or feedback about this project, please feel free to contact us. We highly appreciate your suggestions!

Email: swubs@connect.ust.hk  
GitHub Issues: For more technical inquiries, you can also create a new issue in our GitHub repository.  
We will respond to all questions as soon as possible.

## Acknowledgements
This repository is developed based on https://github.com/megagonlabs/ditto 