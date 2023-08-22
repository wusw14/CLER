# Blocker and Matcher Can Mutually Benefit: A Co-Learning Framework for Low-Resource Entity Resolution

This is the source code of CLER. CLER is an end-to-end iterative Co-learning framework for ER, aimed at jointly training the blocker and the matcher by leveraging their cooperative relationship.

Our paper is submitted to VLDB 2024. 

![Image text](https://github.com/wusw14/CLER/blob/master/figs/CLER.png)
Figure (a) Illustration of the Co-learning between the blocker and the matcher in terms of information breadth and prediction accuracy. (b) The overview of our CLER framework in one training iteration, containing three steps (1) Data Annotation (2) Pseudo-labeling (3) Training.


## Requirements
### Create conda environment and install packages
The implementation requires python 3.7.  
```
conda create -n cler python==3.7
conda activate cler
``` 

All the packages except apex could be installed via "pip intall <package_name>".  
```  
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

For apex, we recommend to install via the following way:
```
git clone https://github.com/NVIDIA/apex.git 
python setup.py install
```

### Hardware environment
Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz  
NVIDIA A100 80GB  
Note: the experiments do not require the same hardware environment.

## Datasets
We conduct experiments on seven widely adopted public datasets from various domains for ER tasks. 
These datasets are obtained from the Magellan data repository and the Alaska benchmark. 
    
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
To reproduce the overall performance of our CLER method, run the run.py (including train.py and test_dynamic.py) for training the model and evaluating the overall performance on the test set with five different random seeds (0,1,2,3,4).  
You can modify the run.py to your own preferable setting. 

run.py only requires the dataset name, gpu id and annotation budget; all other hyperparamters are set to the values we used in our experiments.  
In our experiments, the budgets are in [500, 1000, 1500, 2000].    
```
python run.py <dataset> <gpu id> <annotation budget>  

# For example, to obtain the overall performance of the Fodors-Zagats (FZ) dataset under the annotation budget 500, excute the following command:
python run.py FZ 0 500
```

### Training
If you only want to conduct the training process, you can conduct the following command. In this way, the trained blocker and matcher would be saved and could be used for further evaluation.
```
python -u train.py --fp16 --save_model \
                   --lr <learning_rate> \
                   --total_budget <annotation budget> \
                   --gpu <gpu id> \
                   --dataset <dataset> \
                   --run_id <random_seed> \
                   --batch_size <batch_size> \
                   --topK <K, the number of most similar entries retrieved for each left entry during the training>  

# For example, training the blocker and the matcher on the FZ dataset under annotation budget = 500 and random seed = 0.
python -u train.py --fp16 --save_model \
                   --lr 1e-5 \
                   --total_budget 500 \
                   --gpu 0 \
                   --dataset FZ \
                   --run_id 0 \
                   --batch_size 64 \
                   --topK 10  
```

### Evaluation
To evaluate the overall performance of the ER model, including the blocking and the matching steps
``` 
python -u test_dynamic.py --fp16 \
                          --total_budget <annotation_budget> \
                          --gpu <gpu id> \
                          --dataset <dataset> \
                          --topK <K, the same hyperparamter as used in the training> \
                          --run_id <random_seed>

# For example, given the trained blocker and the matcher, evaluate the overall performance by dynamic inference strategy on the FZ dataset under annotation budget = 500 and random seed = 0.
python -u test_dynamic.py --fp16 --total_budget 500 --gpu 0 --dataset FZ --topK 10 --run_id 0
```

To evaluate the blocker
```
# the output would be the recall values corresponding to retrieve [1,2,5,10,20,50] entries for each entry in the test set

python -u eval_blocker.py --dataset <dataset> \
                          --topK <K, the same hyperparamter as used in the training> \
                          --total_budget <annotation budget> \
                          --run_id <random_seed>

# For example, evaluate the trained blocker on the FZ dataset under annotation budget = 500 and random seed = 0.
python -u eval_blocker.py --dataset FZ --topK 10 --total_budget 500 --run_id 0
```

To evaluate the matcher on the processed megallan dataset (excluding the impact of the blocker)
```
python -u test_magellan.py --fp16 \
                           --total_budget <annotation budget> \
                           --gpu <gpu id> \
                           --dataset <dataset> \
                           --topK <K, the same hyperparamter as used in the training> \
                           --run_id <random_seed>

# For example, evaluate the trained matcher on the processed Magellan FZ dataset under annotation budget = 500 and random seed = 0.
python -u test_magellan.py --fp16 --total_budget 500 --gpu 0 --dataset FZ --topK 10 --run_id 0
```

## Contact Information
If you have any questions or feedback about this project, please feel free to contact us. We highly appreciate your suggestions!

Email: swubs@connect.ust.hk  
GitHub Issues: For more technical inquiries, you can also create a new issue in our GitHub repository.  
We will respond to all questions as soon as possible.

## Acknowledgements
This repository is developed based on https://github.com/megagonlabs/ditto  

Our implementation of baselines is based on these public repositories:
* [CollaborEM](https://github.com/ZJU-DAILY/CollaborEM)
* [DITTO](https://github.com/megagonlabs/ditto)
* [Avanika, et al.](https://github.com/HazyResearch/fm_data_tasks) 
* [DeepBlock](https://github.com/qcri/DeepBlocker)
* [Sudowoodo](https://github.com/megagonlabs/sudowoodo)

Thanks for the authors' great contributions!