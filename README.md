# README #

Please follow the instructions in requirement.txt to download required packages

### Training ###
There are many options to choose from :
1. Enable data augmentation
2. Choose to classify coarse / fine classes
3. Choose different models (SVC/ LinearSVC / RandomForestClassifier/ SGDClassifier)
4. Choose to enable different features (stemmer / lemma)
5. Choose max_iters in training models
6. Enable plotting of learning curve

## Example: ##
    ```python train.py --use_uniq_only --use_stemmer --label_type coarse --model_name SVC --augmentdata```
