#! /bin/bash

# bash src/get_data.sh
# python src/data/preprocess.py 

python src/features/add_item_stats.py data/interim/interactions_clean.csv data/interim/items_clean.csv data/interim/users_clean.csv data/interim/items_w_stats_for_train.csv data/interim/items_w_stats_for_submit.csv

# python src/data/featurize.py

# python implicit_scores.py
# python boosting_train.py
# python submit_prediction.py
