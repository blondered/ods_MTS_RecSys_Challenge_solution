#! /bin/bash

bash get_data.sh
python preprocess.py
python implicit_scores.py
python boosting_train.py
python submit_prediction.py
