#! /bin/bash

RAW_DATA_DIR="data/raw"
INTERIM_DATA_DIR="data/interim"

mkdir -p "$RAW_DATA_DIR"
mkdir -p "$INTERIM_DATA_DIR"

bash src/get_data.sh $RAW_DATA_DIR
python src/data/preprocess.py $RAW_DATA_DIR $INTERIM_DATA_DIR
python src/data/featurize.py $RAW_DATA_DIR $INTERIM_DATA_DIR

# python implicit_scores.py
# python boosting_train.py
# python submit_prediction.py
