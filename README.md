# ods and MTS RecSys Course Competition solution

## Requirements
--------
- Python 3
- NumPy
- Pandas
- Scipy
- Sklearn
- Implicit
- Catboost

## Dataset
You can use the following script to download competition dataset:
```
./get_data.sh
```
## Solution description
My solution included a two-stage model. I used item CF from implicit library to generate candidates with their scores and Catboost classifier to predict final ranks. Recommendations for cold users were made with Popular items.

## Feature importance
![Catboost feature importance](https://github.com/blondered/ods_MTS_RecSys_Challenge_solution
/raw/master/pics/feature_importance.png)
