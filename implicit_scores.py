import random
import datetime
import numpy as np
import scipy.sparse as sp
import pandas as pd
from itertools import islice, cycle
from more_itertools import pairwise
from implicit import nearest_neighbours as NN
import implicit
from recsys_toolkit import *

def get_implicit_candidates(full_train, overall_known_items_mapped, warm_idx):
    '''
    Calculates top candidates from implicit model with their scores.
    Implicit parameters were chosen on time range split cross-validation.
    History offset stands for taking only lask X items from user history.
    Day offset stands for taking items from last X days of user history.
    '''
    k_neighbours = 200
    day_offset = 170
    history_offset = 20
    distance = 'Cosine'
    num_candidates = 100
    full_train['order_from_recent'] = full_train.sort_values(
      by = ['last_watch_dt'], ascending = False).groupby('user_id').cumcount() + 1
    train = full_train.copy()
    date_window = train['last_watch_dt'].max() - pd.DateOffset(days=day_offset)
    train = train[train['last_watch_dt'] >= date_window]
    if history_offset:
        train = train[train['order_from_recent'] < history_offset]
    if distance == 'Cosine':
        model = NN.CosineRecommender(K = k_neighbours)
        weights = None
    else:
        model = NN.TFIDFRecommender(K = k_neighbours)
        weights = None
    train_mat = get_coo_matrix(
        train,
        users_mapping=users_mapping,
        items_mapping=items_mapping,
        weight_col = weights
    ).tocsr()
    model.fit(train_mat.T, show_progress=False) 
    mapper = generate_implicit_recs_mapper( 
        model,
        train_mat,
        num_candidates,
        users_mapping,
        items_inv_mapping,
        False,
        known_items = overall_known_items_mapped, 
        filter_items = None,
        return_scores = True 
    )
    recs = pd.DataFrame({'user_id': warm_idx})
    recs['item_id_score'] = recs['user_id'].map(mapper)
    recs['item_id'] = recs['item_id_score'].apply(lambda x: x[0])
    recs['implicit_score'] = recs['item_id_score'].apply(lambda x: x[1])
    recs = recs.explode(column = ['item_id', 'implicit_score'])
    recs.drop(['item_id_score'], axis = 1, inplace = True)
    return recs


users_df = pd.read_csv('data/users_processed.csv',)
items_df = pd.read_csv('data/items_processed.csv',)
interactions_df = pd.read_csv('data/interactions_processed.csv', 
                              parse_dates=['last_watch_dt'])
submission = pd.read_csv('data/sample_submission.csv')
interactions_df.sort_values(by = 'last_watch_dt', inplace = True)

# Creating items and users mapping
users_inv_mapping = dict(enumerate(interactions_df['user_id'].unique()))
users_mapping = {v: k for k, v in users_inv_mapping.items()}
items_inv_mapping = dict(enumerate(interactions_df['item_id'].unique()))
items_mapping = {v: k for k, v in items_inv_mapping.items()}

# Preparing data for implicit scores for submit
overall_known_items = interactions_df.groupby('user_id')['item_id'].apply(
    list).to_dict()
overall_known_items_mapped = {}
for user, recommend in overall_known_items.items():
    overall_known_items_mapped[user] = list(map(lambda x: 
                                                items_mapping[x], recommend))
interactions_df['order_from_recent'] = interactions_df.sort_values(
      by = ['last_watch_dt'], ascending = False).groupby('user_id').cumcount() + 1
warm_idx = np.intersect1d(interactions_df['user_id'].unique(), 
                          submission['user_id'].unique())

# Preparing data for implicit scores for boosting training
last_date_df = interactions_df['last_watch_dt'].max()
boosting_split_date = last_date_df - pd.Timedelta(days = 14)
boosting_data = interactions_df[(interactions_df['last_watch_dt'] > \
                                 boosting_split_date)].copy() 
before_boosting = interactions_df[(interactions_df['last_watch_dt'] <= \
                                   boosting_split_date)].copy()
before_boosting_known_items = before_boosting.groupby(
    'user_id')['item_id'].apply(list).to_dict()
before_boosting_known_items_mapped = {}
for user, recommend in before_boosting_known_items.items():
    before_boosting_known_items_mapped[user] = list(map(lambda x: 
                                                        items_mapping[x], 
                                                        recommend))
before_boosting['order_from_recent'] = before_boosting.sort_values(
    by = ['last_watch_dt'], ascending = False).groupby('user_id').cumcount() + 1
boost_warm_idx = np.intersect1d(before_boosting['user_id'].unique(), 
                                boosting_data['user_id'].unique())

# Getting implicit scores and saving to csv
impl_recs_for_submit =  get_implicit_candidates(interactions_df, 
                                                overall_known_items_mapped, warm_idx)
impl_recs_for_submit.to_csv('data/impl_scores_for_submit.csv', index = False)

impl_recs_for_boost_train =  get_implicit_candidates(before_boosting, 
                                                     before_boosting_known_items_mapped, 
                                                     boost_warm_idx)
impl_recs_for_boost_train.to_csv('data/impl_scores_for_boost_train.csv', index = False)
