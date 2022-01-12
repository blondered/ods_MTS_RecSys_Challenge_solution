import numpy as np
import pandas as pd
import pickle
from recsys_toolkit import *
from catboost import CatBoostClassifier

# Reading data
users_df = pd.read_csv('data/users_processed.csv')
items_df = pd.read_csv('data/items_processed.csv')
interactions_df = pd.read_csv('data/interactions_processed.csv', 
                              parse_dates=['last_watch_dt'])
submission = pd.read_csv('data/sample_submission.csv')
overall_known_items = interactions_df.groupby('user_id')['item_id'].apply(
    list).to_dict()

# Tools for checking and saving submission
def check_len_recs(recs, top_K = 10):
    '''
    Checks if all users have exactly top_K recs
    '''
    recs['len'] = recs['item_id'].apply(lambda x: len(x))
    print(f"{(recs['len'] > top_K).sum()} юзеров имеют рекомендации длиной более 10")
    print(f"{(recs['len'] < top_K).sum()} юзеров имеют рекомендации длиной менее 10")
    recs.drop('len', axis = 1, inplace = True)
    return 

def save_submission(recs_df, name = 'last_submission', to_string = False):
    '''
    Saves submission
    '''
    if to_string:
        recs_df['item_id'] = recs_df['item_id'].apply(lambda x: list(x))
    recs_df.to_csv(name + '.csv', index=False)
    
def to_string_func(x):
    '''
    Converts list to its string representation
    '''
    x = list(x)
    y = list(map(str, x))
    return '[' + (', ').join(y) + ']'
  
def check_submission(recs, known_items = overall_known_items):
    '''
    Checks resommendations for correct length and for duplicates
    in user history
    '''
    rec_items = recs.set_index('user_id').to_dict()['item_id']
    if type(recs.sample(1)['item_id']) == 'str':
        for key in rec_items:
            rec_items[key] = list(map(int, rec_items[key][1:-1].split(', ')))
    num_duplicated = 0
    for user, recommend in rec_items.items():
        if user in known_items:
            for item in recommend:
                if item in known_items[user]:
                    num_duplicated += 1
                    #users_have_duplicates_in_recs.add(user)
    print(f"Duplicated history items in recommendations: {num_duplicated}")
    return

with open("catboost_trained.pkl", 'rb') as f:
    boost_model = pickle.load(f)

# Constructing data for predictions
user_col = ['user_id', 
            'age', 
            'income', 
            'sex', 
            'kids_flg', 
            'user_watch_cnt_all', 
            'user_watch_cnt_last_14']
item_col = ['item_id', 
            'content_type', 
            'countries_max', 
            'for_kids', 
            'age_rating', 
            'studios_max', 
            'genres_max', 
            'genres_min', 
            'genres_med', 
            'release_novelty']
cat_col = ['age', 
           'income', 
           'sex', 
           'content_type']
warm_idx = np.intersect1d(submission['user_id'].unique(), 
                          interactions_df['user_id'].unique())
candidates = pd.read_csv('data/impl_scores_for_submit.csv', 
                         usecols = ['user_id', 'item_id', 'implicit_score'])
candidates.dropna(subset = ['item_id'], axis = 0, inplace = True)
submit_feat = candidates.merge(users_df[user_col],
                            on = ['user_id'],
                            how = 'left')\
                            .merge(items_df[item_col],
                                  on = ['item_id'],
                                  how = 'left')
full_train = submit_feat.fillna('None')
full_train[cat_col] = full_train[cat_col].astype('category')
item_stats = pd.read_csv('item_stats_for_submit.csv')
full_train = full_train.join(item_stats.set_index('item_id'), 
                             on = 'item_id', how = 'left')

# Renaming columns to match classifier feature names
cols = ['user_id', 'item_id']
cols.extend(boost_model.feature_names_)
cols = cols[:7] + ['user_watch_cnt_all', 'user_watch_cnt_last_14'] + cols[9:]
full_train = full_train[cols]
full_train_new_names = ['user_id', 'item_id'] + boost_model.feature_names_
full_train.columns = full_train_new_names

# Making predictions for warm users
y_pred_all = boost_model.predict_proba(full_train.drop(
    ['user_id', 'item_id'], axis = 1))
full_train['boost_pred'] = y_pred_all[:, 1]
full_train = full_train[['user_id', 'item_id', 'boost_pred']]
full_train = full_train.sort_values(by=['user_id', 'boost_pred'], 
                                    ascending=[True, False])
full_train['rank'] = full_train.groupby('user_id').cumcount() + 1
full_train = full_train[full_train['rank'] <= 10].drop('boost_pred', axis = 1)
full_train['item_id'] = full_train['item_id'].astype('int64')
boost_recs = full_train.groupby('user_id')['item_id'].apply(list)
boost_recs = pd.DataFrame(boost_recs)
boost_recs.reset_index(inplace = True)

# Making predictions for cold users with Popular Recommender
idx_for_popular = list(set(submission['user_id'].unique()).difference(
    set(boost_recs['user_id'].unique())))
pop_model = PopularRecommender(days=30, dt_column='last_watch_dt', 
                               with_filter = True)
pop_model.fit(interactions_df)
recs_popular = pop_model.recommend_with_filter(interactions_df, idx_for_popular, top_K=10)
all_recs = pd.concat([boost_recs, recs_popular], axis = 0)

# Filling short recommendations woth popular items
all_recs = fill_with_popular(all_recs, pop_model, interactions_df)

# Changing recommendations format and saving submission
all_recs['item_id'] = all_recs['item_id'].apply(to_string_func)
save_submission(all_recs, 'submit')
